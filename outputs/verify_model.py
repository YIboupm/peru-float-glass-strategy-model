"""Evidence-linked Peru glass model. Python 3.10+, numpy, scipy.

Run: python verify_model.py
Inputs: public_data.json, construction_history.json, model_assumptions.json.
Outputs: computed_results.json, computed_tables.md.
No network requests. Re-solves a menu MILP, not a statistical causal model.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT.parent / 'work' / 'python_deps'
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

DATA = read('public_data.json')
ASSUMPTIONS = read('model_assumptions.json')
C = {int(y): v for y, v in read('construction_history.json')['values'].items()}

def rows(iso, year=2024):
    return DATA[f'{iso}_{year}']['response']['data']

def record(iso, code, flow='M', year=2024):
    return next((r for r in rows(iso,year) if r['cmdCode']==code and r['flowCode']==flow),None)

def pools(a):
    result = {}
    for iso,m in a['markets'].items():
        tons = value = 0.0
        components = []
        codes = ['700521','700510'] if iso=='COL' else ['700529','700521','700510']
        for code in codes:
            r = record(iso,code)
            w = r['netWgt']
            method = 'Comtrade estimated net weight' if r['isNetWgtEstimated'] else 'Comtrade net weight'
            if w is None:
                assert r['qtyUnitCode']==2 and r['qty']>0
                w = r['qty']*a['missing_clear_glass_thickness_mm']*2.5
                method = f"area x assumed {a['missing_clear_glass_thickness_mm']}mm x 2.5kg/m2/mm"
            tons += w / 1000
            value += r['primaryValue']
            components.append({'hs':code,'import_t':w/1000,'cif_usd':r['primaryValue'],
                               'method':method,'quantity_estimated':r['isQtyEstimated']})
        gross_t = tons
        # Peru: original float production negligible before project; net exports only.
        # Other countries: gross import opportunity, not apparent consumption.
        if iso=='PER':
            tons -= sum(record(iso,c,'X')['netWgt']/1000 for c in codes)
        result[iso] = {'pool_t':tons,'gross_import_t':gross_t,
                       'included_cif_usd':value,'cif_unit_value':value/gross_t,
                       'components':components}
    return result

def regression_audit():
    valid = {}
    missing = []
    for year in range(2012,2025):
        r = record('PER','7005','M',year)
        e = record('PER','7005','X',year)
        if r is None or r['netWgt'] is None or e is None or e['netWgt'] is None:
            missing.append(year)
            continue
        valid[year] = (r['netWgt']-e['netWgt'])/1000
    def ols(x,y):
        X = np.column_stack([np.ones(len(x)),x])
        b = np.linalg.lstsq(X,y,rcond=None)[0]
        resid = y-X@b
        sst = sum((y-np.mean(y))**2)
        return b, float(1-sum(resid**2)/sst) if sst else 0
    output = []
    for mode in ['log_level','log_growth']:
        for lag in range(3):
            yy = [y for y in valid if y-lag in C]
            if mode=='log_growth':
                yy = [y for y in yy if y-1 in valid and y-lag-1 in C]
            if mode=='log_level':
                xx = np.array([math.log(C[y-lag]) for y in yy])
                target = np.array([math.log(valid[y]) for y in yy])
            else:
                xx = np.array([math.log(C[y-lag]/C[y-lag-1]) for y in yy])
                target = np.array([math.log(valid[y]/valid[y-1]) for y in yy])
            b,r2 = ols(xx,target)
            errors=[]; naive=[]; testyears=[]
            for k,y in enumerate(yy):
                if y<2021 or k<4 or y-1 not in valid:
                    continue
                bb,_=ols(xx[:k],target[:k])
                pred=math.exp(bb[0]+bb[1]*xx[k])
                if mode=='log_growth': pred*=valid[y-1]
                errors.append(abs(pred/valid[y]-1))
                naive.append(abs(valid[y-1]/valid[y]-1))
                testyears.append(y)
            output.append({'mode':mode,'lag':lag,'n':len(yy),'elasticity':float(b[1]),
                           'r_squared':r2,'conditional_holdout_years':testyears,
                           'holdout_mape':float(np.mean(errors)) if errors else None,
                           'naive_mape':float(np.mean(naive)) if naive else None})
    return {'valid_net_import_t':valid,'missing_weight_years':missing,'models':output,
            'note':'Small, incomplete annual sample; estimates flagged upstream included. Holdouts use realized construction values, so these are conditional tests, not real-time forecasts. No coefficient is deployed as causal elasticity.'}

def solve(a, demand_scale=1, price_scale=1, freight_scale=1, volume_target=None, fixed_strategy=None):
    pool = pools(a)
    years = a['years']; T = len(years)
    cap = np.array(a['capacity_factors'])*a['design_capacity_t']
    disc = np.array([(1+a['discount_rate'])**-(t+1) for t in range(T)])
    menu = []
    for i,(iso,m) in enumerate(a['markets'].items()):
        ref = (pool[iso]['cif_unit_value']+m['benchmark_inland'])*price_scale
        for ch,c in a['channels'].items():
            for discount in a['price_discount_grid']:
                price = ref*(1-discount)
                margin = price-a['variable_factory_cost_usd_t']-m['freight']*freight_scale-m['other_delivery']-c['variable_cost']
                # Scenario tariffs charged on a simplified pre-duty delivery basis.
                margin -= m['tariff']*price/(1+m['tariff'])
                for entry in ([0] if iso=='PER' else range(T)):
                    vmax = np.zeros(T)
                    target=m['smax']*c['share_multiplier']/(1+math.exp(-a['beta']*(discount-a['switch_threshold'])))
                    for t,y in enumerate(years):
                        if t<entry: continue
                        age=t-entry; lam=c['lambda']
                        avg_decay=math.exp(-lam*age)*(1-math.exp(-lam))/lam
                        share=target+(m['initial_share']-target)*avg_decay
                        demand=pool[iso]['pool_t']*(1+a['growth'])**(y-2024)*m['phi']*demand_scale
                        vmax[t]=max(0,share*demand)
                    fixed = sum(c['annual_fixed']*disc[t] for t in range(entry,T))
                    fixed += c['entry_cost']/(1+a['discount_rate'])**entry
                    menu.append({'iso':iso,'i':i,'channel':ch,'entry':entry,'discount':discount,
                                 'price':price,'margin':margin,'vmax':vmax,'fixed_npv':fixed})
    N=len(menu); nvar=N*(1+T)
    objective=np.zeros(nvar); objective[:N]=[k['fixed_npv']/1e6 for k in menu]
    upper=np.full(nvar,np.inf); upper[:N]=1
    lower=np.zeros(nvar)
    if fixed_strategy is not None:
        fixed_keys={(x['iso'],x['channel'],x['entry'],x['discount']) for x in fixed_strategy}
        for k,op in enumerate(menu):
            match=(op['iso'],op['channel'],op['entry'],op['discount']) in fixed_keys
            upper[k]=1 if match else 0
            lower[k]=1 if match else 0
    integral=np.zeros(nvar); integral[:N]=1
    rr=[]; cc=[]; vv=[]; ub=[]; lb=[]
    def constraint(items, low=-np.inf, high=0):
        row=len(ub)
        for col,val in items:
            if val: rr.append(row); cc.append(col); vv.append(val)
        ub.append(high); lb.append(low)
    for i in range(len(a['markets'])):
        constraint([(k,1) for k,op in enumerate(menu) if op['i']==i],high=1)
    for t in range(T):
        constraint([(N+k*T+t,1) for k in range(N)],high=cap[t]/1000)
    for k,op in enumerate(menu):
        for t in range(T):
            idx=N+k*T+t
            objective[idx]=-op['margin']*disc[t]/1000
            upper[idx]=op['vmax'][t]/1000
            constraint([(idx,1),(k,-op['vmax'][t]/1000)])
    if volume_target is not None:
        constraint([(N+k*T+T-1,1) for k in range(N)],low=cap[-1]*volume_target/1000,high=np.inf)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(ub),nvar)).tocsc()
    result=milp(objective,integrality=integral,bounds=Bounds(lower,upper),
                constraints=LinearConstraint(matrix,np.array(lb),np.array(ub)),
                options={'time_limit':40,'mip_rel_gap':0.0001})
    if result.x is None:
        return {'status':int(result.status),'message':result.message,'feasible':False}
    chosen=[]; totals=np.zeros(T); yearly=np.zeros(T)
    for k,op in enumerate(menu):
        if result.x[k]<0.5: continue
        sales=result.x[N+k*T:N+(k+1)*T]*1000
        totals+=sales
        c=a['channels'][op['channel']]
        annual=sales*op['margin']
        annual[op['entry']:]-=c['annual_fixed']
        annual[op['entry']]-=c['entry_cost'] # Undiscounted cash summary; entry paid start-year for NPV.
        yearly+=annual
        chosen.append({k:v for k,v in op.items() if k not in ['vmax','i']} | {
            'entry_year':years[op['entry']], 'sales_t':sales.tolist(),'potential_t':op['vmax'].tolist(),
            'annual_contribution_usd':annual.tolist()})
    # Independent feasibility / accounting checks, tolerance in tonnes.
    assert np.max(totals-cap)<0.1
    assert all(min(x['sales_t'])>=-0.1 for x in chosen)
    assert len({x['iso'] for x in chosen})==len(chosen)
    for x in chosen:
        assert all(v<=q+0.1 for v,q in zip(x['sales_t'],x['potential_t']))
        assert all(abs(x['sales_t'][t])<0.1 for t,y in enumerate(years) if y<x['entry_year'])
    check=sum(sum(v*op['margin']*d for v,d in zip(op['sales_t'],disc))-op['fixed_npv'] for op in chosen)
    assert abs(check+result.fun*1e6)<1
    return {'feasible':True,'status':int(result.status),'message':result.message,
            'mip_gap':float(result.mip_gap),'npv_contribution_usd':float(-result.fun*1e6),
            'sales_t':totals.tolist(),'capacity_t':cap.tolist(),'utilization':(totals/cap).tolist(),
            'annual_contribution_usd':yearly.tolist(),'selected':chosen}

def main():
    audit=regression_audit(); p=pools(ASSUMPTIONS)
    runs={}
    runs['base']=solve(ASSUMPTIONS)
    print('base',round(runs['base']['npv_contribution_usd']/1e6,3),flush=True)
    for key,kwargs in [('price_minus_20',{'price_scale':0.8}),('price_plus_20',{'price_scale':1.2}),
                       ('freight_minus_20',{'freight_scale':0.8}),('freight_plus_20',{'freight_scale':1.2}),
                       ('demand_minus_20',{'demand_scale':0.8}),('demand_plus_20',{'demand_scale':1.2})]:
        runs[key]=solve(ASSUMPTIONS,**kwargs)
        print(key,round(runs[key].get('npv_contribution_usd',0)/1e6,3),flush=True)
    for key,field,val in [('threshold_minus_20','switch_threshold',0.064),('threshold_plus_20','switch_threshold',0.096),
                          ('beta_minus_20','beta',20),('beta_plus_20','beta',30),
                          ('factory_cost_minus_20','variable_factory_cost_usd_t',192),
                          ('factory_cost_plus_20','variable_factory_cost_usd_t',288),
                          ('growth_zero','growth',0),('growth_six_pct','growth',0.06),
                          ('thickness_4mm','missing_clear_glass_thickness_mm',4),
                          ('thickness_6mm','missing_clear_glass_thickness_mm',6)]:
        a=copy.deepcopy(ASSUMPTIONS); a[field]=val
        runs[key]=solve(a)
        print(key,round(runs[key].get('npv_contribution_usd',0)/1e6,3),flush=True)
    for key,scale in [('smax_minus_20',0.8),('smax_plus_20',1.2),('ramp_minus_20',0.8),('ramp_plus_20',1.2)]:
        a=copy.deepcopy(ASSUMPTIONS)
        if key.startswith('smax'):
            for m in a['markets'].values(): m['smax']=min(1,m['smax']*scale)
        else:
            for c in a['channels'].values(): c['lambda']*=scale
        runs[key]=solve(a)
        print(key,round(runs[key].get('npv_contribution_usd',0)/1e6,3),flush=True)
    runs['target_90pct_year5']=solve(ASSUMPTIONS,volume_target=0.9)
    fine=copy.deepcopy(ASSUMPTIONS)
    fine['price_discount_grid']=[x/100 for x in range(0,41)]
    runs['discount_grid_1pct_to_40pct']=solve(fine)
    # Keep installed channels/entry years and the relative discount rule fixed.
    runs['price_minus_20_fixed_policy']=solve(ASSUMPTIONS,price_scale=0.8,
                                              fixed_strategy=runs['base']['selected'])
    for key,initial in [('initial_share_zero',0),('initial_share_30pct',0.30)]:
        a=copy.deepcopy(ASSUMPTIONS); a['markets']['PER']['initial_share']=initial
        runs[key]=solve(a)
    # Three joint scenarios. Not assigned probabilities.
    for key,changes,kwargs in [
        ('adverse',{'growth':0,'switch_threshold':0.12,'variable_factory_cost_usd_t':264},{'price_scale':0.85,'freight_scale':1.2}),
        ('favorable',{'growth':0.05,'switch_threshold':0.06,'variable_factory_cost_usd_t':216},{'price_scale':1,'freight_scale':0.9})]:
        a=copy.deepcopy(ASSUMPTIONS); a.update(changes)
        runs[key]=solve(a,**kwargs)
    base_dom=p['PER']['pool_t']
    reverse=[]
    for y in ASSUMPTIONS['years']:
        d=base_dom*1.03**(y-2024)
        for share in [0.4,0.5,0.6,0.7]:
            sales=d*0.9*share
            reverse.append({'year':y,'share_of_serviceable':share,'domestic_t':sales,
                            'export_gap_at_90pct_t':max(0,198090*0.9-sales)})
    out={'pools':p,'regression':audit,'runs':runs,'reverse_tests':reverse,
         'limitations':['All forecasts conditional on analyst assumptions.','Colombia excludes unquantified HS700529.',
                        'Chile is whole-country imports, not northern Chile or total demand.',
                        'Ecuador/Bolivia/Chile missing clear-glass mass imputed at assumed thickness.',
                        'Average historical CIF values are not matched current quotes.',
                        'No furnace fixed costs, tax, working capital, production minimum, inventory or factory capex; reported objective is incremental channel contribution NPV, not project NPV.',
                        'Fixed channel and discount from entry through 2031; no switching, product-specific capacity or regional disaggregation.',
                        'No probabilistic claims; parameter distributions and correlations unvalidated.']}
    (ROOT/'computed_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Computed outputs — conditional scenarios, not company forecasts','',
           '## Public-data pools','', '| Market | Pool tonnes | Included CIF USD m | CIF USD/t |','|---|---:|---:|---:|']
    for iso,x in p.items(): lines.append(f"| {iso} | {x['pool_t']:.0f} | {x['included_cif_usd']/1e6:.3f} | {x['cif_unit_value']:.1f} |")
    lines+=['','## Regression diagnostics','', '| Specification | Lag | n | Elasticity | R2 | Conditional holdout MAPE | Naive MAPE |','|---|---:|---:|---:|---:|---:|---:|']
    for x in audit['models']:
        lines.append(f"| {x['mode']} | {x['lag']} | {x['n']} | {x['elasticity']:.3f} | {x['r_squared']:.3f} | {x['holdout_mape']:.1%} | {x['naive_mape']:.1%} |")
    lines+=['','## Scenario optimization','', '| Scenario | Contribution NPV USD m | 2027 tonnes | 2031 tonnes | 2031 utilization | MIP gap |','|---|---:|---:|---:|---:|---:|']
    for key,x in runs.items():
        if not x['feasible']:
            lines.append(f'| {key} | infeasible | | | | |'); continue
        lines.append(f"| {key} | {x['npv_contribution_usd']/1e6:.3f} | {x['sales_t'][0]:.0f} | {x['sales_t'][-1]:.0f} | {x['utilization'][-1]:.1%} | {x['mip_gap']:.6f} |")
    lines+=['','## Base selected menus','', '| Market | Channel | Entry | Discount | 2027 t | 2028 t | 2029 t | 2030 t | 2031 t |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in runs['base']['selected']:
        lines.append(f"| {x['iso']} | {x['channel']} | {x['entry_year']} | {x['discount']:.0%} | "+' | '.join(f'{v:.0f}' for v in x['sales_t'])+' |')
    (ROOT/'computed_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'regression':audit['models'],'base':runs['base'],'reverse':reverse},ensure_ascii=True),flush=True)

if __name__=='__main__': main()
