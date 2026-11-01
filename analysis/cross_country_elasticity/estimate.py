"""Cross-country construction -> flat-glass elasticity and lag.

Run: python analysis/cross_country_elasticity/estimate.py
Inputs : eurostat_indices.json (this folder), outputs/construction_history.json (Peru BCRP).
Outputs: results.json and results.md (this folder).

Idea: Peru alone has ~9 usable annual points, so borrow the *cyclical* response of
glass demand to building activity from European countries with long, deep cycles.

Glass-demand proxy: Eurostat production index for NACE C2312 "shaping and processing
of flat glass" (tempered/laminated/insulating units). Processed units are bulky and
mostly sold domestically, so this tracks national building-glass use far better than
float production (C2311), which is heavily exported. C2311 is estimated only as a
contrast. Indices are scale-free, so log-difference slopes are elasticities.

All regressions are in annual log growth (first differences): levels of trending
indices give spurious fits. The intercept absorbs each country's trend (e.g. more
glass per window from energy rules); the slope is the cyclical elasticity, which is
the part that can plausibly transfer to Peru.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EU = json.loads((HERE / 'eurostat_indices.json').read_text(encoding='utf-8'))['data']
PERU_C = {int(k): v for k, v in json.loads(
    (REPO / 'outputs' / 'construction_history.json').read_text(encoding='utf-8'))['values'].items()}

COVID = {2020, 2021}


def series(name, geo):
    return {int(y): v for y, v in EU[name].get(geo, {}).items()}


def dlog(s):
    return {y: math.log(s[y] / s[y - 1]) for y in s if y - 1 in s}


def ols(X, y):
    """OLS with classical and HC1 standard errors."""
    n, k = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    xtx_inv = np.linalg.inv(X.T @ X)
    s2 = e @ e / (n - k)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    meat = (X * e[:, None]).T @ (X * e[:, None])
    se_hc1 = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv) * n / (n - k))
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (e @ e) / sst if sst else float('nan')
    return b, se, se_hc1, r2, n, k


def design(y_s, x_s, lags, years_ok=None, drop=()):
    """Rows where y_t and x_{t-L} for all L in lags exist."""
    rows = []
    for t in sorted(y_s):
        if t in drop or (years_ok and t not in years_ok):
            continue
        if all(t - L in x_s for L in lags):
            rows.append((t, y_s[t], [x_s[t - L] for L in lags]))
    return rows


def fit_single(glass, constr, geo, lag, drop=()):
    y_s, x_s = dlog(series(glass, geo)), dlog(series(constr, geo))
    rows = design(y_s, x_s, [lag], drop=drop)
    if len(rows) < 8:
        return None
    X = np.column_stack([np.ones(len(rows)), [r[2][0] for r in rows]])
    y = np.array([r[1] for r in rows])
    b, se, hc, r2, n, k = ols(X, y)
    p = 2 * stats.t.sf(abs(b[1] / hc[1]), n - k)
    return {'geo': geo, 'glass': glass, 'constr': constr, 'lag': lag, 'n': n,
            'years': f'{rows[0][0]}-{rows[-1][0]}', 'elasticity': b[1], 'se_hc1': hc[1],
            'p_hc1': p, 'r2': r2, 'trend_pct': 100 * (math.exp(b[0]) - 1), 'drop': sorted(drop)}


def fit_distributed(glass, constr, geos, lags=(0, 1), drop=()):
    """Pooled panel with country intercepts; returns each lag and long-run sum."""
    Xs, ys, ids = [], [], []
    for g in geos:
        y_s, x_s = dlog(series(glass, g)), dlog(series(constr, g))
        for t, yv, xv in design(y_s, x_s, list(lags), drop=drop):
            ys.append(yv); Xs.append(xv); ids.append(g)
    present = sorted(set(ids))
    if len(ys) < 10:
        return None
    D = np.array([[1.0 if i == g else 0.0 for g in present] for i in ids])
    X = np.column_stack([D, np.array(Xs)])
    y = np.array(ys)
    b, se, hc, r2, n, k = ols(X, y)
    m = len(present)
    xtx_inv = np.linalg.inv(X.T @ X)
    e = y - X @ b
    meat = (X * e[:, None]).T @ (X * e[:, None])
    V = xtx_inv @ meat @ xtx_inv * n / (n - k)
    w = np.zeros(k); w[m:] = 1
    lr, lr_se = float(w @ b), float(math.sqrt(w @ V @ w))
    return {'glass': glass, 'constr': constr, 'geos': present, 'lags': list(lags), 'n': n,
            'coef': {f'lag{L}': float(b[m + j]) for j, L in enumerate(lags)},
            'se_hc1': {f'lag{L}': float(hc[m + j]) for j, L in enumerate(lags)},
            'long_run': lr, 'long_run_se': lr_se,
            'long_run_ci95': [lr - 1.96 * lr_se, lr + 1.96 * lr_se], 'r2': r2, 'drop': sorted(drop)}


def peru_check():
    """Same growth spec on Peru's own (sparse) data, for comparison."""
    pub = json.loads((REPO / 'outputs' / 'public_data.json').read_text(encoding='utf-8'))
    net = {}
    for y in range(2012, 2025):
        rows = pub.get(f'PER_{y}', {}).get('response', {}).get('data', [])
        m = next((r for r in rows if r['cmdCode'] == '7005' and r['flowCode'] == 'M'), None)
        x = next((r for r in rows if r['cmdCode'] == '7005' and r['flowCode'] == 'X'), None)
        if m and x and m['netWgt'] and x['netWgt']:
            net[y] = (m['netWgt'] - x['netWgt']) / 1000
    y_s, x_s = dlog(net), dlog(PERU_C)
    out = []
    for lag in (0, 1):
        rows = design(y_s, x_s, [lag])
        X = np.column_stack([np.ones(len(rows)), [r[2][0] for r in rows]])
        b, se, hc, r2, n, k = ols(X, np.array([r[1] for r in rows]))
        out.append({'lag': lag, 'n': n, 'years': [r[0] for r in rows],
                    'elasticity': float(b[1]), 'se_classical': float(se[1]), 'r2': r2})
    return out


def main():
    single = []
    for glass in ('c2312', 'c2311'):
        for constr in ('b41', 'f'):
            for geo in ('DE', 'ES', 'FR', 'IT'):
                for lag in (0, 1, 2):
                    for drop in ((), tuple(COVID)):
                        r = fit_single(glass, constr, geo, lag, drop)
                        if r:
                            single.append(r)
    panel = []
    for glass, constr, geos in [('c2312', 'b41', ('DE', 'ES', 'FR')),
                                ('c2312', 'f', ('DE', 'ES', 'FR', 'IT')),
                                ('c2311', 'f', ('DE', 'ES', 'FR', 'IT'))]:
        for lags in ((0,), (0, 1), (0, 1, 2)):
            for drop in ((), tuple(COVID)):
                r = fit_distributed(glass, constr, geos, lags, drop)
                if r:
                    panel.append(r)
    peru = peru_check()

    # Apply to Peru: BCRP building value added growth 2025, and the BCRP 2026
    # sector projection (6%) quoted in public reporting; 3% = model's assumption.
    g25 = PERU_C[2025] / PERU_C[2024] - 1
    pick = next(p for p in panel if p['glass'] == 'c2312' and p['constr'] == 'b41'
                and p['lags'] == [0, 1] and p['drop'] == sorted(COVID))
    lo, hi = pick['long_run_ci95']
    apply = [{'constr_growth': g, 'label': lab,
              'glass_growth_point': pick['long_run'] * g,
              'glass_growth_lo': lo * g, 'glass_growth_hi': hi * g}
             for g, lab in [(g25, 'BCRP 2025 actual'), (0.06, 'BCRP 2026 projection'),
                            (0.0, 'flat'), (-0.10, 'downturn -10%')]]

    out = {'single_country': single, 'panel': panel, 'peru_own_data': peru,
           'peru_application': {'elasticity_used': pick, 'scenarios': apply}}
    (HERE / 'results.json').write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                                       encoding='utf-8')

    L = ['# Cross-country elasticity results', '',
         'Annual log-growth regressions. Elasticity = % change in glass index per 1% change in construction index.',
         'HC1 robust standard errors. "ex-Covid" drops 2020 and 2021.', '',
         '## Pooled panel (country intercepts)', '',
         '| Glass | Construction | Countries | Lags | Sample | n | Lag coefficients | Long-run sum | 95% CI | R2 |',
         '|---|---|---|---|---|---:|---|---:|---|---:|']
    for p in panel:
        coefs = ', '.join(f"{k}={v:.2f}" for k, v in p['coef'].items())
        L.append(f"| {p['glass']} | {p['constr']} | {'/'.join(p['geos'])} | {p['lags']} | "
                 f"{'ex-Covid' if p['drop'] else 'all'} | {p['n']} | {coefs} | {p['long_run']:.2f} | "
                 f"[{p['long_run_ci95'][0]:.2f}, {p['long_run_ci95'][1]:.2f}] | {p['r2']:.2f} |")
    L += ['', '## Single country, single lag', '',
          '| Glass | Construction | Country | Lag | Sample | Years | n | Elasticity | SE | p | R2 | Trend %/yr |',
          '|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in single:
        L.append(f"| {r['glass']} | {r['constr']} | {r['geo']} | {r['lag']} | {'ex-Covid' if r['drop'] else 'all'} | "
                 f"{r['years']} | {r['n']} | {r['elasticity']:.2f} | {r['se_hc1']:.2f} | {r['p_hc1']:.3f} | "
                 f"{r['r2']:.2f} | {r['trend_pct']:.1f} |")
    L += ['', '## Peru, same growth specification on its own data', '',
          '| Lag | n | Years | Elasticity | SE | R2 |', '|---:|---:|---|---:|---:|---:|']
    for r in peru:
        L.append(f"| {r['lag']} | {r['n']} | {r['years']} | {r['elasticity']:.2f} | {r['se_classical']:.2f} | {r['r2']:.2f} |")
    L += ['', '## Applying the borrowed elasticity to Peru (cyclical part only)', '',
          f"Elasticity used: c2312 on b41, lags 0+1, ex-Covid: {pick['long_run']:.2f} "
          f"(95% CI {lo:.2f} to {hi:.2f}).", '',
          '| Construction growth | Scenario | Implied glass growth | Range |', '|---:|---|---:|---|']
    for a in apply:
        L.append(f"| {a['constr_growth']:.1%} | {a['label']} | {a['glass_growth_point']:.1%} | "
                 f"{a['glass_growth_lo']:.1%} to {a['glass_growth_hi']:.1%} |")
    (HERE / 'results.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
