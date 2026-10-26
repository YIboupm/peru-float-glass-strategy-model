"""Read public UN Comtrade and BCRP data; preserve responses and URLs."""
import concurrent.futures
import json
import pathlib
import urllib.parse
import urllib.request
import ssl
import time

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / 'raw'
RAW.mkdir(exist_ok=True)
CONTEXT = ssl.create_default_context()
for cert, encoding, trust in ssl.enum_certificates('ROOT'):
    if encoding == 'x509_asn':
        try:
            CONTEXT.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(cert))
        except ssl.SSLError:
            pass

def get(task):
    name, url = task
    path = RAW / (name + '.json')
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
    else:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Public research model/1.0'})
            with urllib.request.urlopen(req, timeout=40, context=CONTEXT) as res:
                body = json.load(res)
            data = {'url': url, 'retrieved': '2026-10-26', 'response': body}
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as exc:
            return {'name': name, 'error': str(exc),
                    'details': exc.read().decode(errors='replace') if hasattr(exc,'read') else ''}
    body = data['response']
    if 'data' in body:
        fields = ['period','reporterCode','flowCode','cmdCode','qty','qtyUnitCode','netWgt',
                  'isNetWgtEstimated','isQtyEstimated','primaryValue','cifvalue','fobvalue']
        return {'name': name, 'count': body.get('count'),
                'data': [{k: r.get(k) for k in fields} for r in body.get('data',[])]}
    return {'name': name, 'body': body}

def ct(name, reporter, period, cmd='7005', flow='M,X'):
    args = dict(period=period,reporterCode=reporter,cmdCode=cmd,flowCode=flow,
                partnerCode=0,partner2Code=0,customsCode='C00',motCode=0,maxRecords=500)
    return name, 'https://comtradeapi.un.org/public/v1/preview/C/A/HS?' + urllib.parse.urlencode(args)

tasks = [ct('CHL_2024',152,'2024','7005,700510,700521,700529,700530'),
         ct('COL_2024',170,'2024','7005,700510,700521,700529,700530')]
tasks += [ct('PER_'+str(y),604,str(y)) for y in range(2012,2026) if y != 2024]
for task in tasks:
    result = get(task)
    print(json.dumps(result,ensure_ascii=True),flush=True)
    if '429' in str(result.get('error','')):
        print('Rate limit encountered; stopping Comtrade requests.',flush=True)
        break
    time.sleep(5)
