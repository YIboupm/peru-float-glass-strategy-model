import json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
data={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (root/'work'/'raw').glob('*.json')}
(root/'outputs'/'public_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved',len(data),'public query responses')
