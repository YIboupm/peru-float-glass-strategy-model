# Peru Float Glass Strategy Model

This repository contains the reproducible public-data validation and rebuilt five-year market-entry model for a proposed float-glass plant in Peru. The analysis covers addressable demand, landed-cost competitiveness, share ramp-up, capacity allocation, channel break-even and uncertainty.

## Contents

- `outputs/秘鲁玻璃项目_实证核验与模型重建.md` — main research report in Chinese, with sources and model recommendations.
- `outputs/verify_model.py` — executable deterministic scenario model and sensitivity checks.
- `outputs/*.json` and `outputs/computed_tables.md` — structured inputs, historical construction data and computed outputs.
- `work/fetch_public.py`, `work/prepare_data.py`, `work/raw/` — scripts and raw public trade-data responses used to assemble the analysis.

## Reproduce

Requires Python 3.12+ and the packages listed in `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
python outputs/verify_model.py
```

The analysis is an investment-screening model, not a substitute for current freight quotes, customer research, technical certification checks or legal/tax advice.
