# SmartWealth

A deterministic, explainable goal-planning prototype for the NiveshNow challenge. It uses the supplied synthetic investor, product, and monthly-return CSVs. It is an educational demonstration, not investment advice or a guarantee of returns.

## Run

Windows: run `run.bat` from the project folder. macOS/Linux: run `sh run.sh`. Each launcher creates a local virtual environment, installs `requirements.txt`, and starts Streamlit. Open the `Local URL` shown in the terminal (usually `http://localhost:8501`). The `width="stretch"` UI requires Streamlit 1.49 or newer.

The Streamlit UI caches inputs and calculations and does not write files on widget reruns. Batch CSV creation is a separate command:

```sh
python generate_outputs.py
```

This writes `outputs/allocations_output.csv` and `outputs/summary_output.csv`. The technical report is supplied as `REPORT.pdf` (with `REPORT.md` as its editable source). Create a clean reviewer bundle with `python package_submission.py`; the ZIP excludes `.git`, Python caches, pytest caches, and scratch files.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest tests.py -p no:cacheprovider
```

The suite covers allocations, suitability bands, lock-in and SIP timing, malformed profile rows, risk fallback, preferences, minimum tickets, liquidity, goal materiality and risk trimming, base-case advice, emergency-buffer caps, consistency prompts, scenario narration, projections, and asset-class market shocks.

## Data column mappings

- Investor risk tolerance: `investor_profiles.csv:risk_tolerance_score`
- Product category: `investment_products.csv:asset_class`
- Product risk: `investment_products.csv:risk_score`
- Monthly observed return: `synthetic_monthly_returns.csv:monthly_return`, keyed by `product_id`

The bundled CSVs live in `data/`. Replace all three together to use another dataset. Product and return data errors stop loading; malformed investor rows are reported individually so the other personas remain usable. Supported preferences are `capital_protection`, `low_cost`, `high_growth`, `avoid_equity`, `tax_saving`, and `no_lock_in`; supported experience values are `beginner`, `intermediate`, and `advanced`.

## Decision and projection notes

The modules are split by responsibility: `validation.py`, `allocation.py`, `projection.py`, `explain.py`, and `planner.py`; `engine.py` remains the backwards-compatible facade. The engine enforces a 40% maximum per product, at least two categories, horizon-based risk bands, investor risk tolerance plus 0.25, settlement-aware lock-in eligibility, defensive liquidity, emergency-buffer warnings, and minimum-ticket checks when a guardrail-compliant mix exists. Whole-month horizons from 1 month are allowed; 1–5 months raise a prominent short-horizon warning. Current balance and SIP flows have separate allocation weights: locked products may hold existing balance when maturity fits, while SIPs always go to unlocked products. The product file has no SIP-specific minimum, so its general minimum is a proxy applied per instalment in rupees; when a mix needs repair, the engine searches for the smallest compliant product set. Liquidity shortfalls and conservative fallbacks are visibly flagged. When emergency savings are below three months, locked balance exposure is capped at 0% and the liquid sleeve is ranked first. The 20% reserve overlaps the stated near-term liquidity need rather than adding to it. The UI also compares the current plan with a buffer-first option. Goal statuses include `ON_TRACK`, `WITHIN_1_PERCENT`, `ON_TRACK_BASE_ONLY`, `SHORTFALL`, and `UPSIDE_SHORTFALL`; `WITHIN_1_PERCENT` shows the base-case SIP needed to add a 5% margin. A profile with no balance and no SIP is marked `NEEDS_CLARIFICATION`, while a balance at or above the goal prompts the reviewer to consider a higher target. Goal progress and suitability are reported separately.

The projection uses two exact applications of the brief's monthly FV formula, one for current balance and one for SIP contributions, with catalogue assumed annual returns. It reports the required `rp ± 3 percentage points` range with a 0% floor and 18% cap, plus a supplemental 5th/50th/95th percentile sensitivity from 1,000 paired three-month block-bootstrap paths using the 36 synthetic monthly returns. The bootstrap preserves observed cross-product co-movement, but the short synthetic history is not a forecast. It describes path variability only; ±3pp represents assumption error. For long goals, the planning envelope uses the wider of the two ranges. Matured fixed deposits are assumed to be reinvested at their catalogue assumed rate through the goal date because the source data gives no reinvestment rule. The asset-class shock applies 0% to cash, 10% of the shock to debt, 50% to hybrid, and 100% to equity; it affects existing holdings only.

Annual product cost on current capital is shown beside an estimated first-year SIP cost; net-of-fee projection is shown beside gross projection. See [REPORT.pdf](REPORT.pdf) for the technical report and [docs/demo_script.md](docs/demo_script.md) for the narrated reviewer-demo script. The four-minute demo video is provided separately through the Google Drive link in the submission form.

