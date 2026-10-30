# SmartWealth report

Prototype for NiveshNow: turn a short investor profile and a fictional catalogue into a suitable, cost-aware, explainable plan. Not a price forecast.

## Architecture

- `engine.py` — stable public facade. `validation.py`, `allocation.py`, `projection.py`, `explain.py`, and `planner.py` hold the separated implementation.
- `app.py` — Streamlit reviewer UI (persona, mix, plan, reasons, what-if).
- `data/` — the three challenge CSVs.
- `tests.py` — guardrail and edge-case checks.

Flow: validate catalogue files and each investor row independently → build separate mixes for current balance and recurring SIPs → apply timing, cash-need, risk, diversification and ticket checks to each relevant mix → trim high-risk weight only when the downside still funds the goal or the base projection remains at least 5% above target → label a **safer fallback** if any suitability check remains unmet → project FV from catalogue assumed returns.

## Allocation logic

Additional rules beyond the brief, all secondary to suitability:

1. Near-term liquid means only the supplied cash reserve (PROD01) and designated short-duration debt fund (PROD03), both unlocked and settling within 7 days. Medium-duration/credit-sensitive PROD04, equity, hybrid, and lock-ins never cover a cash need.
2. Emergency fund below 3 months: locked balance exposure is capped at **0%** (within the requested 0–10% cap), and the liquid sleeve is ranked first. A 20% reserve of existing capital—or future SIPs when balance is zero—is placed in cash/short-duration debt. This reserve **overlaps** the stated liquidity need; it is not additive. The UI compares the active plan against a buffer-first alternative.
3. Preferences apply after hard rules: `avoid_equity` drops equity, `no_lock_in` drops lock-ins, `tax_saving` prefers PROD07 in the high sleeve, `high_growth` prefers higher assumed return inside the horizon cap, `low_cost` / `capital_protection` sort by expense and risk.
4. Beginner experience does not stretch the high-risk cap.
5. Minimum tickets use the catalogue general minimum as a per-instalment SIP proxy because the source has no SIP-specific minimum. SIP repair works in rupee amounts and selects the smallest feasible compliant product set; for INV11, it routes the ₹3,000 SIP to three ₹1,000 tickets. If no compliant ticket mix exists, the UI flags that the proxy was not enforced. The UI shows distinct balance and SIP weights and rupee amounts.
6. Recurring SIPs never route to locked products. Current balance may use a locked product only when lock-in plus settlement ends by the goal date.
7. Whole-month horizons from 1 month are supported; 1–5 month goals receive a short-horizon warning, and the shorter-horizon what-if floors at 1 month. Horizon risk bands are ceilings. High-risk exposure is trimmed only when the downside still reaches the goal or the base projection retains at least a 5% margin. Goal labels include `WITHIN_1_PERCENT`, shown as “Within 1% of target”; it includes the SIP increment estimated to add a 5% margin. Infeasible-goal advice uses the base-case SIP requirement, labels upside amounts optimistic, and suggests resetting the goal near the base projection. A zero-balance, zero-SIP profile is `NEEDS_CLARIFICATION`; a balance at or above target prompts a higher-target review.
8. The asset-class shock uses illustrative assumptions: cash absorbs none of the shock, debt 10%, hybrid 50%, and equity 100%. These factors are transparent demo inputs, not calibrated forecasts.

If weighted risk, horizon bands, lock-in, liquidity coverage, minimum tickets, or diversification still fail, `planning_status=FALLBACK` and the unmet constraint is written on the summary row. An upside projection below target is separately marked `goal_status=UPSIDE_SHORTFALL`.

## Projections

The balance and SIP mixes use separate catalogue assumed annual returns. Projection is the sum of two exact FV calculations: `FV(balance, 0, r_balance, n) + FV(0, SIP, r_SIP, n)`. The displayed blended `rp` is exposure-weighted across those two streams; the projection does not approximate them with that blended rate. A separate paired circular moving-block bootstrap resamples the 36 monthly return rows in three-month blocks, using the same sampled months for every product. The app reports the 5th, 50th, and 95th percentiles over 1,000 paths. This preserves observed cross-product co-movement, but 36 synthetic months are limited and do not make the range a forecast. The bootstrap measures path variability only, while ±3pp represents assumption error; for long goals, the planning envelope uses the wider of the two.

Scenario range: `rp−3pp`, `rp`, `rp+3pp`, floored at 0% and capped at 18%; the same shift is applied separately to the balance and SIP rates. The FV formula matches the brief, including the `r = 0` limit. Required SIP is solved against the balance growth rate and SIP growth rate separately. The bootstrap percentiles are supplemental and do not replace the required range. Gross projections are shown beside a net-of-fee base projection, and current-balance annual cost is shown separately from first-year SIP cost. A 15% shock applies differentiated losses to current holdings by asset class (cash 0%, debt 1.5%, hybrid 7.5%, equity 15%); it leaves future SIPs and target weights unchanged.

The catalogue does not specify what happens to matured fixed deposits after their lock-in. For projection, maturity proceeds are assumed to be reinvested at that product's assumed annual return through the goal date; this is a modelling assumption, not a promised rate.

## Trade-offs

The engine is a transparent rules system, not a mean-variance optimiser. That keeps every line attributable to a constraint, at the cost of not maximising expected surplus. Cost is a sort key inside a sleeve, never a reason to breach liquidity or risk. INV03 cannot hit risk ≤ 1.25 with a 40% cap and only one eligible risk-1 product; we show that conflict instead of hiding it.

## Failure cases

- INV03: the goal projection can show ON_TRACK while suitability is FALLBACK because weighted risk exceeds its 1.25 limit. Reviewers should act on the red FALLBACK suitability status; the goal label describes only the arithmetic projection.
- INV11: ₹0 starting balance; its ₹3,000 SIP is routed as three ₹1,000 instalments, and the first-year cost is shown explicitly. The catalogue has no SIP-specific minimum, so its general minimum is a transparent per-instalment proxy.
- INV08: goal is labelled infeasible because even the upside case remains far below target; advice quotes the base-case SIP requirement and suggests resetting the goal near the base projection. Any upside amount is explicitly optimistic.
- INV04 and INV11 also have `UPSIDE_SHORTFALL` under the supplied assumed returns.
- Short horizon + `high_growth`: preference loses to the 10% risk 4–5 cap.
- Malformed catalogue files fail at load; malformed investor rows are flagged individually and skipped for planning so other personas remain usable.

## What we would improve with more time

Constrained optimisation with explicit penalty on expense and drawdown; mapping SIP cash-flows into product lots over time; a custom-profile form; and a proper Monte Carlo using the synthetic returns as a stress distribution rather than a single trailing rate.

