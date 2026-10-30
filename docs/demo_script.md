# Narrated SmartWealth reviewer demo

This is the word-for-word narration for `outputs/SmartWealth_Demo.mp4`. It is designed for a 3–5 minute playback. The MP4 pairs captured Streamlit screens with a narrated English voiceover and on-screen section captions; it covers INV01, the tolerance what-if, INV07, INV08, and INV03.

## 0:00–0:35 — Product overview

Welcome to SmartWealth, an explainable goal-planning prototype for the NiveshNow challenge. The app uses the supplied synthetic investor, product, and return data. It is an educational planning estimate, not a recommendation or a promise of returns. Each profile is checked against diversification, risk, liquidity, settlement, and cost rules. I’ll show a short goal, a long goal, a tolerance what-if, and two different kinds of infeasibility.

## 0:35–1:20 — INV01: short goal and near-target advice

Select INV01, Wedding expenses. The goal horizon is 24 months, and the app shows both current-balance and SIP allocations in rupees. Only cash and short-duration debt count toward the near-term liquidity need. The projection is within one percent of target, so it is labelled “Within 1% of target,” rather than presenting a tiny gap as a major shortfall. The adjustment also estimates the total monthly SIP, and the extra amount, needed to create a five-percent margin. The downside, base, and optimistic upside figures are assumptions; the bootstrap band shows synthetic path variability and is not a guarantee.

## 1:20–1:50 — Tolerance what-if and binding rule

Open the tolerance what-if for INV04 and raise risk tolerance from three to five. The portfolio mix does not change because the horizon cap remains binding. The app says this directly: “Risk tolerance is not binding here; horizon cap remains binding.” That message helps explain why raising tolerance alone does not alter the recommendation.

## 1:50–2:35 — INV07: long goal and distinct balance/SIP routing

Select INV07, Child higher education. Its 144-month horizon permits a larger growth sleeve, but the risk ceiling is a limit, not a target. The current balance and recurring SIP use separate mixes: PROD07 can hold existing money that matures before the goal, while future SIP instalments avoid locked products. The product reasons connect that routing to the tax-saving preference, and the cost panel keeps fees visible alongside the suitability rationale.

## 2:35–3:25 — INV08: goal infeasibility

Select INV08, Start a business. Here the goal is not reached even in the optimistic upside case. The app labels that case optimistic, gives the monthly SIP required under the base return assumption, and suggests resetting the target near the base projection. A longer horizon is another option, but the displayed SIP and goal reset are based on the base case rather than the more favorable upside assumption. Because the emergency fund is below three months, the locked sleeve is capped and the UI compares the current plan with a buffer-first option.

## 3:25–4:15 — INV03: suitability infeasibility and short-horizon what-if

Select INV03, Home renovation. Its risk tolerance and short 15-month horizon make a fully compliant diversified portfolio impossible with the available products. The red FALLBACK suitability status is the one to act on; goal progress is a separate projection calculation and does not make that fallback suitable. Choose “Shorter horizon by 12 months.” The scenario reduces the horizon to three months and still returns a result with a short-horizon warning instead of rejecting the profile. In custom adjustments, horizons down to one month are also supported.

## 4:15–4:35 — Close

SmartWealth makes the constraints, costs, assumptions, and trade-offs visible. The allocations and summary CSVs can be downloaded from the sidebar, and `python generate_outputs.py` refreshes the batch files. Thank you.

