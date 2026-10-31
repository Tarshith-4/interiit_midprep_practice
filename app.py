"""Streamlit review interface for SmartWealth. File exports are handled by generate_outputs.py."""
import pandas as pd
import altair as alt
import streamlit as st

from engine import (
    DATA_DIR, OUTPUT_DIR, DataValidationError, SuitabilityError, check_consistency,
    load_data, plan_buffer_first, plan_profile, scenario_narrative, shorter_horizon_months, validate_profile,
)

st.set_page_config(page_title="SmartWealth | NiveshNow", layout="wide")


@st.cache_data(ttl=60)
def cached_data(data_dir: str):
    return load_data(data_dir)


@st.cache_data(ttl=60)
def cached_plan(data_dir: str, investor_id: str, balance: float, sip: float,
                horizon: int, tolerance: float, market_shock: float):
    profiles, products, returns = load_data(data_dir)
    profile = profiles.loc[profiles.investor_id == investor_id].iloc[0].copy()
    profile["existing_investment_inr"] = balance
    profile["monthly_sip_inr"] = sip
    profile["horizon_months"] = horizon
    profile["risk_tolerance_score"] = tolerance
    return plan_profile(profile, products, returns, market_shock=market_shock)


@st.cache_data(ttl=60)
def cached_buffer_first(data_dir: str, investor_id: str):
    profiles, products, returns = load_data(data_dir)
    profile = profiles.loc[profiles.investor_id == investor_id].iloc[0].copy()
    return plan_buffer_first(profile, products, returns)


st.title("SmartWealth")
st.caption("Explainable goal planning · Educational estimates only, not personal financial advice or guaranteed outcomes.")
try:
    profiles, products, returns = cached_data(str(DATA_DIR))
except DataValidationError as exc:
    st.error(f"Input data validation failed: {exc}")
    st.stop()

st.sidebar.caption("Exports are generated separately; the app does not rewrite files during UI reruns.")
for path, label in ((OUTPUT_DIR / "allocations_output.csv", "Download allocations CSV"),
                    (OUTPUT_DIR / "summary_output.csv", "Download summary CSV")):
    if path.exists():
        st.sidebar.download_button(label, path.read_bytes(), file_name=path.name, mime="text/csv")
    else:
        st.sidebar.info(f"Run `python generate_outputs.py` to create {path.name}.")

ids = profiles.investor_id.dropna().astype(str).tolist()
requested_persona = str(st.query_params.get("persona", ""))
persona_index = ids.index(requested_persona) if requested_persona in ids else 0
selected_id = st.sidebar.selectbox(
    "Investor persona", ids,
    index=persona_index,
    format_func=lambda x: f"{x} · {profiles.loc[profiles.investor_id.astype(str)==x, 'goal_label'].iloc[0]}",
)
profile = profiles.loc[profiles.investor_id.astype(str) == selected_id].iloc[0]
profile_errors = validate_profile(profile)
consistency = check_consistency(profile)
if profile_errors:
    st.error("This persona has invalid profile data: " + "; ".join(profile_errors))
    for prompt in consistency["prompts"]:
        st.warning("Clarification: " + prompt)
    st.stop()
for prompt in consistency["prompts"]:
    if not prompt.startswith("Needs clarification:"):
        st.warning("Clarification: " + prompt)
st.subheader(f"{profile.goal_label} · {selected_id}")
st.write(
    f"Goal ₹{profile.goal_amount_inr:,.0f} in {profile.horizon_months:g} months · "
    f"Starting balance ₹{profile.existing_investment_inr:,.0f} · Monthly SIP ₹{profile.monthly_sip_inr:,.0f} · "
    f"Risk {profile.risk_tolerance_score:g}/5 · Emergency fund {profile.emergency_fund_months:g} months · "
    f"Preference `{profile.preference or 'none'}` · Experience {profile.experience}"
)
try:
    allocation, summary = cached_plan(
        str(DATA_DIR), selected_id, float(profile.existing_investment_inr),
        float(profile.monthly_sip_inr), int(profile.horizon_months),
        float(profile.risk_tolerance_score), 0.0,
    )
except (DataValidationError, SuitabilityError) as exc:
    st.error(f"This profile cannot be planned: {exc}")
    st.stop()

if summary["needs_clarification"]:
    st.warning("Needs clarification — no starting balance or monthly SIP is planned. Add a contribution or confirm that no investment plan is intended.")
if summary["suitability_status"] == "FALLBACK":
    residual = [item for item in summary["planning_exception"].split("; ") if item.startswith("Fallback residual:")]
    primary = [item for item in summary["planning_exception"].split("; ") if item.startswith("Primary:")]
    if residual:
        st.error("Suitability: FALLBACK — no compliant recommendation is available. The conservative fallback still has unresolved constraints: " + "; ".join(residual))
    else:
        st.error("Suitability: FALLBACK — this conservative portfolio does not pass the risk-fit rules and is not a compliant recommendation. " + "; ".join(primary))
elif summary["suitability_status"] == "INVALID_PROFILE":
    st.error(summary["planning_exception"])
elif not summary["needs_clarification"]:
    st.success("Suitability: PASS — portfolio guardrails pass for this profile.")
st.markdown(f"**Goal progress:** `{summary['goal_status_label']}`. This projection status is separate from portfolio suitability.")
if summary["liquidity_covered"] is False:
    st.error("Current liquid assets do not cover the stated near-term liquidity need.")
if summary["goal_status"] == "UPSIDE_SHORTFALL":
    st.warning(
        f"Goal is infeasible within the current horizon even in the optimistic upside scenario. "
        f"Optimistic upside estimate: ₹{summary['projection_upside_inr']:,.0f} vs goal ₹{summary['goal_amount_inr']:,.0f}."
    )
elif summary["goal_status"] == "SHORTFALL":
    st.warning("The base estimate misses the goal, although the upside case reaches it. Review the contribution or target.")
elif summary["goal_status"] == "WITHIN_1_PERCENT":
    st.info(summary["goal_materiality_note"] or "The base estimate reaches the goal; the downside estimate does not.")
for warning in filter(None, summary["warnings"].split(" | ")):
    st.warning(warning)

a, b, c, d, e = st.columns(5)
a.metric("Weighted risk", f"{summary['weighted_risk_score']:.2f} / {summary['risk_limit']:.2f}")
b.metric("Annual expense ratio", f"{summary['weighted_annual_expense_ratio']:.2%}")
c.metric("Annual cost on current balance", f"₹{summary['estimated_annual_cost_inr']:,.0f}")
d.metric("Estimated first-year SIP cost", f"₹{summary['first_year_sip_product_cost_inr']:,.0f}")
e.metric("Near-term liquid now", f"₹{summary['liquid_allocation_inr']:,.0f}",
         "no contributions" if summary["liquidity_covered"] is None else ("covers need" if summary["liquidity_covered"] else "shortfall"))
if float(summary["starting_balance_inr"]) == 0:
    st.caption("Current-balance annual cost is ₹0 because no balance is invested; the first-year SIP cost estimate is shown separately.")

if float(summary["emergency_fund_months"]) < 3:
    st.markdown("#### Buffer-first comparison")
    st.caption("The conservative option reserves 20% of available capital (or SIP flows when balance is zero) in unlocked cash/short-duration debt. This reserve overlaps the stated liquidity need; it is not added on top.")
    try:
        buffer_allocation, buffer_summary = cached_buffer_first(str(DATA_DIR), selected_id)
        compare = pd.DataFrame([
            {"Option": "Current plan", "Liquid balance (₹)": summary["liquid_allocation_inr"],
             "Locked balance %": summary["locked_balance_weight_pct"],
             "Downside (₹)": summary["projection_downside_inr"],
             "Base (₹)": summary["projection_base_inr"], "Upside (₹)": summary["projection_upside_inr"]},
            {"Option": "Buffer-first", "Liquid balance (₹)": buffer_summary["liquid_allocation_inr"],
             "Locked balance %": buffer_summary["locked_balance_weight_pct"],
             "Downside (₹)": buffer_summary["projection_downside_inr"],
             "Base (₹)": buffer_summary["projection_base_inr"], "Upside (₹)": buffer_summary["projection_upside_inr"]},
        ])
        st.dataframe(compare, hide_index=True, width="stretch")
        st.caption("Buffer-first suitability: " + buffer_summary["planning_status"] +
                   (" — " + buffer_summary["planning_exception"] if buffer_summary["planning_exception"] else ""))
    except (SuitabilityError, DataValidationError, ValueError) as exc:
        st.warning(f"Buffer-first option could not be generated: {exc}")

st.markdown("#### Asset-class mix · current balance")
mix = pd.DataFrame(summary["asset_class_mix"])
if not mix.empty:
    mix = mix.rename(columns={"asset_class": "Category", "weight_pct": "Weight %"})
    left, right = st.columns([1, 2])
    left.dataframe(mix, hide_index=True, width="stretch")
    right.bar_chart(mix.set_index("Category"))

st.markdown("#### Recommended allocation")
columns = ["product_id", "product_name", "asset_class", "risk_score", "balance_weight_pct", "sip_weight_pct",
           "allocated_amount_inr", "monthly_sip_allocation_inr", "annual_expense_ratio",
           "lock_in_months", "liquidity_days", "near_term_liquid"]
view = allocation[columns].copy()
view.columns = ["Product ID", "Product", "Category", "Risk", "Balance allocation %", "SIP allocation %", "Balance amount (₹)",
                "Monthly SIP (₹)", "Expense ratio", "Lock-in (months)", "Liquidity (days)", "Near-term liquid"]
st.dataframe(view, width="stretch", hide_index=True)
st.caption("The product file has no SIP-specific minimum. The engine treats its general ticket as a per-instalment rupee proxy, preferring the smallest compliant product set; if the guardrails prevent that, it flags the products where the proxy could not be enforced.")

st.markdown("#### Goal projection")
pcols = st.columns(4)
for col, label, key in zip(
    pcols, ["Downside (rp − 3pp)", "Base gross", "Optimistic upside (rp + 3pp)", "Base net of fees"],
    ["projection_downside_inr", "projection_base_inr", "projection_upside_inr", "net_projection_base_inr"],
):
    col.metric(label, f"₹{summary[key]:,.0f}", f"vs goal ₹{summary[key]-summary['goal_amount_inr']:,.0f}")
st.caption("Net projection subtracts each mix's weighted expense ratio from its assumed annual return; the gross figures above retain catalogue returns.")
scenario_chart = pd.DataFrame({
    "scenario": ["Downside", "Base", "Optimistic upside"],
    "value": [summary["projection_downside_inr"], summary["projection_base_inr"], summary["projection_upside_inr"]],
})
order = ["Downside", "Base", "Optimistic upside", "Bootstrap band"]
line = alt.Chart(scenario_chart).mark_line(point=alt.OverlayMarkDef(size=75), color="#1f5f8b").encode(
    x=alt.X("scenario:N", sort=order, title="Scenario"),
    y=alt.Y("value:Q", title="Projected value (₹)", axis=alt.Axis(format="~s"), scale=alt.Scale(zero=False)),
    tooltip=[alt.Tooltip("scenario:N"), alt.Tooltip("value:Q", format=",.0f")],
)
band_data = pd.DataFrame({
    "scenario": ["Bootstrap band"],
    "p05": [summary["bootstrap_projection_p05_inr"]],
    "p50": [summary["bootstrap_projection_p50_inr"]],
    "p95": [summary["bootstrap_projection_p95_inr"]],
})
band = alt.Chart(band_data).mark_rule(color="#57a773", strokeWidth=16, opacity=0.35).encode(
    x=alt.X("scenario:N", sort=order), y=alt.Y("p05:Q"), y2=alt.Y2("p95:Q"),
    tooltip=[alt.Tooltip("p05:Q", format=",.0f", title="Bootstrap 5th"), alt.Tooltip("p50:Q", format=",.0f", title="Median"), alt.Tooltip("p95:Q", format=",.0f", title="Bootstrap 95th")],
)
median = alt.Chart(band_data).mark_point(color="#257a46", filled=True, size=80).encode(
    x=alt.X("scenario:N", sort=order), y=alt.Y("p50:Q")
)
goal_rule = alt.Chart(pd.DataFrame({"goal": [summary["goal_amount_inr"]]})).mark_rule(
    color="#b33a3a", strokeDash=[6, 4]
).encode(y="goal:Q")
st.altair_chart((line + band + median + goal_rule).properties(height=260), width="stretch")
st.markdown("##### Synthetic-return bootstrap sensitivity")
bootstrap_cols = st.columns(3)
for col, label, key in zip(
    bootstrap_cols,
    ["5th percentile", "Median", "95th percentile"],
    ["bootstrap_projection_p05_inr", "bootstrap_projection_p50_inr", "bootstrap_projection_p95_inr"],
):
    col.metric(label, f"₹{summary[key]:,.0f}", f"vs goal ₹{summary[key]-summary['goal_amount_inr']:,.0f}")
st.caption(
    f"Paired circular block bootstrap of {summary['bootstrap_history_months']} synthetic months "
    f"({summary['bootstrap_replicates']:,} paths, {summary['bootstrap_block_months']}-month blocks). "
    "The same sampled months are used across products to preserve their observed co-movement. "
    "This shows path variability only, not assumption error; the required ±3pp range separately represents assumption error. "
    "For long goals, use the wider of the bootstrap and ±3pp ranges as the planning envelope."
)
if int(summary["horizon_months"]) > 84:
    envelope_low = min(summary["bootstrap_projection_p05_inr"], summary["projection_downside_inr"])
    envelope_high = max(summary["bootstrap_projection_p95_inr"], summary["projection_upside_inr"])
    st.info(f"Long-goal planning envelope (wider range): ₹{envelope_low:,.0f}–₹{envelope_high:,.0f}.")
st.info(summary["actionable_adjustment"])
st.caption(f"Weighted assumed return {summary['weighted_annual_return_estimate']:.2%}. {summary['projection_label']}")
st.caption(f"Synthetic-history consistency check: {summary['consistency_check']}")

st.markdown("#### Why this portfolio")
for line in summary["explainability"].split(" | "):
    st.write(f"• {line}")
detail = allocation[["product_id", "product_name", "why_included", "constraint_served", "key_downside", "excluded_alternative"]].copy()
detail = detail.rename(columns={
    "product_id": "Product ID", "product_name": "Product", "why_included": "Why included",
    "constraint_served": "Constraint served", "key_downside": "Key downside",
    "excluded_alternative": "Product comparison and alternative rationale",
})
st.dataframe(detail, width="stretch", hide_index=True)

st.markdown("#### What-if simulator")
st.write("Asset-class shock affects current holdings only; allocation weights stay fixed because suitability inputs are unchanged.")
scenario = st.selectbox("Scenario", ["15% asset-class market shock", "20% SIP reduction",
                                     "Shorter horizon by 12 months", "Change risk tolerance", "Custom adjustments"])
scenario_balance = float(profile.existing_investment_inr)
scenario_sip = float(profile.monthly_sip_inr)
scenario_horizon = int(profile.horizon_months)
scenario_risk = float(profile.risk_tolerance_score)
shock = 0.0
if scenario == "15% asset-class market shock":
    shock = -0.15
elif scenario == "20% SIP reduction":
    scenario_sip *= 0.8
elif scenario == "Shorter horizon by 12 months":
    scenario_horizon = shorter_horizon_months(scenario_horizon)
elif scenario == "Change risk tolerance":
    scenario_risk = st.slider("Adjusted risk tolerance", 1.0, 5.0, scenario_risk, 0.5)
else:
    scenario_balance = st.number_input("Starting balance (₹)", 0.0, max(5_000_000.0, scenario_balance * 3), scenario_balance, 10_000.0)
    scenario_sip = st.number_input("Monthly SIP (₹)", 0.0, max(500_000.0, scenario_sip * 3), scenario_sip, 1_000.0)
    scenario_horizon = st.slider("Goal horizon (months)", 1, 360, scenario_horizon)
    scenario_risk = st.slider("Risk tolerance", 1.0, 5.0, scenario_risk, 0.5)

try:
    scenario_allocation, scenario_summary = cached_plan(
        str(DATA_DIR), selected_id, scenario_balance, scenario_sip,
        scenario_horizon, scenario_risk, shock,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Scenario projection", f"₹{scenario_summary['projection_base_inr']:,.0f}",
              f"Δ ₹{scenario_summary['projection_base_inr']-summary['projection_base_inr']:,.0f}")
    c2.metric("Scenario risk", f"{scenario_summary['weighted_risk_score']:.2f}", scenario_summary["planning_status"])
    c3.metric("Scenario goal status", scenario_summary["goal_status_label"])
    scenario_name = scenario
    narrative = scenario_narrative(allocation, scenario_allocation, summary, scenario_summary, scenario_name)
    risk_tolerance_changed = abs(float(scenario_summary["risk_tolerance_score"]) - float(summary["risk_tolerance_score"])) > 1e-8
    before_mix = allocation.set_index("product_id")[["balance_weight_pct", "sip_weight_pct"]]
    after_mix = scenario_allocation.set_index("product_id")[["balance_weight_pct", "sip_weight_pct"]]
    mix_delta = after_mix.sub(before_mix, fill_value=0).abs().to_numpy().sum()
    if risk_tolerance_changed and mix_delta < 1e-7:
        binding = str(scenario_summary.get("binding_constraint", "none")).replace("_", " ")
        binding_text = "no other hard constraint" if binding == "none" else binding
        st.warning(
            f"Risk tolerance is not binding here; {binding_text} remains binding. "
            "Changing tolerance did not change product or SIP weights."
        )
    st.info(narrative)
    projection_comparison = pd.DataFrame([
        {"Case": label, "Before (₹)": summary[key], "After (₹)": scenario_summary[key],
         "Change (₹)": scenario_summary[key] - summary[key]}
        for label, key in (("Downside", "projection_downside_inr"),
                           ("Base", "projection_base_inr"),
                           ("Upside", "projection_upside_inr"),
                           ("Base surplus / shortfall", "base_shortfall_surplus_inr"))
    ])
    st.dataframe(projection_comparison, hide_index=True, width="stretch")
    delta = (scenario_allocation.set_index("product_id")["weight_pct"]
             .sub(allocation.set_index("product_id")["weight_pct"], fill_value=0).reset_index())
    delta.columns = ["Product ID", "Weight change (percentage points)"]
    if abs(shock) > 0 and delta[delta.columns[1]].abs().sum() < 1e-8:
        delta["Explanation"] = "Weights stay fixed: this scenario changes current asset values, not suitability inputs."
    st.dataframe(delta, width="stretch", hide_index=True)
    st.caption(scenario_summary["market_shock_note"])
    st.caption(scenario_summary["actionable_adjustment"])
except (SuitabilityError, DataValidationError, ValueError) as exc:
    st.error(f"Scenario cannot be built: {exc}")

