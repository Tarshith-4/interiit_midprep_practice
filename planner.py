"""Planning orchestration, what-if narration and batch output generation."""
from __future__ import annotations
import json
import math
from pathlib import Path
import pandas as pd
from validation import (DATA_DIR, OUTPUT_DIR, DataValidationError, SuitabilityError, BOOTSTRAP_REPLICATES, BOOTSTRAP_BLOCK_MONTHS, RETURN_CAP, RETURN_FLOOR, RISK_SLACK, validate_profile, check_consistency, load_data)
from projection import (_mix_rates, _separate_mix_future_value, _required_sip_separate, _extra_months_separate, _bootstrap_projection_percentiles, _realized_annual)
from allocation import build_allocation, _weighted_risk, is_near_term_liquid
from explain import explain, _binding_constraint


def shorter_horizon_months(months: int, reduction: int = 12) -> int:
    """Return a shortened whole-month horizon with a one-month floor."""
    return max(1, int(months) - max(0, int(reduction)))

def plan_profile(
    profile: pd.Series,
    products: pd.DataFrame,
    returns: pd.DataFrame,
    sip_multiplier: float = 1.0,
    horizon_override: int | None = None,
    market_shock: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    p = profile.copy()
    if horizon_override is not None:
        p.horizon_months = horizon_override
    profile_errors = validate_profile(p)
    if profile_errors:
        raise DataValidationError("; ".join(profile_errors))
    allocation, status, issues = build_allocation(p, products)
    if not -1.0 <= market_shock <= 0.0:
        raise ValueError("market_shock must be between -1.0 and 0.0")
    shock_exposure = {"cash": 0.0, "debt": 0.10, "hybrid": 0.50, "equity": 1.0}
    balance = float(p.existing_investment_inr)
    shocked_principal = 0.0
    for _, row in allocation.iterrows():
        share = float(row.weight_pct) / 100.0
        asset_shock = float(market_shock) * shock_exposure.get(str(row.asset_class), 1.0)
        shocked_principal += balance * share * (1.0 + asset_shock)
    principal = shocked_principal
    if principal < 0:
        raise ValueError("Market shock cannot make principal negative")
    sip = float(p.monthly_sip_inr) * sip_multiplier
    if sip < 0:
        raise ValueError("SIP multiplier cannot produce a negative contribution")
    months = int(p.horizon_months)
    allocation = allocation.copy()
    allocation["allocated_amount_inr"] = (allocation.balance_weight_pct / 100.0 * float(p.existing_investment_inr)).round(2)
    allocation["monthly_sip_allocation_inr"] = (allocation.sip_weight_pct / 100.0 * sip).round(2)
    allocation["annual_cost_inr"] = (allocation.allocated_amount_inr * allocation.annual_expense_ratio).round(2)
    allocation["estimated_annual_return"] = allocation.assumed_annual_return
    allocation["near_term_liquid"] = allocation.apply(is_near_term_liquid, axis=1)
    shock_exposure = {"cash": 0.0, "debt": 0.10, "hybrid": 0.50, "equity": 1.0}
    allocation["market_shock_factor"] = allocation.asset_class.map(shock_exposure).fillna(1.0) * market_shock
    # Report a contribution-exposure weighted rate, while projections below
    # calculate the balance and SIP streams separately at their own rates.
    sip_exposure = sip * months
    total_exposure = balance + sip_exposure
    balance_exposure = balance / total_exposure if total_exposure > 0 else 0.0
    sip_exposure_share = sip_exposure / total_exposure if total_exposure > 0 else 0.0
    balance_rate, sip_rate = _mix_rates(allocation)
    rp = balance_exposure * balance_rate + sip_exposure_share * sip_rate if total_exposure > 0 else 0.0
    down_balance_rate = max(RETURN_FLOOR, balance_rate - 0.03)
    down_sip_rate = max(RETURN_FLOOR, sip_rate - 0.03)
    up_balance_rate = min(RETURN_CAP, balance_rate + 0.03)
    up_sip_rate = min(RETURN_CAP, sip_rate + 0.03)
    projections = {
        "downside": _separate_mix_future_value(principal, sip, down_balance_rate, down_sip_rate, months),
        "base": _separate_mix_future_value(principal, sip, balance_rate, sip_rate, months),
        "upside": _separate_mix_future_value(principal, sip, up_balance_rate, up_sip_rate, months),
    }
    bootstrap_p05, bootstrap_p50, bootstrap_p95 = _bootstrap_projection_percentiles(
        p, allocation, returns, principal, sip, months
    )
    goal_amount = float(p.goal_amount_inr)
    goal_infeasible = projections["upside"] + 0.01 < goal_amount
    balance_risk = _weighted_risk(allocation)
    sip_risk = float((allocation.sip_weight_pct / 100.0 * allocation.risk_score).sum())
    risk = balance_exposure * balance_risk + sip_exposure_share * sip_risk
    balance_expense = float((allocation.balance_weight_pct / 100.0 * allocation.annual_expense_ratio).sum())
    sip_expense = (
        float((allocation.sip_weight_pct / 100.0 * allocation.annual_expense_ratio).sum())
        if sip > 0 else balance_expense
    )
    expense = balance_expense
    net_balance_rate = max(RETURN_FLOOR, balance_rate - balance_expense)
    net_sip_rate = max(RETURN_FLOOR, sip_rate - sip_expense)
    net_base_projection = _separate_mix_future_value(principal, sip, net_balance_rate, net_sip_rate, months)
    first_year_sip_cost = sip * 5.5 * sip_expense
    liquid_amount = float(allocation.loc[allocation.near_term_liquid, "allocated_amount_inr"].sum())
    need = float(p.liquidity_need_inr)
    reasons, warnings, detail = explain(p, allocation, products)
    consistency = check_consistency(p)
    warnings.extend(consistency["warnings"])
    if issues:
        warnings = [f"FALLBACK: {'; '.join(issues)}"] + warnings
    base = projections["base"]
    gap = float(p.goal_amount_inr) - base
    sip_needed = _required_sip_separate(principal, float(p.goal_amount_inr), balance_rate, sip_rate, months)
    sip_for_margin = _required_sip_separate(principal, float(p.goal_amount_inr) * 1.05, balance_rate, sip_rate, months)
    extra_sip_for_margin = max(0.0, sip_for_margin - sip)
    extra = _extra_months_separate(principal, sip, balance_rate, sip_rate, months, float(p.goal_amount_inr))
    materiality_gap = goal_amount * 0.01
    if goal_infeasible:
        goal_status = "UPSIDE_SHORTFALL"
    elif projections["downside"] + 0.01 >= goal_amount:
        goal_status = "ON_TRACK"
    elif abs(gap) <= materiality_gap:
        goal_status = "WITHIN_1_PERCENT"
    elif base + 0.01 >= goal_amount:
        goal_status = "ON_TRACK_BASE_ONLY"
    else:
        goal_status = "SHORTFALL"
    if balance == 0 and sip == 0:
        adjustment = "No contributions are planned. Add a starting balance or monthly SIP to estimate progress toward this goal."
    elif goal_status == "ON_TRACK":
        adjustment = "Downside and base estimates reach the goal under these assumptions; outcomes are not guaranteed."
    elif goal_status == "WITHIN_1_PERCENT":
        margin_sip_text = (
            f"₹{sip_for_margin:,.0f}/month total (₹{extra_sip_for_margin:,.0f}/month more than now)"
        )
        target_gap_text = (
            f"₹{abs(gap):,.0f} short of the goal" if gap > 0
            else f"₹{abs(gap):,.0f} above the goal"
        )
        adjustment = (
            f"Within 1% of target: the base estimate is {target_gap_text}. "
            f"To add a 5% margin, estimated SIP is {margin_sip_text}. "
            "This is an estimate under the same return assumption, not a guarantee."
        )
    elif goal_status == "ON_TRACK_BASE_ONLY":
        if gap > 0:
            adjustment = (
                f"Essentially on target: the base estimate is within 1% of the goal, short by ₹{gap:,.0f}. "
                "Review the contribution or target if you want additional margin."
            )
        else:
            adjustment = "The base estimate reaches the goal, but the downside estimate does not; consider adding a margin of safety."
    elif goal_status == "UPSIDE_SHORTFALL":
        extra_txt = f" or extend the horizon by about {extra} months at the current SIP" if extra else " or extend the horizon"
        adjustment = (
            f"The base estimate is ₹{base:,.0f}; consider resetting the goal near that amount, "
            f"increasing SIP toward the base-case estimate of ₹{sip_needed:,.0f}/month{extra_txt}. "
            f"The optimistic upside estimate is ₹{projections['upside']:,.0f}; it is not the planning target."
        )
    else:
        extra_txt = f" or extend the horizon by about {extra} months" if extra else " or extend the horizon"
        adjustment = (
            f"Base estimate is short by ₹{gap:,.0f}. A planning SIP of about "
            f"₹{sip_needed:,.0f}/month (vs ₹{sip:,.0f} now){extra_txt} would close the gap "
            "under the same return assumption. This is not a guaranteed outcome."
        )
    stress_bits = []
    for _, row in allocation.iterrows():
        realized = _realized_annual(returns, row.product_id)
        if realized is None:
            continue
        if abs(realized - float(row.assumed_annual_return)) >= 0.02:
            stress_bits.append(
                f"{row.product_id} assumed {row.assumed_annual_return:.1%} vs "
                f"synthetic trailing {realized:.1%}"
            )
    asset_mix = (
        allocation.groupby("asset_class", as_index=False)["balance_weight_pct"].sum()
        .rename(columns={"balance_weight_pct": "weight_pct"})
        .sort_values("weight_pct", ascending=False)
    )
    detail_lookup = detail.set_index("product_id")
    for field in ("why_included", "constraint_served", "key_downside", "excluded_alternative"):
        allocation[field] = allocation.product_id.map(detail_lookup[field])
    liquid_sip_amount = float(allocation.loc[allocation.near_term_liquid, "monthly_sip_allocation_inr"].sum())
    has_contributions = balance > 0 or sip > 0
    liquidity_covered = (liquid_amount + 1e-6 >= need) if has_contributions else None
    sip_holds = allocation.sip_weight_pct > 0
    sip_minimums_enforced = bool(
        sip <= 0
        or (
            allocation.loc[sip_holds, "monthly_sip_allocation_inr"] + 0.01
            >= allocation.loc[sip_holds, "minimum_investment_inr"]
        ).all()
    )
    if sip > 0 and not sip_minimums_enforced:
        under_minimum = allocation.loc[
            sip_holds
            & (allocation.monthly_sip_allocation_inr + 0.01 < allocation.minimum_investment_inr),
            "product_id",
        ].astype(str).tolist()
        warnings.append(
            "The source has no SIP-specific minimum. The general minimum could not be enforced per instalment "
            "without violating portfolio guardrails for: " + ", ".join(under_minimum)
        )
    risk_fit_ok = (
        balance_risk <= min(float(p.risk_tolerance_score) + RISK_SLACK, 5.0) + 1e-6
        and sip_risk <= min(float(p.risk_tolerance_score) + RISK_SLACK, 5.0) + 1e-6
    )
    no_contributions = balance == 0 and sip == 0
    reported_status = "NEEDS_CLARIFICATION" if no_contributions else status
    no_contribution_prompt = next(
        (prompt for prompt in consistency["prompts"] if "Needs clarification:" in prompt), ""
    )
    summary = {
        "investor_id": p.investor_id,
        "goal_label": p.goal_label,
        "goal_amount_inr": float(p.goal_amount_inr),
        "horizon_months": months,
        "starting_balance_inr": float(p.existing_investment_inr),
        "shocked_principal_inr": principal,
        "monthly_sip_inr": sip,
        "weighted_risk_score": round(risk, 4),
        "balance_weighted_risk_score": round(balance_risk, 4),
        "risk_tolerance_score": float(p.risk_tolerance_score),
        "risk_limit": min(float(p.risk_tolerance_score) + RISK_SLACK, 5.0),
        "weighted_annual_expense_ratio": round(expense, 6),
        "sip_weighted_annual_expense_ratio": round(sip_expense, 6),
        "estimated_annual_cost_inr": round(float(allocation.annual_cost_inr.sum()), 2),
        "first_year_sip_product_cost_inr": round(first_year_sip_cost, 2),
        "net_weighted_annual_return_estimate": round(
            balance_exposure * net_balance_rate + sip_exposure_share * net_sip_rate, 6
        ),
        "net_projection_base_inr": net_base_projection,
        "liquid_allocation_inr": round(liquid_amount, 2),
        "liquidity_need_inr": need,
        "liquidity_covered": liquidity_covered,
        "liquid_monthly_sip_inr": round(liquid_sip_amount, 2),
        "risk_fit_ok": bool(risk_fit_ok),
        "sip_weighted_risk_score": round(sip_risk, 4),
        "sip_risk_fit_ok": bool(sip_risk <= min(float(p.risk_tolerance_score) + RISK_SLACK, 5.0) + 1e-6),
        "emergency_fund_months": float(p.emergency_fund_months),
        "emergency_buffer_applied": bool(float(p.emergency_fund_months) < 3 and float(p.existing_investment_inr) > 0),
        "emergency_buffer_sip_routing": bool(float(p.emergency_fund_months) < 3 and float(p.existing_investment_inr) == 0 and sip > 0 and liquid_sip_amount > 0),
        "locked_balance_weight_pct": float(allocation.loc[allocation.lock_in_months > 0, "balance_weight_pct"].sum()),
        "sip_minimums_enforced": sip_minimums_enforced,
        "weighted_annual_return_estimate": round(rp, 6),
        "bootstrap_projection_p05_inr": bootstrap_p05,
        "bootstrap_projection_p50_inr": bootstrap_p50,
        "bootstrap_projection_p95_inr": bootstrap_p95,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_block_months": BOOTSTRAP_BLOCK_MONTHS,
        "bootstrap_history_months": int(returns.month.nunique()),
        "bootstrap_method": "Paired circular moving-block bootstrap; product returns resampled together in 3-month blocks; synthetic sensitivity, not a forecast.",
        "projection_downside_inr": projections["downside"],
        "projection_base_inr": projections["base"],
        "projection_upside_inr": projections["upside"],
        "base_shortfall_surplus_inr": base - float(p.goal_amount_inr),
        "goal_status": goal_status,
        "goal_status_label": "Within 1% of target" if goal_status == "WITHIN_1_PERCENT" else goal_status.replace("_", " ").title(),
        "goal_materiality_note": (
            adjustment if goal_status == "WITHIN_1_PERCENT" else ""
        ),
        "goal_infeasible": bool(goal_infeasible),
        "required_sip_inr": round(sip_needed, 2),
        "sip_for_5pct_margin_inr": round(sip_for_margin, 2),
        "extra_sip_for_5pct_margin_inr": round(extra_sip_for_margin, 2),
        "extra_months_to_goal": extra,
        "actionable_adjustment": adjustment,
        "binding_constraint": _binding_constraint(p, allocation, projections["downside"], base, liquidity_covered),
        "consistency_warnings": " | ".join(consistency["warnings"]),
        "clarification_prompts": " | ".join(consistency["prompts"]),
        "warnings": " | ".join(dict.fromkeys(warnings)),
        "explainability": " | ".join(reasons),
        "explainability_detail": detail.to_dict(orient="records"),
        "asset_class_mix": asset_mix.to_dict(orient="records"),
        "projection_label": (
            "Educational planning approximation using catalogue assumed returns; "
            f"scenario range is rp−3pp / rp / rp+3pp floored at {RETURN_FLOOR:.0%} "
            f"and capped at {RETURN_CAP:.0%}. Not guaranteed returns or advice."
        ),
        "consistency_check": " | ".join(stress_bits) if stress_bits else "Synthetic trailing returns are within 2pp of assumed rates for held products.",
        "planning_status": reported_status,
        "suitability_status": status,
        "needs_clarification": no_contributions,
        "planning_exception": no_contribution_prompt or ("; ".join(issues) if issues else ""),
        "preference": p.preference,
        "market_shock": market_shock,
        "market_shock_note": "Stylized stress factors applied to existing holdings: cash 0%, debt 10%, hybrid 50%, equity 100% of the selected shock. These are transparent scenario assumptions, not calibrated forecasts; future SIPs are unchanged.",
    }
    numeric_ok = all(
        math.isfinite(float(v))
        for v in summary.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    )
    if not numeric_ok:
        raise DataValidationError("Projection produced a non-finite value")
    return allocation, summary

def plan_buffer_first(
    profile: pd.Series,
    products: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Side-by-side conservative option: reserve liquidity before long-term risk."""
    option = profile.copy()
    option["emergency_fund_months"] = 0.0
    option["preference"] = "capital_protection"
    return plan_profile(option, products, returns)

def scenario_narrative(
    before_allocation: pd.DataFrame,
    after_allocation: pd.DataFrame,
    before_summary: dict,
    after_summary: dict,
    scenario_name: str,
) -> str:
    """Explain changed inputs, the binding rule, product routing, and goal impact."""
    inputs = []
    for field, label in (
        ("monthly_sip_inr", "monthly SIP"),
        ("starting_balance_inr", "starting balance"),
        ("horizon_months", "horizon"),
        ("risk_tolerance_score", "risk tolerance"),
    ):
        old = float(before_summary[field] if field in before_summary else 0)
        new = float(after_summary[field] if field in after_summary else 0)
        if abs(old - new) > 1e-8:
            inputs.append(f"{label} changed from {old:g} to {new:g}")
    if float(after_summary.get("market_shock", 0)) != float(before_summary.get("market_shock", 0)):
        inputs.append("an asset-class market shock was applied")
    if inputs:
        change_text = ", ".join(inputs)
    else:
        change_text = scenario_name

    before = before_allocation.set_index("product_id")
    after = after_allocation.set_index("product_id")
    products = sorted(set(before.index) | set(after.index))
    before_active = set(before.index[(before.balance_weight_pct > 0) | (before.sip_weight_pct > 0)])
    after_active = set(after.index[(after.balance_weight_pct > 0) | (after.sip_weight_pct > 0)])
    added = sorted(after_active - before_active)
    removed = sorted(before_active - after_active)
    changes = []
    for product_id in products:
        for column, label in (("balance_weight_pct", "balance"), ("sip_weight_pct", "SIP")):
            old = float(before[column].get(product_id, 0.0)) if column in before else 0.0
            new = float(after[column].get(product_id, 0.0)) if column in after else 0.0
            if abs(new - old) >= 0.05:
                changes.append(f"{product_id} {label} {old:.1f}%→{new:.1f}%")
    if not changes and abs(float(before_summary.get("risk_tolerance_score", 0)) - float(after_summary.get("risk_tolerance_score", 0))) > 1e-8:
        binding = str(after_summary.get("binding_constraint", "none")).replace("_", " ")
        constraint_text = "no suitability rule" if binding == "none" else binding
        route_text = f"Risk tolerance is not binding here; {constraint_text} remains binding"
    elif changes:
        binding = str(after_summary.get("binding_constraint", "none")).replace("_", " ")
        product_delta = []
        if added:
            product_delta.append("added " + ", ".join(added))
        if removed:
            product_delta.append("removed " + ", ".join(removed))
        product_note = ("; products " + "; ".join(product_delta)) if product_delta else ""
        route_text = f"Binding constraint: {binding}; product routing changed: " + ", ".join(changes[:5]) + product_note
    elif float(after_summary.get("market_shock", 0)) != 0:
        route_text = "Asset-class stress changed the current balance value; product and SIP weights stayed fixed"
    else:
        binding = str(after_summary.get("binding_constraint", "none")).replace("_", " ")
        route_text = f"No product or SIP mix changed; binding constraint remains {binding}"

    before_values = [before_summary[key] for key in (
        "projection_downside_inr", "projection_base_inr", "projection_upside_inr", "base_shortfall_surplus_inr"
    )]
    after_values = [after_summary[key] for key in (
        "projection_downside_inr", "projection_base_inr", "projection_upside_inr", "base_shortfall_surplus_inr"
    )]
    metric_names = ("downside", "base", "upside", "base surplus/shortfall")
    metric_text = "; ".join(
        f"{name} ₹{old:,.0f}→₹{new:,.0f}"
        for name, old, new in zip(metric_names, before_values, after_values)
    )
    return f"{change_text}. {route_text}; {metric_text}."

def run_all(data_dir: str | Path = DATA_DIR, *, loader=load_data) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles, products, returns = loader(data_dir)
    allocations, summaries = [], []
    for _, profile in profiles.iterrows():
        try:
            alloc, summary = plan_profile(profile, products, returns)
            allocations.append(alloc.assign(investor_id=profile.investor_id, goal_label=profile.goal_label))
        except (SuitabilityError, DataValidationError) as exc:
            def numeric_or_none(field: str):
                try:
                    value = float(profile.get(field))
                except (TypeError, ValueError):
                    return None
                return value if math.isfinite(value) else None

            summary = {
                "investor_id": profile.investor_id,
                "goal_label": profile.goal_label,
                "goal_amount_inr": numeric_or_none("goal_amount_inr"),
                "horizon_months": numeric_or_none("horizon_months"),
                "planning_status": "INVALID_PROFILE" if isinstance(exc, DataValidationError) else "SUITABILITY_EXCEPTION",
                "planning_exception": str(exc),
                "warnings": "No allocation issued; review constraints and available products.",
                "projection_label": "Educational planning approximation; not guaranteed returns.",
            }
        summaries.append(summary)
    allocation_frame = pd.concat(allocations, ignore_index=True) if allocations else pd.DataFrame()
    return allocation_frame, pd.DataFrame(summaries)

def export_outputs(data_dir: str | Path = DATA_DIR, output_dir: str | Path = OUTPUT_DIR, *, loader=load_data) -> tuple[Path, Path]:
    allocations, summaries = run_all(data_dir, loader=loader)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_export = summaries.copy()
    for col in ("explainability_detail", "asset_class_mix"):
        if col in summary_export.columns:
            summary_export[col] = summary_export[col].apply(
                lambda x: x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
            )
    ap, sp = out / "allocations_output.csv", out / "summary_output.csv"
    # Avoid exporting the legacy balance-only alias; keep the two routed mixes explicit.
    allocations.drop(columns=["weight_pct"], errors="ignore").to_csv(ap, index=False)
    summary_export.to_csv(sp, index=False)
    return ap, sp

__all__ = ['plan_profile', 'plan_buffer_first', 'scenario_narrative', 'shorter_horizon_months', 'run_all', 'export_outputs']

