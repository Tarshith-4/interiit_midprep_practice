"""Plain-language reasons, alternatives, warnings and binding constraints."""
from __future__ import annotations
import pandas as pd
from validation import RISK_SLACK, EMERGENCY_BUFFER_PCT
from allocation import (_risk_limits, _weighted_risk, _preference, _goal_value, is_near_term_liquid)

def _downside(row: pd.Series) -> str:
    if row.asset_class == "cash":
        return "Cash drag: expected return is below most goal-required rates."
    if row.asset_class == "debt" and row.lock_in_months > 0:
        return "Lock-in: the amount cannot be withdrawn until maturity."
    if row.asset_class == "debt":
        return "Interest-rate and credit-spread moves can reduce NAV before the goal."
    if row.asset_class == "hybrid":
        return "Equity sleeve can fall in a market shock even though the fund looks 'balanced'."
    if row.risk_score >= 5:
        return "High volatility and theme concentration can produce large drawdowns."
    if row.asset_class == "equity":
        return "Market drawdowns can be large and may not recover inside a short horizon."
    return "Value can fall; this is not a guaranteed payout."

def _alternative_limit_reason(
    product: pd.Series,
    chosen: pd.Series,
    profile: pd.Series,
    allocation: pd.DataFrame,
) -> str:
    horizon = int(profile.horizon_months)
    if int(product.lock_in_months) + float(product.liquidity_days) / 30.0 > horizon:
        return "its lock-in plus settlement would finish after the goal date"
    preference = _preference(profile)
    if preference == "avoid_equity" and product.asset_class == "equity":
        return "the profile explicitly avoids equity"
    if preference == "no_lock_in" and int(product.lock_in_months) > 0:
        return "the profile explicitly avoids lock-ins"
    if float(profile.monthly_sip_inr) > 0 and int(product.lock_in_months) > 0:
        return "recurring SIPs cannot enter locked products under the goal-date rule"
    high_cap, _ = _risk_limits(horizon)
    high_balance = float(allocation.loc[allocation.risk_score >= 4, "balance_weight_pct"].sum()) / 100.0
    high_sip = float(allocation.loc[allocation.risk_score >= 4, "sip_weight_pct"].sum()) / 100.0
    if product.risk_score >= 4 and max(high_balance, high_sip) >= high_cap - 1e-6:
        return f"adding it would exceed the {high_cap:.0%} horizon ceiling on risk 4–5 products"
    if product.risk_score >= 4 and _goal_value(profile, allocation) + 0.01 >= float(profile.goal_amount_inr):
        return "the base goal is already funded without this extra high-risk exposure, so the least-risk rule excludes it"
    alt_score = float(product.assumed_annual_return - product.annual_expense_ratio
                      - 0.10 * product.volatility_proxy - 0.005 * product.risk_score)
    chosen_score = float(chosen.assumed_annual_return - chosen.annual_expense_ratio
                         - 0.10 * chosen.volatility_proxy - 0.005 * chosen.risk_score)
    if preference == "low_cost" and product.annual_expense_ratio > chosen.annual_expense_ratio:
        return f"the low_cost preference favors the selected {chosen.product_id} fee ({chosen.annual_expense_ratio:.2%} vs {product.annual_expense_ratio:.2%})"
    if preference == "high_growth" and product.assumed_annual_return < chosen.assumed_annual_return:
        return (f"the high_growth preference favors the selected {chosen.product_id} return "
                f"({chosen.assumed_annual_return:.1%} vs {product.assumed_annual_return:.1%}), within the risk ceiling")
    if preference == "tax_saving" and chosen.product_id == "PROD07":
        return "the tax_saving preference prioritizes the selected tax-saver product within the risk ceiling"
    if alt_score < chosen_score:
        return (f"its sleeve score is lower ({alt_score:.4f} vs {chosen_score:.4f} for {chosen.product_id}); "
                "the score accounts for assumed return, annual fee, volatility, and ordinal risk")
    return (f"it is not a one-for-one substitute under the profile's liquidity, risk, and diversification rules; "
            f"the selected {chosen.product_id} serves a different portfolio role")

def _constraint_for(row: pd.Series, profile: pd.Series) -> str:
    if is_near_term_liquid(row):
        if float(profile.emergency_fund_months) < 3:
            return "Emergency buffer and near-term liquidity"
        if float(profile.liquidity_need_inr) > 0:
            return "Stated near-term liquidity need"
        return "Defensive / liquid sleeve"
    if row.risk_score <= 2:
        return "Horizon low-risk floor and capital stability"
    if row.risk_score == 3:
        return "Balanced growth without using the full high-risk cap"
    if _preference(profile) == "tax_saving" and row.product_id == "PROD07":
        return "Tax-saving preference, inside the high-risk cap"
    if row.risk_score >= 4:
        return "Horizon growth sleeve (capped)"
    return "Diversification"

def explain(profile: pd.Series, allocation: pd.DataFrame, products: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    reasons = [
        f"{profile.goal_label} has a {int(profile.horizon_months)}-month horizon, so the "
        f"{_risk_limits(int(profile.horizon_months))[0]:.0%} high-risk cap and "
        f"{_risk_limits(int(profile.horizon_months))[1]:.0%} low-risk floor apply.",
        f"Investor risk tolerance is {profile.risk_tolerance_score:g}/5; weighted portfolio "
        f"risk is {_weighted_risk(allocation):.2f} (limit {min(float(profile.risk_tolerance_score)+RISK_SLACK, 5):.2f}).",
    ]
    warnings: list[str] = []
    pref = _preference(profile)
    if pref:
        reasons.append(f"Preference '{profile.preference}' is applied only after hard suitability rules.")
    if pref == "avoid_equity" and float(profile.risk_tolerance_score) >= 4:
        warnings.append("High stated risk tolerance does not override the explicit avoid_equity preference.")
    if pref == "avoid_equity":
        hybrid_balance = 0.5 * float(allocation.loc[allocation.asset_class == "hybrid", "balance_weight_pct"].sum())
        hybrid_sip = 0.5 * float(allocation.loc[allocation.asset_class == "hybrid", "sip_weight_pct"].sum())
        if hybrid_balance > 0 or hybrid_sip > 0:
            warnings.append(
                f"avoid_equity excludes pure-equity products, but the selected hybrid fund is roughly 50% equity; "
                f"estimated indirect equity exposure is {hybrid_balance:.1f}% of balance and {hybrid_sip:.1f}% of SIPs."
            )
    if str(profile.experience).lower() == "beginner":
        reasons.append("Beginner experience is treated as a reason to avoid stretching the high-risk cap.")
    if float(profile.emergency_fund_months) < 3:
        if float(profile.existing_investment_inr) > 0:
            warnings.append(
                f"Emergency fund is only {profile.emergency_fund_months:g} months. "
                f"At least {EMERGENCY_BUFFER_PCT:.0%} of existing capital is assigned to the defensive liquid sleeve; "
                "this reserve overlaps with, rather than adds to, the stated liquidity need. Locked products are capped at 0% until the emergency fund reaches 3 months."
            )
        elif float(profile.monthly_sip_inr) > 0:
            warnings.append(
                f"Emergency fund is only {profile.emergency_fund_months:g} months and no starting capital is available. "
                "The selected allocation routes part of future SIP instalments to liquid products; no buffer is available today. The 20% SIP reserve overlaps with the stated liquidity need. Locked products are capped at 0% until the emergency fund reaches 3 months."
            )
        else:
            warnings.append(
                f"Emergency fund is only {profile.emergency_fund_months:g} months and no capital or SIP is available to fund a buffer."
            )
    if float(profile.liquidity_need_inr) > 0 and (
        float(profile.existing_investment_inr) > 0 or float(profile.monthly_sip_inr) > 0
    ):
        liquid_amt = float(allocation.loc[allocation.apply(is_near_term_liquid, axis=1), "allocated_amount_inr"].sum()) if "allocated_amount_inr" in allocation.columns else 0.0
        covered = liquid_amt + 1e-6 >= float(profile.liquidity_need_inr)
        warnings.append(
            f"Near-term liquidity need is ₹{profile.liquidity_need_inr:,.0f}. "
            f"Unlocked cash/short-debt sleeve is ₹{liquid_amt:,.0f} "
            f"({'covered' if covered else 'shortfall — growth assets are not treated as cash'})."
        )
    if float(profile.existing_investment_inr) == 0 and float(profile.monthly_sip_inr) == 0:
        warnings.append("No contributions are planned, so neither the goal nor a liquidity need can be funded by this portfolio yet.")
    if pref == "high_growth" and int(profile.horizon_months) <= 36:
        warnings.append(
            "high_growth preference is overridden by the short-horizon 10% cap on risk 4–5 products."
        )
    if pref == "tax_saving" and float(profile.monthly_sip_inr) > 0:
        warnings.append("Tax-saver products with lock-ins are excluded from recurring SIP routing so later instalments can settle by the goal date.")
    selected = set(allocation.product_id)
    excluded_notes = []
    for _, product in products.iterrows():
        if product.product_id in selected:
            continue
        if int(product.lock_in_months) > int(profile.horizon_months):
            excluded_notes.append(
                f"{product.product_name} excluded because its {int(product.lock_in_months)}-month "
                "lock-in extends beyond the goal date."
            )
        elif pref == "avoid_equity" and product.asset_class == "equity":
            excluded_notes.append(
                f"{product.product_name} excluded to honour avoid_equity after the horizon floor was met."
            )
        elif pref == "no_lock_in" and int(product.lock_in_months) > 0:
            excluded_notes.append(
                f"{product.product_name} excluded because the profile requests no lock-in."
            )
        elif float(profile.monthly_sip_inr) > 0 and int(product.lock_in_months) > 0:
            excluded_notes.append(
                f"{product.product_name} excluded from recurring SIP routing because later instalments would remain locked past the goal."
            )
        elif int(product.lock_in_months) + float(product.liquidity_days) / 30.0 > int(profile.horizon_months):
            excluded_notes.append(
                f"{product.product_name} excluded because lock-in plus settlement would finish after the goal date."
            )
        elif product.risk_score >= 5:
            excluded_notes.append(
                f"{product.product_name} limited/excluded: risk 5, {product.volatility_proxy:.0%} volatility "
                f"proxy and {product.annual_expense_ratio:.2%} expense would not improve suitability."
            )
    warnings.extend(excluded_notes[:3])

    rows = []
    alternatives = products.loc[~products.product_id.isin(selected)].copy()
    for _, row in allocation.iterrows():
        balance_pct = float(row.balance_weight_pct)
        sip_pct = float(row.sip_weight_pct)
        flows = []
        if balance_pct > 0:
            flows.append(f"{balance_pct:.1f}% of current balance")
        if sip_pct > 0:
            flows.append(f"{sip_pct:.1f}% of each SIP")
        flow_text = " and ".join(flows)
        if is_near_term_liquid(row):
            why = f"{row.product_name} receives {flow_text} as the designated cash/short-duration sleeve for liquidity and the low-risk floor."
        elif row.asset_class == "debt" and row.lock_in_months > 0:
            why = f"{row.product_name} receives {flow_text} because this existing balance matures before the goal; new SIP instalments are routed elsewhere."
        elif row.asset_class == "debt":
            why = f"{row.product_name} receives {flow_text} to meet the low-risk floor; it is not counted as near-term cash because it is a medium-duration credit fund."
        elif row.asset_class == "hybrid":
            why = f"{row.product_name} receives {flow_text} to add growth at risk {int(row.risk_score)} without using more high-risk capacity."
        elif _preference(profile) == "high_growth" and str(row.product_id) == "PROD08":
            why = (
                f"{row.product_name} is preferred over PROD06 for the explicit high_growth choice: "
                f"its assumed return is {row.assumed_annual_return:.1%} versus 11.0%, while its "
                "1.6% fee (versus 0.2%) and risk 5 (versus 4) are material costs accepted by this preference."
            )
        else:
            why = f"{row.product_name} contributes {row.assumed_annual_return:.1%} catalogue return; risk 4–5 exposure remains within the {_risk_limits(int(profile.horizon_months))[0]:.0%} horizon ceiling."
        if alternatives.empty:
            alternative = "No unselected catalogue product is available for a direct comparison."
        else:
            peers = alternatives.loc[
                (alternatives.asset_class == row.asset_class)
                & (alternatives.risk_score == row.risk_score)
            ]
            if peers.empty:
                peers = alternatives.loc[alternatives.asset_class == row.asset_class]
            if peers.empty:
                peers = alternatives
            peers = peers.copy()
            peers["_risk_distance"] = (peers.risk_score - row.risk_score).abs()
            alt = peers.sort_values(
                ["_risk_distance", "assumed_annual_return", "product_id"],
                ascending=[True, False, True],
            ).iloc[0]
            reason = _alternative_limit_reason(alt, row, profile, allocation)
            alternative = (
                f"{row.product_id} vs {alt.product_id}: risk {int(row.risk_score)} vs {int(alt.risk_score)}, "
                f"assumed return {row.assumed_annual_return:.1%} vs {alt.assumed_annual_return:.1%}, "
                f"annual fee {row.annual_expense_ratio:.2%} vs {alt.annual_expense_ratio:.2%}. "
                f"{alt.product_id} not selected because {reason}."
            )
        rows.append({
            "product_id": row.product_id,
            "product_name": row.product_name,
            "why_included": why,
            "constraint_served": _constraint_for(row, profile),
            "key_downside": _downside(row),
            "excluded_alternative": alternative,
        })
    return reasons, warnings, pd.DataFrame(rows)

def _binding_constraint(
    profile: pd.Series,
    allocation: pd.DataFrame,
    downside: float,
    base: float,
    liquidity_covered: bool | None,
) -> str:
    if liquidity_covered is False:
        return "liquidity"
    high_cap, _ = _risk_limits(int(profile.horizon_months))
    high_balance = float(allocation.loc[allocation.risk_score >= 4, "balance_weight_pct"].sum()) / 100.0
    high_sip = float(allocation.loc[allocation.risk_score >= 4, "sip_weight_pct"].sum()) / 100.0
    if max(high_balance, high_sip) >= high_cap - 0.005:
        return "horizon_cap"
    risk_limit = min(float(profile.risk_tolerance_score) + RISK_SLACK, 5.0)
    balance_risk = _weighted_risk(allocation)
    sip_risk = float((allocation.sip_weight_pct / 100.0 * allocation.risk_score).sum())
    if max(balance_risk, sip_risk) >= risk_limit - 0.05:
        return "tolerance_limit"
    preference = _preference(profile)
    if preference == "avoid_equity" and not (allocation.asset_class == "equity").any():
        return "preference"
    if preference == "no_lock_in" and not (allocation.lock_in_months > 0).any():
        return "preference"
    goal = float(profile.goal_amount_inr)
    if goal > 0 and (base <= goal * 1.05 or downside < goal):
        return "goal_need"
    return "none"

__all__ = ['_downside', '_alternative_limit_reason', '_constraint_for', 'explain', '_binding_constraint']

