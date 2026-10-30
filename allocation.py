"""Suitability checks and constrained portfolio construction."""
from __future__ import annotations
from itertools import combinations
import math
import pandas as pd
from validation import (DataValidationError, SuitabilityError, PRODUCT_CAP, RISK_SLACK, RETURN_CAP, RETURN_FLOOR, EMERGENCY_BUFFER_PCT, NEAR_TERM_MAX_DAYS, validate_profile, validate_allocations)
from projection import _mix_rates, _separate_mix_future_value

def _risk_limits(months: int) -> tuple[float, float]:
    if months <= 36:
        return 0.10, 0.70
    if months <= 84:
        return 0.30, 0.45
    return 0.55, 0.25

def is_near_term_liquid(row: pd.Series) -> bool:
    """Only cash and the designated short-duration debt fund count as liquid."""
    return (
        str(row.product_id) in {"PROD01", "PROD03"}
        and
        int(row.lock_in_months) == 0
        and int(row.liquidity_days) <= NEAR_TERM_MAX_DAYS
        and float(row.risk_score) <= 2
        and str(row.asset_class) in {"cash", "debt"}
    )

def _preference(profile: pd.Series) -> str:
    return str(profile.preference).strip().lower()

def _eligible_products(products: pd.DataFrame, profile: pd.Series, apply_preference: bool, *, for_sip: bool = False) -> pd.DataFrame:
    horizon = int(profile.horizon_months)
    # Include settlement time after lock-in. SIP instalments must all mature by the goal.
    settlement_months = products.liquidity_days / 30.0
    within_goal = products.lock_in_months + settlement_months <= horizon
    # With an emergency fund below three months, keep invested capital unlocked.
    # This conservative policy sets the locked sleeve cap to 0%.
    if float(profile.emergency_fund_months) < 3:
        within_goal &= products.lock_in_months == 0
    if for_sip and float(profile.monthly_sip_inr) > 0:
        within_goal &= products.lock_in_months == 0
    usable = products.loc[within_goal].copy()
    pref = _preference(profile)
    if apply_preference and pref == "no_lock_in":
        usable = usable.loc[usable.lock_in_months == 0]
    if apply_preference and pref == "avoid_equity":
        usable = usable.loc[usable.asset_class != "equity"]
    return usable

def _sort_products(frame: pd.DataFrame, preference: str, sleeve: str) -> pd.DataFrame:
    df = frame.copy()
    if df.empty:
        return df
    if sleeve == "high" and preference == "tax_saving":
        df["_pref"] = (df.product_id != "PROD07").astype(int)
        return df.sort_values(["_pref", "annual_expense_ratio", "product_id"])
    if sleeve == "high" and preference == "high_growth":
        return df.sort_values(["assumed_annual_return", "product_id"], ascending=[False, True])
    if preference == "low_cost":
        return df.sort_values(["annual_expense_ratio", "risk_score", "product_id"])
    if preference == "capital_protection":
        return df.sort_values(["risk_score", "volatility_proxy", "annual_expense_ratio", "product_id"])
    # Suitability bands are the outer constraint. Within them, balance return,
    # fees, volatility, and ordinal product risk instead of filling by lowest risk.
    df["_quality"] = (
        df.assumed_annual_return - df.annual_expense_ratio
        - 0.10 * df.volatility_proxy - 0.005 * df.risk_score
    )
    return df.sort_values(["_quality", "risk_score", "product_id"], ascending=[False, True, True])

def _round_weights(weights: dict[str, float], decimals: int = 1) -> dict[str, float]:
    if not weights:
        return {}
    pct = {pid: round(w * 100, decimals) for pid, w in weights.items() if w > 1e-9}
    if not pct:
        return {}
    drift = round(100.0 - sum(pct.values()), decimals)
    top = max(pct, key=pct.get)
    pct[top] = round(pct[top] + drift, decimals)
    return {pid: v / 100.0 for pid, v in pct.items() if v > 0}

def _distribute(pool: pd.DataFrame, target: float, preference: str, sleeve: str, weights: dict[str, float]) -> float:
    """Fill up to `target` from pool without exceeding the 40% product cap. Returns unfilled remainder."""
    if target <= 1e-9 or pool.empty:
        return target
    ordered = _sort_products(pool, preference, sleeve)
    remaining = target
    for _, row in ordered.iterrows():
        if remaining <= 1e-9:
            break
        pid = row.product_id
        room = PRODUCT_CAP - weights.get(pid, 0.0)
        add = min(remaining, max(0.0, room))
        if add > 1e-9:
            weights[pid] = weights.get(pid, 0.0) + add
            remaining -= add
    return remaining

def _frame_from_weights(products: pd.DataFrame, weights: dict[str, float], *, decimals: int = 1) -> pd.DataFrame:
    weights = _round_weights(weights, decimals=decimals)
    rows = []
    for pid, w in weights.items():
        product = products.loc[products.product_id == pid].iloc[0]
        rows.append({**product.to_dict(), "weight_pct": w * 100.0})
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["risk_score", "product_id"]).reset_index(drop=True)
        validate_allocations(result)
    return result

def _weighted_risk(allocation: pd.DataFrame) -> float:
    return float((allocation.weight_pct / 100.0 * allocation.risk_score).sum())

def _liquid_share(allocation: pd.DataFrame) -> float:
    mask = allocation.apply(is_near_term_liquid, axis=1)
    return float(allocation.loc[mask, "weight_pct"].sum() / 100.0)

def _horizon_ok(allocation: pd.DataFrame, months: int) -> tuple[bool, float, float]:
    max_high, min_low = _risk_limits(months)
    high = float(allocation.loc[allocation.risk_score >= 4, "weight_pct"].sum() / 100.0)
    low = float(allocation.loc[allocation.risk_score <= 2, "weight_pct"].sum() / 100.0)
    return high <= max_high + 1e-6 and low >= min_low - 1e-6, high, low

def _derisk(allocation: pd.DataFrame, products: pd.DataFrame, max_risk: float, months: int) -> pd.DataFrame:
    """Move weight from the riskiest names into the cheapest room under the 40% cap."""
    weights = {row.product_id: row.weight_pct / 100.0 for _, row in allocation.iterrows()}
    max_high, min_low = _risk_limits(months)
    usable = products.loc[products.product_id.isin(weights) | (products.lock_in_months <= months)].copy()
    low_ids = set(usable.loc[usable.risk_score <= 2, "product_id"])
    for _ in range(80):
        alloc = _frame_from_weights(products, weights)
        if alloc.empty:
            break
        risk = _weighted_risk(alloc)
        ok, high, low = _horizon_ok(alloc, months)
        if risk <= max_risk + 1e-9 and ok:
            break
        source = alloc.sort_values(["risk_score", "weight_pct"], ascending=[False, False]).iloc[0]
        # Prefer parking in low-risk names that already appear, else any low-risk with room.
        sinks = _sort_products(alloc.loc[alloc.product_id.isin(low_ids)], "", "low")
        moved = False
        # Move the minimum amount needed to satisfy the stricter of risk fit and
        # horizon bands; a fixed 1% loop made batch review needlessly slow.
        risk_excess = max(0.0, risk - max_risk)
        high_excess = max(0.0, high - max_high)
        low_deficit = max(0.0, min_low - low)
        source_risk = float(source.risk_score)
        step = max(0.001, risk_excess / max(0.1, source_risk - 1.0))
        if source.risk_score >= 4 and high_excess > 0:
            step = max(step, high_excess)
        if source.risk_score <= 2 and low_deficit > 0:
            step = min(step, max(0.001, low_deficit))
        src_w = weights.get(source.product_id, 0.0)
        step = min(step, src_w)
        if step <= 1e-9:
            break
        candidates = list(sinks.product_id) + [pid for pid in low_ids if pid not in weights]
        for sink_id in candidates:
            if sink_id == source.product_id:
                continue
            room = PRODUCT_CAP - weights.get(sink_id, 0.0)
            add = min(step, max(0.0, room))
            if add <= 1e-9:
                continue
            weights[source.product_id] = src_w - add
            weights[sink_id] = weights.get(sink_id, 0.0) + add
            moved = True
            break
        if not moved:
            break
        if low - 1e-6 < min_low and source.risk_score <= 2:
            break
        if high + 1e-6 > max_high:
            continue
    return _frame_from_weights(products, weights)

def _safest_weights(profile: pd.Series, products: pd.DataFrame, *, for_sip: bool = False) -> dict[str, float]:
    usable = _eligible_products(products, profile, apply_preference=False, for_sip=for_sip)
    if usable.empty:
        raise SuitabilityError("No products mature on or before the goal date")
    pref = "capital_protection"
    weights: dict[str, float] = {}
    liquid_pool = usable[usable.apply(is_near_term_liquid, axis=1)]
    low_pool = usable[usable.risk_score <= 2]
    # Cover liquidity first, then fill remaining capacity with the lowest-risk names.
    balance = float(profile.existing_investment_inr)
    need = float(profile.liquidity_need_inr)
    liquid_target = min(1.0, need / balance) if balance > 0 else 0.0
    if float(profile.emergency_fund_months) < 3 and (balance > 0 or float(profile.monthly_sip_inr) > 0):
        liquid_target = max(liquid_target, EMERGENCY_BUFFER_PCT)
    _distribute(liquid_pool, liquid_target, pref, "low", weights)
    _distribute(low_pool, max(0.0, 1.0 - sum(weights.values())), pref, "low", weights)
    leftover = max(0.0, 1.0 - sum(weights.values()))
    leftover = _distribute(usable, leftover, pref, "low", weights)
    if leftover > 1e-6:
        raise SuitabilityError("Unable to build a diversified fallback under the 40% product cap")
    return weights

def _target_sleeves(profile: pd.Series) -> dict[str, float]:
    months = int(profile.horizon_months)
    max_high, min_low = _risk_limits(months)
    pref = _preference(profile)
    balance = float(profile.existing_investment_inr)
    need = float(profile.liquidity_need_inr)
    liquid = min(1.0, need / balance) if balance > 0 else 0.0
    if float(profile.emergency_fund_months) < 3 and (balance > 0 or float(profile.monthly_sip_inr) > 0):
        liquid = max(liquid, EMERGENCY_BUFFER_PCT)
    low = max(min_low, liquid)
    remaining = max(0.0, 1.0 - low)
    high = 0.0
    mid = 0.0
    if pref in {"capital_protection"} or float(profile.risk_tolerance_score) <= 1:
        low = 1.0
    elif pref == "avoid_equity":
        mid = min(remaining, 0.20)
        low = 1.0 - mid
        high = 0.0
    elif pref == "high_growth":
        high = min(remaining, max_high)
        mid = remaining - high
    else:
        high = min(remaining, max_high)
        leftover = remaining - high
        mid = min(leftover, 0.20)
        low = 1.0 - mid - high
    # Experience: beginners do not get extra high-risk beyond the horizon cap.
    if str(profile.experience).lower() == "beginner":
        high = min(high, max_high)
    liquid = min(liquid, low)
    other_low = max(0.0, low - liquid)
    return {"liquid": liquid, "other_low": other_low, "mid": mid, "high": high}

def _build_primary(profile: pd.Series, products: pd.DataFrame, *, for_sip: bool = False) -> pd.DataFrame:
    usable = _eligible_products(products, profile, apply_preference=True, for_sip=for_sip)
    if usable.empty:
        usable = _eligible_products(products, profile, apply_preference=False, for_sip=for_sip)
    if usable.empty:
        raise SuitabilityError("No products mature on or before the goal date")
    pref = _preference(profile)
    sleeves = _target_sleeves(profile)
    liquid_pool = usable[usable.apply(is_near_term_liquid, axis=1)]
    low_pool = usable[usable.risk_score <= 2]
    mid_pool = usable[usable.risk_score == 3]
    high_pool = usable[usable.risk_score >= 4]
    weights: dict[str, float] = {}
    _distribute(liquid_pool, sleeves["liquid"], pref, "low", weights)
    filled_liquid = sum(weights.values())
    other_low = sleeves["other_low"] + max(0.0, sleeves["liquid"] - filled_liquid)
    _distribute(low_pool, other_low, pref, "low", weights)
    _distribute(mid_pool, sleeves["mid"], pref, "mid", weights)
    _distribute(high_pool, sleeves["high"], pref, "high", weights)
    leftover = 1.0 - sum(weights.values())
    if leftover > 1e-6:
        _distribute(low_pool, leftover, pref, "low", weights)
        leftover = 1.0 - sum(weights.values())
    if leftover > 1e-6:
        _distribute(usable, leftover, pref, "low", weights)
        leftover = 1.0 - sum(weights.values())
    if leftover > 1e-6:
        raise SuitabilityError("Unable to allocate 100% within the 40% product cap")
    allocation = _frame_from_weights(products, weights)
    if allocation.asset_class.nunique() < 2:
        extra = usable.loc[~usable.product_id.isin(allocation.product_id)].sort_values(
            ["risk_score", "annual_expense_ratio"]
        )
        if extra.empty:
            raise SuitabilityError("Cannot meet two-category diversification")
        donor = allocation.sort_values("weight_pct", ascending=False).iloc[0]
        take = min(0.10, donor.weight_pct / 100.0 - 0.01)
        if take > 0:
            weights[donor.product_id] = donor.weight_pct / 100.0 - take
            weights[extra.iloc[0].product_id] = take
            allocation = _frame_from_weights(products, weights)
    max_risk = min(float(profile.risk_tolerance_score) + RISK_SLACK, 5.0)
    allocation = _derisk(allocation, usable, max_risk, int(profile.horizon_months))
    return allocation

def _constraint_report(profile: pd.Series, allocation: pd.DataFrame, *, weight_col: str = "weight_pct", for_sip: bool = False) -> list[str]:
    months = int(profile.horizon_months)
    max_high, min_low = _risk_limits(months)
    max_risk = min(float(profile.risk_tolerance_score) + RISK_SLACK, 5.0)
    issues: list[str] = []
    if allocation.empty:
        return ["No allocation produced"]
    weights = allocation[weight_col]
    if weights.max() > 40.0001:
        issues.append("A product exceeds the 40% concentration cap")
    if allocation.loc[weights > 0, "asset_class"].nunique() < 2:
        issues.append("Portfolio uses fewer than two product categories")
    settlement_months = allocation.liquidity_days / 30.0
    if ((allocation.lock_in_months + settlement_months) > months + 1e-9).any():
        issues.append("A selected product's lock-in plus settlement extends beyond the goal date")
    if for_sip and (allocation.loc[weights > 0, "lock_in_months"] > 0).any():
        issues.append("Recurring SIP cannot be routed to a locked product")
    investment_basis = float(profile.monthly_sip_inr) if for_sip else float(profile.existing_investment_inr)
    if investment_basis > 0:
        below_minimum = allocation.loc[
            (weights > 0)
            & (investment_basis * weights / 100.0 + 0.01 < allocation.minimum_investment_inr)
        ]
        if not below_minimum.empty:
            issues.append(
                "Investment ticket is below product minimum for "
                + ", ".join(below_minimum.product_id.astype(str))
            )
    liquidity_need = float(profile.liquidity_need_inr)
    available_liquid = 0.0 if for_sip else float(profile.existing_investment_inr) * _liquid_share(allocation)
    has_contributions = float(profile.existing_investment_inr) > 0 or float(profile.monthly_sip_inr) > 0
    if has_contributions and not for_sip and available_liquid + 1e-6 < liquidity_need:
        issues.append(
            f"Near-term liquidity shortfall: ₹{max(0.0, liquidity_need-available_liquid):,.0f}"
        )
    audit_allocation = allocation.copy()
    audit_allocation["weight_pct"] = weights
    ok, high, low = _horizon_ok(audit_allocation, months)
    if high > max_high + 1e-6:
        issues.append(f"Risk 4–5 share {high:.0%} exceeds the {max_high:.0%} horizon cap")
    if low < min_low - 1e-6:
        issues.append(f"Risk 1–2 share {low:.0%} is below the {min_low:.0%} horizon floor")
    risk = _weighted_risk(audit_allocation)
    if risk > max_risk + 1e-6:
        issues.append(
            f"Weighted risk {risk:.2f} exceeds the investor limit {max_risk:.2f}"
        )
    return issues

def _goal_value(profile: pd.Series, allocation: pd.DataFrame) -> float:
    months = int(profile.horizon_months)
    balance = float(profile.existing_investment_inr)
    sip = float(profile.monthly_sip_inr)
    balance_rate, sip_rate = _mix_rates(allocation)
    return _separate_mix_future_value(balance, sip, balance_rate, sip_rate, months)

def _goal_scenarios(profile: pd.Series, allocation: pd.DataFrame) -> tuple[float, float, float]:
    balance_rate, sip_rate = _mix_rates(allocation)
    months = int(profile.horizon_months)
    balance = float(profile.existing_investment_inr)
    sip = float(profile.monthly_sip_inr)
    downside = _separate_mix_future_value(
        balance, sip, max(RETURN_FLOOR, balance_rate - 0.03), max(RETURN_FLOOR, sip_rate - 0.03), months
    )
    base = _separate_mix_future_value(balance, sip, balance_rate, sip_rate, months)
    upside = _separate_mix_future_value(
        balance, sip, min(RETURN_CAP, balance_rate + 0.03), min(RETURN_CAP, sip_rate + 0.03), months
    )
    return downside, base, upside

def _trim_unneeded_high_risk(profile: pd.Series, allocation: pd.DataFrame) -> pd.DataFrame:
    """Trim high-risk weight only while downside or a 5% base margin protects the goal."""
    result = allocation.copy()
    goal = float(profile.goal_amount_inr)
    initial_downside, base_value, _ = _goal_scenarios(profile, result)
    if base_value + 1e-6 < goal or (initial_downside + 1e-6 < goal and base_value < goal * 1.05):
        return result
    for _ in range(60):
        sources = result.loc[result.risk_score >= 4].sort_values("risk_score", ascending=False)
        changed = False
        for _, source in sources.iterrows():
            if _preference(profile) == "tax_saving" and source.product_id == "PROD07":
                continue
            sinks = result.loc[result.risk_score < source.risk_score].sort_values(
                ["risk_score", "assumed_annual_return"], ascending=[True, False]
            )
            for _, sink in sinks.iterrows():
                candidate = result.copy()
                source_idx = candidate.index[candidate.product_id == source.product_id][0]
                sink_idx = candidate.index[candidate.product_id == sink.product_id][0]
                moved = False
                for col in ("balance_weight_pct", "sip_weight_pct"):
                    available = float(candidate.loc[source_idx, col])
                    room = 40.0 - float(candidate.loc[sink_idx, col])
                    move = min(5.0, available, room)
                    if move > 0.05 and not (col == "sip_weight_pct" and float(sink.lock_in_months) > 0):
                        candidate.loc[source_idx, col] -= move
                        candidate.loc[sink_idx, col] += move
                        moved = True
                if not moved:
                    continue
                candidate["weight_pct"] = candidate.balance_weight_pct
                downside, base, _ = _goal_scenarios(profile, candidate)
                if downside + 1e-6 >= goal or base + 1e-6 >= goal * 1.05:
                    result = candidate
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return result

def _apply_minimums(
    profile: pd.Series,
    allocation: pd.DataFrame,
    products: pd.DataFrame,
    *,
    for_sip: bool = False,
) -> pd.DataFrame:
    if for_sip:
        return _apply_sip_minimums(profile, allocation, products)
    # The catalogue only provides a general product minimum. For SIP routing,
    # apply it to existing balance tickets here.
    investment_basis = float(profile.monthly_sip_inr) if for_sip else float(profile.existing_investment_inr)
    if investment_basis <= 0 or allocation.empty:
        return allocation
    weights = {row.product_id: row.weight_pct / 100.0 for _, row in allocation.iterrows()}
    excluded: set[str] = set()
    eligible = _eligible_products(products, profile, apply_preference=True, for_sip=for_sip)
    for _ in range(max(1, len(products) * 3)):
        alloc = _frame_from_weights(products, weights)
        changed = False
        for _, row in alloc.iterrows():
            amount = investment_basis * row.weight_pct / 100.0
            minimum = float(row.minimum_investment_inr)
            if amount + 1e-6 >= minimum or row.weight_pct <= 0:
                continue
            # Allocation output is rounded to tenths of a percentage point, so
            # round the required weight upward to a representable ticket.
            need_w = math.ceil((minimum / investment_basis) * 1000.0 - 1e-10) / 1000.0
            if need_w > PRODUCT_CAP + 1e-9:
                # Cannot hold this product at all.
                leftover = weights.pop(row.product_id, 0.0)
                excluded.add(row.product_id)
                low = eligible[(eligible.risk_score <= 2) & (~eligible.product_id.isin(excluded))]
                low = low.loc[low.minimum_investment_inr <= investment_basis * PRODUCT_CAP + 1e-6]
                _distribute(low, leftover, _preference(profile), "low", weights)
                changed = True
                break
            donors = alloc.loc[alloc.product_id != row.product_id].sort_values("weight_pct", ascending=False)
            donor = None
            take = 0.0
            for _, candidate in donors.iterrows():
                candidate_min = float(candidate.minimum_investment_inr) / investment_basis
                available = float(candidate.weight_pct) / 100.0 - max(0.01, candidate_min)
                if available > 1e-9:
                    donor = candidate
                    take = min(need_w - row.weight_pct / 100.0, available)
                    break
            if take <= 1e-9 or donor is None:
                leftover = weights.pop(row.product_id, 0.0)
                excluded.add(row.product_id)
                low = eligible[(eligible.risk_score <= 2) & (~eligible.product_id.isin(excluded))]
                low = low.loc[low.minimum_investment_inr <= investment_basis * PRODUCT_CAP + 1e-6]
                _distribute(low, leftover, _preference(profile), "low", weights)
                changed = True
                break
            weights[donor.product_id] = donor.weight_pct / 100.0 - take
            weights[row.product_id] = row.weight_pct / 100.0 + take
            changed = True
            break
        if not changed:
            break
    return _frame_from_weights(products, weights)


def _apply_sip_minimums(
    profile: pd.Series,
    allocation: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Plan SIP tickets in rupees, using the smallest feasible product set."""
    budget = round(float(profile.monthly_sip_inr), 2)
    if budget <= 0 or allocation.empty:
        return allocation
    active = allocation.loc[allocation.weight_pct > 0]
    if (
        budget * active.weight_pct / 100.0 + 0.01
        >= active.minimum_investment_inr
    ).all():
        return allocation
    eligible = _eligible_products(products, profile, apply_preference=True, for_sip=True)
    if eligible.empty:
        return allocation
    eligible = eligible.loc[eligible.minimum_investment_inr <= budget * PRODUCT_CAP + 1e-6]
    minimum_count = max(2, math.ceil(1.0 / PRODUCT_CAP - 1e-12))
    original = allocation.set_index("product_id").weight_pct.to_dict()
    feasible: list[tuple[tuple[float, ...], pd.DataFrame]] = []
    for count in range(minimum_count, len(eligible) + 1):
        for rows in combinations((row for _, row in eligible.iterrows()), count):
            if len({str(row.asset_class) for row in rows}) < 2:
                continue
            cents = {str(row.product_id): math.ceil(float(row.minimum_investment_inr) * 100 - 1e-9) for row in rows}
            budget_cents = round(budget * 100)
            if sum(cents.values()) > budget_cents:
                continue
            capacities = {str(row.product_id): math.floor(budget_cents * PRODUCT_CAP + 1e-8) for row in rows}
            if any(cents[pid] > capacities[pid] for pid in cents):
                continue
            remaining = budget_cents - sum(cents.values())
            while remaining > 0:
                # Fill the selected tickets toward their original mix, then by
                # product quality when a selected product had no prior weight.
                candidates = sorted(
                    (str(row.product_id) for row in rows if capacities[str(row.product_id)] > cents[str(row.product_id)]),
                    key=lambda pid: (
                        original.get(pid, 0.0) - cents[pid] / budget_cents,
                        float(eligible.loc[eligible.product_id == pid, "assumed_annual_return"].iloc[0])
                        - float(eligible.loc[eligible.product_id == pid, "annual_expense_ratio"].iloc[0]),
                    ),
                    reverse=True,
                )
                if not candidates:
                    break
                pid = candidates[0]
                add = min(remaining, capacities[pid] - cents[pid])
                cents[pid] += add
                remaining -= add
            if remaining:
                continue
            weights = {pid: amount / budget_cents for pid, amount in cents.items() if amount > 0}
            candidate = _frame_from_weights(products, weights, decimals=6)
            if _constraint_report(profile, candidate, for_sip=True):
                continue
            deviation = sum(abs(float(candidate.set_index("product_id").weight_pct.get(pid, 0.0)) - original.get(pid, 0.0)) for pid in set(original) | set(weights))
            weighted_risk = _weighted_risk(candidate)
            fee = float((candidate.weight_pct / 100.0 * candidate.annual_expense_ratio).sum())
            feasible.append(((float(count), deviation, weighted_risk, fee), candidate))
        if feasible:
            break
    return min(feasible, key=lambda item: item[0])[1] if feasible else allocation

def build_allocation(profile: pd.Series, products: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    """Return (allocation, status, residual hard-constraint issues)."""
    profile_errors = validate_profile(profile)
    if profile_errors:
        raise DataValidationError("; ".join(profile_errors))
    def make_mix(for_sip: bool) -> pd.DataFrame:
        mix_profile = profile.copy()
        if for_sip:
            mix_profile.existing_investment_inr = 0.0
        usable = _eligible_products(products, mix_profile, True, for_sip=for_sip)
        if usable.empty:
            usable = _eligible_products(products, mix_profile, False, for_sip=for_sip)
        result = _build_primary(mix_profile, products, for_sip=for_sip)
        ticket_adjusted = _apply_minimums(mix_profile, result, products, for_sip=for_sip)
        ticket_issues = _constraint_report(
            mix_profile, ticket_adjusted, weight_col="weight_pct", for_sip=for_sip
        )
        if not ticket_issues:
            result = ticket_adjusted
        return result

    try:
        balance_mix = make_mix(False)[["product_id", "weight_pct"]].rename(columns={"weight_pct": "balance_weight_pct"})
        sip_mix = make_mix(True)[["product_id", "weight_pct"]].rename(columns={"weight_pct": "sip_weight_pct"}) if float(profile.monthly_sip_inr) > 0 else pd.DataFrame(columns=["product_id", "sip_weight_pct"])
        allocation = products.merge(balance_mix, on="product_id", how="left").merge(sip_mix, on="product_id", how="left")
        allocation["balance_weight_pct"] = allocation.balance_weight_pct.fillna(0.0)
        allocation["sip_weight_pct"] = allocation.sip_weight_pct.fillna(0.0)
        allocation = allocation.loc[(allocation.balance_weight_pct > 0) | (allocation.sip_weight_pct > 0)].copy()
        allocation["weight_pct"] = allocation.balance_weight_pct
        allocation = allocation.sort_values(["risk_score", "product_id"]).reset_index(drop=True)
        allocation = _trim_unneeded_high_risk(profile, allocation)
        allocation = allocation.loc[
            (allocation.balance_weight_pct > 1e-9) | (allocation.sip_weight_pct > 1e-9)
        ].copy().reset_index(drop=True)
        allocation["weight_pct"] = allocation.balance_weight_pct
        issues = _constraint_report(profile, allocation)
        if float(profile.monthly_sip_inr) > 0:
            issues.extend(_constraint_report(profile, allocation, weight_col="sip_weight_pct", for_sip=True))
        issues = list(dict.fromkeys(issues))
        if not issues:
            return allocation, "PASS", []
    except SuitabilityError:
        allocation = pd.DataFrame()
        issues = ["Primary construction failed"]
    fallback = _frame_from_weights(products, _safest_weights(profile, products))
    fallback = _apply_minimums(profile, fallback, products)
    fallback_balance = fallback[["product_id", "weight_pct"]].rename(columns={"weight_pct": "balance_weight_pct"})
    if float(profile.monthly_sip_inr) > 0:
        sip_profile = profile.copy()
        sip_profile.existing_investment_inr = 0.0
        sip_fallback = _frame_from_weights(products, _safest_weights(sip_profile, products, for_sip=True))
        fallback_sip = sip_fallback[["product_id", "weight_pct"]].rename(columns={"weight_pct": "sip_weight_pct"})
    else:
        fallback_sip = pd.DataFrame(columns=["product_id", "sip_weight_pct"])
    fallback = products.merge(fallback_balance, on="product_id", how="left").merge(fallback_sip, on="product_id", how="left")
    fallback["balance_weight_pct"] = fallback.balance_weight_pct.fillna(0.0)
    fallback["sip_weight_pct"] = fallback.sip_weight_pct.fillna(0.0)
    fallback = fallback.loc[(fallback.balance_weight_pct > 0) | (fallback.sip_weight_pct > 0)].copy()
    fallback["weight_pct"] = fallback.balance_weight_pct
    fallback_issues = _constraint_report(profile, fallback)
    if float(profile.monthly_sip_inr) > 0:
        fallback_issues += _constraint_report(profile, fallback, weight_col="sip_weight_pct", for_sip=True)
    fallback_issues = list(dict.fromkeys(fallback_issues))
    issues = list(dict.fromkeys(issues))
    details = [f"Primary: {item}" for item in issues]
    details.extend(f"Fallback residual: {item}" for item in fallback_issues)
    return fallback, "FALLBACK", details

__all__ = ['_risk_limits', 'is_near_term_liquid', '_preference', '_eligible_products', '_sort_products', '_round_weights', '_distribute', '_frame_from_weights', '_weighted_risk', '_liquid_share', '_horizon_ok', '_derisk', '_safest_weights', '_target_sleeves', '_build_primary', '_constraint_report', '_goal_value', '_goal_scenarios', '_trim_unneeded_high_risk', '_apply_minimums', 'build_allocation']

