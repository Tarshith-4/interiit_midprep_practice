"""Future-value, scenario and synthetic-return projection functions."""
from __future__ import annotations
import math
import zlib
import numpy as np
import pandas as pd
from validation import DataValidationError, RETURN_CAP, RETURN_FLOOR, BOOTSTRAP_REPLICATES, BOOTSTRAP_BLOCK_MONTHS

def future_value(principal: float, sip: float, annual_rate: float, months: int) -> float:
    if months < 0 or min(principal, sip) < 0 or not all(map(math.isfinite, [principal, sip, annual_rate])):
        raise ValueError("Invalid future-value input")
    annual_rate = min(RETURN_CAP, max(RETURN_FLOOR, annual_rate))
    monthly = annual_rate / 12
    if abs(monthly) < 1e-15:
        return principal + sip * months
    factor = (1 + monthly) ** months
    return principal * factor + sip * (factor - 1) / monthly

def _mix_rates(allocation: pd.DataFrame) -> tuple[float, float]:
    balance_rate = float((allocation.balance_weight_pct / 100.0 * allocation.assumed_annual_return).sum())
    sip_rate = (
        float((allocation.sip_weight_pct / 100.0 * allocation.assumed_annual_return).sum())
        if float(allocation.sip_weight_pct.sum()) > 0 else balance_rate
    )
    return balance_rate, sip_rate

def _separate_mix_future_value(principal: float, sip: float, balance_rate: float, sip_rate: float, months: int) -> float:
    """Project existing balance and SIP flows at their respective portfolio rates."""
    return future_value(principal, 0.0, balance_rate, months) + future_value(0.0, sip, sip_rate, months)

def _required_sip_separate(principal: float, goal: float, balance_rate: float, sip_rate: float, months: int) -> float:
    if months <= 0:
        return float("inf")
    balance_value = future_value(principal, 0.0, balance_rate, months)
    if balance_value >= goal:
        return 0.0
    unit_sip_value = future_value(0.0, 1.0, sip_rate, months)
    return max(0.0, (goal - balance_value) / unit_sip_value)

def _extra_months_separate(principal: float, sip: float, balance_rate: float, sip_rate: float, months: int, goal: float) -> int | None:
    for extra in range(0, 361):
        if _separate_mix_future_value(principal, sip, balance_rate, sip_rate, months + extra) >= goal:
            return extra
    return None

def required_monthly_sip(principal: float, goal: float, annual_rate: float, months: int) -> float:
    """Invert the brief's FV formula for the monthly contribution that hits `goal`."""
    if months <= 0:
        return float("inf")
    annual_rate = min(RETURN_CAP, max(RETURN_FLOOR, annual_rate))
    monthly = annual_rate / 12
    if abs(monthly) < 1e-15:
        return max(0.0, (goal - principal) / months)
    factor = (1 + monthly) ** months
    residual = goal - principal * factor
    if residual <= 0:
        return 0.0
    return residual * monthly / (factor - 1)

def extra_months_to_goal(principal: float, sip: float, annual_rate: float, months: int, goal: float) -> int | None:
    if future_value(principal, sip, annual_rate, months) >= goal:
        return 0
    for extra in range(1, 361):
        if future_value(principal, sip, annual_rate, months + extra) >= goal:
            return extra
    return None

def _realized_annual(returns: pd.DataFrame, product_id: str) -> float | None:
    series = returns.loc[returns.product_id == product_id, "monthly_return"]
    if series.empty:
        return None
    return float((1 + series).prod() ** (12 / len(series)) - 1)

def _bootstrap_projection_percentiles(
    profile: pd.Series,
    allocation: pd.DataFrame,
    returns: pd.DataFrame,
    principal: float,
    sip: float,
    months: int,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    block_months: int = BOOTSTRAP_BLOCK_MONTHS,
) -> tuple[float, float, float]:
    """Paired circular block bootstrap of synthetic monthly product returns."""
    if replicates < 100 or block_months < 1 or months < 1:
        raise ValueError("Bootstrap requires at least 100 replicates and positive horizon/block length")
    product_ids = allocation.product_id.astype(str).tolist()
    try:
        monthly = returns.pivot(index="month", columns="product_id", values="monthly_return").sort_index()
    except ValueError as exc:
        raise DataValidationError("Synthetic returns must have one row per month and product") from exc
    missing = sorted(set(product_ids) - set(monthly.columns.astype(str)))
    if missing:
        raise DataValidationError("Synthetic returns are missing products: " + ", ".join(missing))
    matrix = monthly.loc[:, product_ids].to_numpy(dtype=float)
    if matrix.size == 0 or not np.isfinite(matrix).all():
        raise DataValidationError("Synthetic return history must be a complete finite month-by-product matrix")

    history_months = matrix.shape[0]
    block_months = min(block_months, history_months)
    block_count = math.ceil(months / block_months)
    seed = 20261030 + zlib.crc32(str(profile.investor_id).encode("utf-8"))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, history_months, size=(replicates, block_count))
    offsets = np.arange(block_months)
    sampled_months = ((starts[:, :, None] + offsets[None, None, :]) % history_months)
    sampled_months = sampled_months.reshape(replicates, -1)[:, :months]
    sampled_returns = matrix[sampled_months]

    balance_weights = allocation.balance_weight_pct.to_numpy(dtype=float) / 100.0
    sip_weights = allocation.sip_weight_pct.to_numpy(dtype=float) / 100.0
    if not np.isclose(balance_weights.sum(), 1.0, atol=0.001):
        raise DataValidationError("Balance weights must sum to 100% for the bootstrap projection")
    if sip > 0 and not np.isclose(sip_weights.sum(), 1.0, atol=0.001):
        raise DataValidationError("SIP weights must sum to 100% for the bootstrap projection")
    if sip <= 0:
        sip_weights = balance_weights

    balance_monthly = np.einsum("rmp,p->rm", sampled_returns, balance_weights)
    sip_monthly = np.einsum("rmp,p->rm", sampled_returns, sip_weights)
    balance_values = np.full(replicates, float(principal), dtype=float)
    sip_values = np.zeros(replicates, dtype=float)
    for month in range(months):
        balance_values *= 1.0 + balance_monthly[:, month]
        sip_values *= 1.0 + sip_monthly[:, month]
        sip_values += float(sip)  # end-of-month contribution, matching the FV convention
    percentiles = np.percentile(balance_values + sip_values, [5, 50, 95])
    return tuple(float(value) for value in percentiles)

__all__ = ['future_value', '_mix_rates', '_separate_mix_future_value', '_required_sip_separate', '_extra_months_separate', 'required_monthly_sip', 'extra_months_to_goal', '_realized_annual', '_bootstrap_projection_percentiles']

