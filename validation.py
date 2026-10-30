"""Strict input validation and profile consistency checks."""
from __future__ import annotations
from pathlib import Path
import math
import pandas as pd

_WORKSPACE = Path(__file__).resolve().parent
_BUNDLED_DATA = _WORKSPACE / "data"
DATA_DIR = _BUNDLED_DATA if (_BUNDLED_DATA / "investor_profiles.csv").exists() else _WORKSPACE
OUTPUT_DIR = _WORKSPACE / "outputs"
SUPPORTED_PREFERENCES = {"", "capital_protection", "low_cost", "high_growth", "avoid_equity", "tax_saving", "no_lock_in"}
SUPPORTED_EXPERIENCE = {"beginner", "intermediate", "advanced"}
PROFILE_COLUMNS = {
    "investor_id", "goal_label", "goal_amount_inr", "horizon_months",
    "existing_investment_inr", "monthly_sip_inr", "risk_tolerance_score",
    "emergency_fund_months", "liquidity_need_inr", "experience", "preference",
}
PRODUCT_COLUMNS = {
    "product_id", "product_name", "asset_class", "risk_score",
    "assumed_annual_return", "volatility_proxy", "annual_expense_ratio",
    "liquidity_days", "minimum_investment_inr", "lock_in_months",
    "category_description",
}
RETURN_COLUMNS = {"month", "product_id", "monthly_return"}
PROFILE_NUMERIC = [
    "goal_amount_inr", "horizon_months", "existing_investment_inr",
    "monthly_sip_inr", "risk_tolerance_score", "emergency_fund_months",
    "liquidity_need_inr",
]
PRODUCT_NUMERIC = [
    "risk_score", "assumed_annual_return", "volatility_proxy",
    "annual_expense_ratio", "liquidity_days", "minimum_investment_inr",
    "lock_in_months",
]
NEAR_TERM_MAX_DAYS = 7
EMERGENCY_BUFFER_PCT = 0.20
PRODUCT_CAP = 0.40
RISK_SLACK = 0.25
RETURN_FLOOR = 0.0
RETURN_CAP = 0.18
MIN_HORIZON_MONTHS = 1
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_BLOCK_MONTHS = 3

class DataValidationError(ValueError):
    """Input data is missing, malformed, or inconsistent."""
class SuitabilityError(ValueError):
    """No portfolio can be constructed from eligible products."""

def _read_csv(path: Path, required: set[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(f"{label} not found: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DataValidationError(f"Could not read {label}: {exc}") from exc
    _validate_required(frame, required, label)
    return frame

def _validate_required(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(
            f"{label} missing required columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise DataValidationError(f"{label} is empty")

def _coerce_numeric(df: pd.DataFrame, cols: list[str], label: str) -> None:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any() or not df[col].map(math.isfinite).all():
            raise DataValidationError(f"{label}.{col} contains an empty or non-finite value")

def load_data(data_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and strictly validate the three source CSV files."""
    root = Path(data_dir)
    profiles = _read_csv(root / "investor_profiles.csv", PROFILE_COLUMNS, "investor_profiles.csv")
    products = _read_csv(root / "investment_products.csv", PRODUCT_COLUMNS, "investment_products.csv")
    returns = _read_csv(root / "synthetic_monthly_returns.csv", RETURN_COLUMNS, "synthetic_monthly_returns.csv")
    # Keep malformed profile rows so the rest of the personas remain usable.
    for col in PROFILE_NUMERIC:
        profiles[col] = pd.to_numeric(profiles[col], errors="coerce")
    for df, cols, label in [
        (products, PRODUCT_NUMERIC, "products"),
        (returns, ["monthly_return"], "returns"),
    ]:
        _coerce_numeric(df, cols, label)
    profiles["preference"] = profiles["preference"].fillna("").astype(str).str.strip()
    profiles["experience"] = profiles["experience"].fillna("").astype(str).str.strip()
    text_profile = (PROFILE_COLUMNS - set(PROFILE_NUMERIC)) - {"preference", "experience", "investor_id", "goal_label"}
    for df, cols, label in [
        (profiles, text_profile, "profiles"),
        (products, PRODUCT_COLUMNS - set(PRODUCT_NUMERIC), "products"),
        (returns, {"month", "product_id"}, "returns"),
    ]:
        for col in cols:
            if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
                raise DataValidationError(f"{label}.{col} contains an empty value")
    valid_ids = profiles.investor_id.dropna().astype(str).str.strip()
    valid_ids = valid_ids.loc[valid_ids.ne("")]
    if valid_ids.duplicated().any() or products.product_id.duplicated().any():
        raise DataValidationError("Non-empty investor IDs and product IDs must be unique")
    if not products.risk_score.between(1, 5).all() or (
        products[["liquidity_days", "lock_in_months", "minimum_investment_inr", "annual_expense_ratio"]] < 0
    ).any().any():
        raise DataValidationError("Product risk or non-negative fields are out of range")
    if not set(products.asset_class).issubset({"cash", "debt", "hybrid", "equity"}):
        raise DataValidationError("Products contain an unsupported asset_class")
    if not set(returns.product_id).issubset(set(products.product_id)):
        raise DataValidationError("Returns contain unknown product_id values")
    if not returns.monthly_return.map(lambda x: x > -1).all():
        raise DataValidationError("Monthly returns must be greater than -100%")
    return profiles, products, returns

def validate_profile(profile: pd.Series) -> list[str]:
    """Return row-local validation errors without invalidating other personas."""
    issues: list[str] = []
    for field in ("investor_id", "goal_label"):
        value = profile.get(field)
        if pd.isna(value) or not str(value).strip():
            issues.append(f"{field} is required")
    for field in PROFILE_NUMERIC:
        value = profile.get(field)
        try:
            valid = not pd.isna(value) and math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            issues.append(f"{field} must be a finite number")
    if issues:
        return issues
    if float(profile.goal_amount_inr) <= 0 or float(profile.horizon_months) <= 0:
        issues.append("goal_amount_inr and horizon_months must be positive")
    if not float(profile.horizon_months).is_integer():
        issues.append("horizon_months must be a whole number of months")
    elif float(profile.horizon_months) < MIN_HORIZON_MONTHS:
        issues.append(f"horizon_months must be at least {MIN_HORIZON_MONTHS} months")
    for field in ("existing_investment_inr", "monthly_sip_inr", "emergency_fund_months", "liquidity_need_inr"):
        if float(profile[field]) < 0:
            issues.append(f"{field} cannot be negative")
    if not 1 <= float(profile.risk_tolerance_score) <= 5:
        issues.append("risk_tolerance_score must be between 1 and 5")
    preference = str(profile.get("preference", "")).strip().lower()
    if preference not in SUPPORTED_PREFERENCES:
        issues.append(f"unsupported preference '{preference}'")
    experience = str(profile.get("experience", "")).strip().lower()
    if experience not in SUPPORTED_EXPERIENCE:
        issues.append(f"unsupported experience '{experience}'")
    return issues

def check_consistency(profile: pd.Series) -> dict[str, list[str]]:
    """Return human-readable profile checks and clarification prompts."""
    warnings: list[str] = []
    prompts: list[str] = []
    def numeric(field: str) -> float:
        try:
            value = profile.get(field, 0)
            return 0.0 if pd.isna(value) else float(value)
        except (TypeError, ValueError):
            return 0.0

    horizon = profile.get("horizon_months")
    if pd.notna(horizon):
        try:
            value = float(horizon)
            if not value.is_integer():
                prompts.append("Please confirm a whole-number goal horizon; fractional months are not supported.")
            elif value < 6:
                warnings.append("Very short goal horizon (1–5 months): only immediate-liquidity and short-settlement products may fit; projections are especially sensitive to timing.")
        except (TypeError, ValueError):
            prompts.append("Please enter the goal horizon in whole months.")
    if str(profile.get("preference", "")).strip().lower() == "avoid_equity" and numeric("risk_tolerance_score") >= 4:
        prompts.append("Risk tolerance is high while avoid_equity is selected. Should hybrid funds with roughly 50% equity exposure be allowed?")
    balance = numeric("existing_investment_inr")
    sip = numeric("monthly_sip_inr")
    goal = numeric("goal_amount_inr")
    if goal > 0 and balance >= goal:
        warnings.append("Goal already met; consider a higher target or confirm whether additional contributions are intended.")
    if balance == 0 and sip == 0:
        prompts.append("Needs clarification: starting balance and monthly SIP are both zero. Add a contribution or confirm that no investment plan is intended.")
    return {"warnings": warnings, "prompts": prompts}

def validate_allocations(allocations: pd.DataFrame, tolerance: float = 0.1) -> None:
    if allocations.empty or "weight_pct" not in allocations.columns:
        raise DataValidationError("Allocation is empty or lacks weight_pct")
    w = pd.to_numeric(allocations.weight_pct, errors="coerce")
    if w.isna().any() or not w.map(math.isfinite).all() or (w < -1e-9).any():
        raise DataValidationError("Allocation weights must be finite and non-negative")
    if abs(float(w.sum()) - 100) > tolerance:
        raise DataValidationError(f"Allocation weights sum to {w.sum():.4f}%, expected 100% ± {tolerance}%")

__all__ = ['_WORKSPACE', '_BUNDLED_DATA', 'DATA_DIR', 'OUTPUT_DIR', 'SUPPORTED_PREFERENCES', 'SUPPORTED_EXPERIENCE', 'PROFILE_COLUMNS', 'PRODUCT_COLUMNS', 'RETURN_COLUMNS', 'PROFILE_NUMERIC', 'PRODUCT_NUMERIC', 'NEAR_TERM_MAX_DAYS', 'EMERGENCY_BUFFER_PCT', 'PRODUCT_CAP', 'RISK_SLACK', 'RETURN_FLOOR', 'RETURN_CAP', 'MIN_HORIZON_MONTHS', 'BOOTSTRAP_REPLICATES', 'BOOTSTRAP_BLOCK_MONTHS', 'DataValidationError', 'SuitabilityError', '_read_csv', '_validate_required', '_coerce_numeric', 'load_data', 'validate_profile', 'check_consistency', 'validate_allocations']

