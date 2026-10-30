"""Pytest checks for allocation and suitability guardrails."""
import pandas as pd
import pytest
import engine
from engine import (
    DataValidationError,
    _coerce_numeric,
    _bootstrap_projection_percentiles,
    _validate_required,
    build_allocation,
    check_consistency,
    future_value,
    is_near_term_liquid,
    load_data,
    plan_buffer_first,
    plan_profile,
    required_monthly_sip,
    scenario_narrative,
    shorter_horizon_months,
    validate_profile,
    _goal_scenarios,
    _trim_unneeded_high_risk,
    validate_allocations,
)


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_allocations_sum_to_100_and_nonnegative(data):
    profiles, products, _ = data
    for _, profile in profiles.iterrows():
        allocation, _, _ = build_allocation(profile, products)
        validate_allocations(allocation)
        assert (allocation.weight_pct >= 0).all()
        assert abs(allocation.weight_pct.sum() - 100) <= 0.1


def test_product_concentration_and_category_diversity(data):
    profiles, products, _ = data
    for _, profile in profiles.iterrows():
        allocation, _, _ = build_allocation(profile, products)
        assert allocation.weight_pct.max() <= 40.0001
        assert allocation.asset_class.nunique() >= 2


def test_lockin_is_not_beyond_goal_horizon(data):
    profiles, products, _ = data
    for _, profile in profiles.iterrows():
        allocation, _, _ = build_allocation(profile, products)
    assert ((allocation.lock_in_months + allocation.liquidity_days / 30.0) <= profile.horizon_months + 1e-9).all()
    if profile.monthly_sip_inr > 0:
        assert (allocation.loc[allocation.sip_weight_pct > 0, "lock_in_months"] == 0).all()


def test_locked_products_not_routed_to_recurring_sips(data):
    profiles, products, _ = data
    profile = profiles.loc[profiles.investor_id == "INV01"].iloc[0]
    allocation, _, _ = build_allocation(profile, products)
    assert (allocation.loc[allocation.sip_weight_pct > 0, "lock_in_months"] == 0).all()
    assert allocation.loc[allocation.product_id == "PROD02", "balance_weight_pct"].sum() == 0


def test_horizon_risk_caps_and_risk_tolerance_on_pass_cases(data):
    profiles, products, _ = data
    for _, profile in profiles.iterrows():
        allocation, status, issues = build_allocation(profile, products)
        if status != "PASS":
            assert issues, "Fallback rows must record the unmet constraint"
            continue
        high = allocation.loc[allocation.risk_score >= 4, "weight_pct"].sum() / 100
        low = allocation.loc[allocation.risk_score <= 2, "weight_pct"].sum() / 100
        cap, floor = (
            (0.10, 0.70)
            if profile.horizon_months <= 36
            else ((0.30, 0.45) if profile.horizon_months <= 84 else (0.55, 0.25))
        )
        assert high <= cap + 1e-6
        assert low >= floor - 1e-6
        assert (allocation.weight_pct / 100 * allocation.risk_score).sum() <= profile.risk_tolerance_score + 0.25 + 1e-6


def test_malformed_profile_fails_clearly():
    malformed = pd.DataFrame([{"investor_id": "X"}])
    with pytest.raises(DataValidationError, match="missing required columns"):
        _validate_required(malformed, {"investor_id", "goal_amount_inr"}, "profiles")


def test_malformed_numeric_field_fails_clearly():
    profiles = pd.DataFrame([{"goal_amount_inr": "not-a-number"}])
    with pytest.raises(DataValidationError, match="empty or non-finite"):
        _coerce_numeric(profiles, ["goal_amount_inr"], "profiles")


def test_malformed_or_unknown_persona_is_row_local(data):
    profiles, products, returns = data
    malformed = profiles.iloc[0].copy()
    malformed["goal_amount_inr"] = float("nan")
    with pytest.raises(DataValidationError, match="goal_amount_inr must be a finite number"):
        plan_profile(malformed, products, returns)
    typo = profiles.iloc[0].copy()
    typo["preference"] = "hgih_growth"
    with pytest.raises(DataValidationError, match="unsupported preference"):
        plan_profile(typo, products, returns)


def test_batch_keeps_valid_personas_when_one_profile_row_is_invalid(monkeypatch, data):
    profiles, products, returns = data
    malformed = profiles.iloc[0:1].copy()
    malformed.loc[malformed.index[0], "experience"] = "wizard"
    monkeypatch.setattr(engine, "load_data", lambda *_args, **_kwargs: (malformed, products, returns))
    allocations, summary = engine.run_all()
    assert allocations.empty
    assert summary.iloc[0].planning_status == "INVALID_PROFILE"
    assert "unsupported experience" in summary.iloc[0].planning_exception


def test_batch_reports_unparseable_numeric_profile_fields_without_crashing(monkeypatch, data):
    profiles, products, returns = data
    bad_goal = profiles.iloc[0].copy()
    bad_goal["goal_amount_inr"] = "not-a-number"
    fractional_horizon = profiles.iloc[1].copy()
    fractional_horizon["horizon_months"] = "12.7"
    malformed = pd.DataFrame([bad_goal, fractional_horizon])
    monkeypatch.setattr(engine, "load_data", lambda *_args, **_kwargs: (malformed, products, returns))
    allocations, summaries = engine.run_all()
    assert allocations.empty
    assert summaries.planning_status.eq("INVALID_PROFILE").all()
    assert pd.isna(summaries.iloc[0].goal_amount_inr)
    assert summaries.iloc[1].horizon_months == 12.7
    assert "goal_amount_inr must be a finite number" in summaries.iloc[0].planning_exception
    assert "whole number" in summaries.iloc[1].planning_exception


def test_liquidity_shortfall_cannot_be_reported_as_pass(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV01"].iloc[0].copy()
    profile["liquidity_need_inr"] = 900_000
    _, summary = plan_profile(profile, products, returns)
    assert not summary["liquidity_covered"]
    assert summary["planning_status"] == "FALLBACK"
    assert "liquidity shortfall" in summary["planning_exception"].lower()


def test_projection_is_finite_and_zero_rate_exact():
    assert future_value(1000, 50, 0, 12) == 1600
    assert future_value(1000, 50, 0.06, 12) > 1600
    assert required_monthly_sip(1000, 1600, 0, 12) == 50


def test_equity_and_lockins_are_not_near_term_liquid(data):
    _, products, _ = data
    equity = products.loc[products.asset_class == "equity"].iloc[0]
    locked = products.loc[products.lock_in_months > 0].iloc[0]
    cash = products.loc[products.product_id == "PROD01"].iloc[0]
    assert not is_near_term_liquid(equity)
    assert not is_near_term_liquid(locked)
    assert is_near_term_liquid(cash)
    medium_credit = products.loc[products.product_id == "PROD04"].iloc[0]
    short_debt = products.loc[products.product_id == "PROD03"].iloc[0]
    assert not is_near_term_liquid(medium_credit)
    assert is_near_term_liquid(short_debt)


def test_infeasible_risk_profile_returns_safer_fallback(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV03"].iloc[0]
    allocation, status, issues = build_allocation(profile, products)
    assert status == "FALLBACK"
    assert issues
    validate_allocations(allocation)
    _, summary = plan_profile(profile, products, returns)
    assert summary["planning_status"] == "FALLBACK"
    assert summary["liquid_allocation_inr"] + 1e-6 >= profile.liquidity_need_inr


def test_avoid_equity_excludes_equity_products(data):
    profiles, products, _ = data
    profile = profiles.loc[profiles.investor_id == "INV06"].iloc[0]
    allocation, status, _ = build_allocation(profile, products)
    assert status == "PASS"
    assert (allocation.asset_class != "equity").all()


def test_no_lock_in_preference_excludes_lockins(data):
    profiles, products, _ = data
    profile = profiles.loc[profiles.investor_id == "INV11"].iloc[0]
    allocation, _, _ = build_allocation(profile, products)
    assert (allocation.lock_in_months == 0).all()


def test_preferences_rank_products_inside_guardrails(data):
    profiles, products, _ = data
    growth = profiles.loc[profiles.investor_id == "INV02"].iloc[0]
    tax = profiles.loc[profiles.investor_id == "INV07"].iloc[0].copy()
    tax["monthly_sip_inr"] = 0  # One-time funds may use the tax-saver lock-in.
    growth_allocation, growth_status, _ = build_allocation(growth, products)
    tax_allocation, tax_status, _ = build_allocation(tax, products)
    assert growth_status == "PASS"
    assert "PROD08" in set(growth_allocation.product_id)
    assert tax_status == "PASS"
    assert "PROD07" in set(tax_allocation.product_id)


def test_tax_saver_can_hold_balance_but_never_receive_sip(data):
    profiles, products, _ = data
    profile = profiles.loc[profiles.investor_id == "INV07"].iloc[0]
    allocation, status, issues = build_allocation(profile, products)
    assert status == "PASS", issues
    prod07 = allocation.loc[allocation.product_id == "PROD07"].iloc[0]
    assert prod07.balance_weight_pct > 0
    assert prod07.sip_weight_pct == 0
    assert allocation.sip_weight_pct.sum() == pytest.approx(100)


def test_sip_and_balance_mix_have_independent_guardrails(data):
    profiles, products, _ = data
    for _, profile in profiles.iterrows():
        allocation, _, _ = build_allocation(profile, products)
        assert allocation.balance_weight_pct.sum() == pytest.approx(100, abs=0.1)
        assert allocation.balance_weight_pct.max() <= 40.0001
        if profile.monthly_sip_inr > 0:
            assert allocation.sip_weight_pct.sum() == pytest.approx(100, abs=0.1)
            assert allocation.sip_weight_pct.max() <= 40.0001
            assert allocation.loc[allocation.balance_weight_pct > 0, "asset_class"].nunique() >= 2
            assert allocation.loc[allocation.sip_weight_pct > 0, "asset_class"].nunique() >= 2
            assert allocation.loc[allocation.sip_weight_pct > 0, "lock_in_months"].eq(0).all()


def test_long_goal_uses_only_the_risk_needed_to_reach_target(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV07"].iloc[0]
    allocation, summary = plan_profile(profile, products, returns)
    high_balance = allocation.loc[allocation.risk_score >= 4, "balance_weight_pct"].sum()
    high_sip = allocation.loc[allocation.risk_score >= 4, "sip_weight_pct"].sum()
    assert summary["goal_status"] in {"ON_TRACK", "ON_TRACK_BASE_ONLY", "WITHIN_1_PERCENT"}
    assert high_balance < 55
    assert high_sip < 55
    assert allocation.loc[allocation.product_id == "PROD07", "balance_weight_pct"].iloc[0] > 0
    assert allocation.loc[allocation.product_id == "PROD07", "sip_weight_pct"].iloc[0] == 0
    assert allocation.loc[allocation.product_id == "PROD08", ["balance_weight_pct", "sip_weight_pct"]].to_numpy().sum() == 0
    assert ((allocation.balance_weight_pct > 0) | (allocation.sip_weight_pct > 0)).all()


def test_split_mix_projection_uses_two_exact_future_value_calculations(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV07"].iloc[0]
    allocation, summary = plan_profile(profile, products, returns)
    balance_rate = (allocation.balance_weight_pct / 100 * allocation.assumed_annual_return).sum()
    sip_rate = (allocation.sip_weight_pct / 100 * allocation.assumed_annual_return).sum()
    exact = (
        future_value(profile.existing_investment_inr, 0, balance_rate, int(profile.horizon_months))
        + future_value(0, profile.monthly_sip_inr, sip_rate, int(profile.horizon_months))
    )
    assert summary["projection_base_inr"] == pytest.approx(exact)
    assert ((allocation.balance_weight_pct > 0) | (allocation.sip_weight_pct > 0)).all()
    assert allocation.why_included.notna().all()
    alternative_text = " ".join(allocation.excluded_alternative.astype(str))
    assert "PROD04 vs PROD03" in alternative_text
    assert "7.4% vs 6.8%" in alternative_text
    alternative_ids = {text.split(":", 1)[0].split(" vs ")[1] for text in allocation.excluded_alternative}
    assert len(alternative_ids) > 1
    assert summary["bootstrap_projection_p05_inr"] <= summary["bootstrap_projection_p50_inr"]
    assert summary["bootstrap_projection_p50_inr"] <= summary["bootstrap_projection_p95_inr"]
    assert summary["bootstrap_replicates"] == 1000


def test_paired_block_bootstrap_is_reproducible(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV07"].iloc[0]
    allocation, _, _ = build_allocation(profile, products)
    first = _bootstrap_projection_percentiles(
        profile, allocation, returns, profile.existing_investment_inr,
        profile.monthly_sip_inr, int(profile.horizon_months), replicates=200,
    )
    second = _bootstrap_projection_percentiles(
        profile, allocation, returns, profile.existing_investment_inr,
        profile.monthly_sip_inr, int(profile.horizon_months), replicates=200,
    )
    assert first == second
    assert first[0] <= first[1] <= first[2]


def test_market_shock_is_asset_class_specific(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV07"].iloc[0]
    base_allocation, base = plan_profile(profile, products, returns)
    shocked_allocation, shocked = plan_profile(profile, products, returns, market_shock=-0.15)
    expected = sum(
        profile.existing_investment_inr * row.weight_pct / 100 *
        (1 - 0.15 * {"cash": 0, "debt": 0.10, "hybrid": 0.50, "equity": 1.0}[row.asset_class])
        for _, row in base_allocation.iterrows()
    )
    assert shocked["shocked_principal_inr"] == pytest.approx(expected)
    assert shocked["weighted_annual_return_estimate"] == pytest.approx(base["weighted_annual_return_estimate"])
    assert shocked["projection_base_inr"] < base["projection_base_inr"]
    assert (shocked_allocation.market_shock_factor.nunique() > 1)
    for asset, group in shocked_allocation.groupby("asset_class"):
        expected_factor = -0.15 * {"cash": 0, "debt": 0.10, "hybrid": 0.50, "equity": 1.0}[asset]
        assert all(value == pytest.approx(expected_factor) for value in group.market_shock_factor)


def test_upside_infeasible_goal_is_flagged(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV08"].iloc[0]
    _, summary = plan_profile(profile, products, returns)
    assert summary["goal_infeasible"]
    assert summary["goal_status"] == "UPSIDE_SHORTFALL"
    assert "resetting the goal" in summary["actionable_adjustment"]


def test_minimum_ticket_respected_when_balance_positive(data):
    profiles, products, _ = data
    for _, profile in profiles.iterrows():
        if profile.existing_investment_inr <= 0:
            continue
        allocation, _, _ = build_allocation(profile, products)
        amounts = profile.existing_investment_inr * allocation.weight_pct / 100
        held = allocation.weight_pct > 0
        assert (amounts[held] + 0.01 >= allocation.loc[held, "minimum_investment_inr"]).all()


def test_zero_balance_does_not_apply_lump_sum_minimums_to_sips(data):
    profiles, products, _ = data
    profile = profiles.loc[profiles.investor_id == "INV11"].iloc[0]
    allocation, status, issues = build_allocation(profile, products)
    assert profile.existing_investment_inr == 0
    assert allocation.sip_weight_pct.sum() == pytest.approx(100)
    assert (allocation.loc[allocation.sip_weight_pct > 0, "lock_in_months"] == 0).all()
    assert status == "PASS", issues


def test_sip_only_emergency_warning_does_not_claim_funded_buffer(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV11"].iloc[0]
    _, summary = plan_profile(profile, products, returns)
    assert not summary["emergency_buffer_applied"]
    assert summary["emergency_buffer_sip_routing"]
    assert "no buffer is available today" in summary["warnings"]


def test_margin_trim_preserves_downside_or_five_percent_base_margin(data):
    profiles, products, _ = data
    profile = profiles.loc[profiles.investor_id == "INV01"].iloc[0].copy()
    allocation, _, _ = build_allocation(profile, products)
    _, base, _ = _goal_scenarios(profile, allocation)
    profile["goal_amount_inr"] = base * 0.99
    downside, base, _ = _goal_scenarios(profile, allocation)
    assert downside < profile.goal_amount_inr < base * 1.05
    trimmed = _trim_unneeded_high_risk(profile, allocation)
    pd.testing.assert_frame_equal(trimmed, allocation)


def test_materiality_status_for_inv01_and_inv02(data):
    profiles, products, returns = data
    for investor_id in ("INV01", "INV02", "INV05"):
        profile = profiles.loc[profiles.investor_id == investor_id].iloc[0]
        _, summary = plan_profile(profile, products, returns)
        assert summary["goal_status"] == "WITHIN_1_PERCENT"
        assert summary["goal_status_label"] == "Within 1% of target"
        assert abs(summary["base_shortfall_surplus_inr"]) <= profile.goal_amount_inr * 0.01
        assert "To add a 5% margin" in summary["goal_materiality_note"]
        assert summary["extra_sip_for_5pct_margin_inr"] >= 0


def test_infeasible_advice_uses_base_case_required_sip(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV08"].iloc[0]
    _, summary = plan_profile(profile, products, returns)
    assert "base-case estimate" in summary["actionable_adjustment"]
    assert f"₹{summary['required_sip_inr']:,.0f}/month" in summary["actionable_adjustment"]
    assert "optimistic upside estimate" in summary["actionable_adjustment"]


def test_thin_emergency_fund_excludes_locked_balance(data):
    profiles, products, returns = data
    for investor_id in ("INV05", "INV08"):
        profile = profiles.loc[profiles.investor_id == investor_id].iloc[0]
        allocation, summary = plan_profile(profile, products, returns)
        assert summary["locked_balance_weight_pct"] <= 10
        assert allocation.loc[allocation.lock_in_months > 0, "balance_weight_pct"].sum() <= 10
    profile = profiles.loc[profiles.investor_id == "INV05"].iloc[0]
    buffer_allocation, buffer_summary = plan_buffer_first(profile, products, returns)
    assert buffer_summary["locked_balance_weight_pct"] <= 10
    assert buffer_allocation.loc[buffer_allocation.near_term_liquid, "balance_weight_pct"].sum() >= 20


def test_avoid_equity_hybrid_exposure_is_disclosed(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV06"].iloc[0]
    allocation, summary = plan_profile(profile, products, returns)
    if (allocation.asset_class == "hybrid").any():
        assert "indirect equity exposure" in summary["warnings"]
        assert "roughly 50% equity" in summary["warnings"]


def test_consistency_prompts_cover_horizon_and_conflicting_preferences(data):
    profiles, _, _ = data
    profile = profiles.loc[profiles.investor_id == "INV06"].iloc[0].copy()
    profile["risk_tolerance_score"] = 5
    prompts = check_consistency(profile)["prompts"]
    assert any("high while avoid_equity" in item for item in prompts)
    profile["preference"] = "none"
    profile["horizon_months"] = 12.7
    assert any("whole-number" in item for item in check_consistency(profile)["prompts"])
    profile["horizon_months"] = 1
    assert not check_consistency(profile)["prompts"]
    assert any("Very short goal horizon" in item for item in check_consistency(profile)["warnings"])


def test_one_to_five_month_horizons_are_allowed_and_shorter_scenario_clamps(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV03"].iloc[0].copy()
    profile["horizon_months"] = 3
    assert validate_profile(profile) == []
    allocation, summary = plan_profile(profile, products, returns)
    assert summary["horizon_months"] == 3
    assert summary["planning_status"] in {"PASS", "FALLBACK"}
    assert "Very short goal horizon" in summary["warnings"]
    assert shorter_horizon_months(15) == 3
    assert shorter_horizon_months(3) == 1
    one_month = profile.copy()
    one_month["horizon_months"] = shorter_horizon_months(3)
    _, one_month_summary = plan_profile(one_month, products, returns)
    assert one_month_summary["horizon_months"] == 1


def test_consistency_flags_goal_already_met_and_zero_contributions(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV01"].iloc[0].copy()
    profile["existing_investment_inr"] = profile.goal_amount_inr
    consistency = check_consistency(profile)
    assert any("Goal already met; consider a higher target" in item for item in consistency["warnings"])
    profile["existing_investment_inr"] = 0
    profile["monthly_sip_inr"] = 0
    allocation, summary = plan_profile(profile, products, returns)
    assert summary["planning_status"] == "NEEDS_CLARIFICATION"
    assert summary["needs_clarification"] is True
    assert "Needs clarification" in summary["clarification_prompts"]


def test_sip_minimums_are_applied_as_exact_rupee_tickets_with_fewer_products(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV11"].iloc[0]
    allocation, summary = plan_profile(profile, products, returns)
    sip_rows = allocation.loc[allocation.monthly_sip_allocation_inr > 0]
    assert len(sip_rows) == 3
    assert sip_rows.monthly_sip_allocation_inr.tolist() == [1000.0, 1000.0, 1000.0]
    assert (sip_rows.monthly_sip_allocation_inr + 0.01 >= sip_rows.minimum_investment_inr).all()
    assert summary["sip_minimums_enforced"]


def test_no_contributions_are_not_misreported_as_liquidity_shortfall(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV11"].iloc[0].copy()
    profile["existing_investment_inr"] = 0
    profile["monthly_sip_inr"] = 0
    profile["liquidity_need_inr"] = 100_000
    _, summary = plan_profile(profile, products, returns)
    assert summary["liquidity_covered"] is None
    assert summary["planning_status"] == "NEEDS_CLARIFICATION"
    assert summary["suitability_status"] == "PASS"
    assert "no contributions" in summary["actionable_adjustment"].lower()
    assert "liquidity shortfall" not in summary["planning_exception"].lower()


def test_scenario_narrative_explains_inputs_constraints_and_projection_changes(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV07"].iloc[0]
    before_alloc, before = plan_profile(profile, products, returns)
    shorter = profile.copy()
    shorter["horizon_months"] = 132
    after_alloc, after = plan_profile(shorter, products, returns)
    narrative = scenario_narrative(before_alloc, after_alloc, before, after, "shorter horizon")
    assert "horizon changed" in narrative
    assert "Binding constraint:" in narrative
    assert "downside ₹" in narrative and "base ₹" in narrative and "upside ₹" in narrative


def test_tolerance_what_if_names_the_binding_constraint_when_mix_is_unchanged(data):
    profiles, products, returns = data
    profile = profiles.loc[profiles.investor_id == "INV04"].iloc[0].copy()
    before_allocation, before = plan_profile(profile, products, returns)
    profile["risk_tolerance_score"] = 5
    after_allocation, after = plan_profile(profile, products, returns)
    delta = after_allocation.set_index("product_id")[["balance_weight_pct", "sip_weight_pct"]].sub(
        before_allocation.set_index("product_id")[["balance_weight_pct", "sip_weight_pct"]], fill_value=0
    ).abs().to_numpy().sum()
    assert delta < 1e-7
    assert after["binding_constraint"] == "horizon_cap"
    narrative = scenario_narrative(before_allocation, after_allocation, before, after, "Change risk tolerance")
    assert "Risk tolerance is not binding here; horizon cap remains binding" in narrative

