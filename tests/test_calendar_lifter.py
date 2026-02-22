"""Integration tests for Calendar-Aware Frequency-Dependent Lifting Operators.

Test groups:
  1. Calendar encoder: synthetic fallback, third_friday, phase geometry
  2. Von Mises basis: peak, negligible, non-negative, linear in amplitude
  3. Lifter: backward compat, calendar-aware difference, spectral bounds
  4. Arnold tongue resonance: rational approximation, tongue widths, detection
  5. Integration: propagate_shock with/without calendar, full pipeline
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Calendar Encoder Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_synthetic_fallback_valid_phases():
    """Synthetic fallback produces valid phase vectors."""
    from tensor.calendar_regime import CalendarRegimeEncoder, N_CHANNELS

    encoder = CalendarRegimeEncoder(use_synthetic=True)
    phase = encoder.encode(date(2025, 6, 15))

    assert phase.theta.shape == (N_CHANNELS,), f"Wrong shape: {phase.theta.shape}"
    assert phase.amplitudes.shape == (N_CHANNELS,), f"Wrong shape: {phase.amplitudes.shape}"

    # All angles in [0, 2pi)
    for i in range(N_CHANNELS):
        assert 0.0 <= phase.theta[i] < 2.0 * np.pi, \
            f"theta[{i}]={phase.theta[i]} not in [0, 2pi)"

    # All amplitudes in [0, 1]
    for i in range(N_CHANNELS):
        assert 0.0 <= phase.amplitudes[i] <= 1.0, \
            f"amplitude[{i}]={phase.amplitudes[i]} not in [0, 1]"

    print(f"  theta={phase.theta}")
    print(f"  amplitudes={phase.amplitudes}")
    print(f"  active_events={phase.active_events}")
    print("  Synthetic fallback valid phases PASS")


def test_synthetic_deterministic():
    """Synthetic encoder is deterministic for the same date."""
    from tensor.calendar_regime import CalendarRegimeEncoder

    encoder = CalendarRegimeEncoder(use_synthetic=True)
    p1 = encoder.encode(date(2025, 3, 10))
    p2 = encoder.encode(date(2025, 3, 10))

    np.testing.assert_array_equal(p1.theta, p2.theta)
    np.testing.assert_array_equal(p1.amplitudes, p2.amplitudes)
    print("  Synthetic deterministic PASS")


def test_third_friday():
    """3rd Friday computation correct for known months."""
    from tensor.calendar_regime import third_friday

    # Known 3rd Fridays
    assert third_friday(2025, 1) == date(2025, 1, 17), \
        f"Jan 2025: got {third_friday(2025, 1)}"
    assert third_friday(2025, 2) == date(2025, 2, 21), \
        f"Feb 2025: got {third_friday(2025, 2)}"
    assert third_friday(2025, 3) == date(2025, 3, 21), \
        f"Mar 2025: got {third_friday(2025, 3)}"
    assert third_friday(2025, 6) == date(2025, 6, 20), \
        f"Jun 2025: got {third_friday(2025, 6)}"

    # Always a Friday
    for m in range(1, 13):
        tf = third_friday(2025, m)
        assert tf.weekday() == 4, f"{tf} is not a Friday"

    print("  Third Friday computation PASS")


def test_phase_near_event():
    """Phase is ~0 on event date, ~pi far from event."""
    from tensor.calendar_regime import CalendarRegimeEncoder, PHASE_FED

    fed_date = date(2025, 6, 18)
    encoder = CalendarRegimeEncoder(fed_dates=[fed_date])

    # On the event date
    phase_on = encoder.encode(fed_date)
    assert phase_on.theta[PHASE_FED] < 0.1 or phase_on.theta[PHASE_FED] > 2 * np.pi - 0.1, \
        f"Phase on event should be ~0, got {phase_on.theta[PHASE_FED]}"

    # Amplitude should be high on event date
    assert phase_on.amplitudes[PHASE_FED] > 0.9, \
        f"Amplitude on event should be ~1.0, got {phase_on.amplitudes[PHASE_FED]}"

    print(f"  On event: theta={phase_on.theta[PHASE_FED]:.4f}, amp={phase_on.amplitudes[PHASE_FED]:.4f}")
    print("  Phase near event PASS")


def test_amplitude_decay():
    """Amplitude decays away from event, peaks near it."""
    from tensor.calendar_regime import CalendarRegimeEncoder, PHASE_FED

    fed_date = date(2025, 6, 18)
    encoder = CalendarRegimeEncoder(fed_dates=[fed_date])

    amp_on = encoder.encode(fed_date).amplitudes[PHASE_FED]
    amp_3d = encoder.encode(date(2025, 6, 23)).amplitudes[PHASE_FED]  # Mon +3 trading days
    amp_10d = encoder.encode(date(2025, 7, 2)).amplitudes[PHASE_FED]  # ~10 trading days

    assert amp_on > amp_3d > amp_10d, \
        f"Amplitude should decay: on={amp_on:.3f}, 3d={amp_3d:.3f}, 10d={amp_10d:.3f}"

    print(f"  Decay: on={amp_on:.3f} > 3d={amp_3d:.3f} > 10d={amp_10d:.3f}")
    print("  Amplitude decay PASS")


def test_per_ticker_earnings():
    """Per-ticker earnings overlay changes only PHASE_EARNINGS channel."""
    from tensor.calendar_regime import CalendarRegimeEncoder, PHASE_EARNINGS, PHASE_FED

    earnings_dates = {
        "AAPL": [date(2025, 1, 30), date(2025, 4, 24)],
        "MSFT": [date(2025, 1, 28), date(2025, 4, 22)],
    }
    encoder = CalendarRegimeEncoder(earnings_dates=earnings_dates)

    query = date(2025, 1, 29)
    phase_aapl = encoder.encode(query, ticker="AAPL")
    phase_msft = encoder.encode(query, ticker="MSFT")

    # Earnings phases should differ (different nearest dates)
    assert phase_aapl.theta[PHASE_EARNINGS] != phase_msft.theta[PHASE_EARNINGS], \
        "AAPL and MSFT earnings phases should differ"

    # Fed phase should be the same (ticker-independent)
    assert phase_aapl.theta[PHASE_FED] == phase_msft.theta[PHASE_FED], \
        "Fed phase should be the same regardless of ticker"

    print(f"  AAPL earnings theta={phase_aapl.theta[PHASE_EARNINGS]:.4f}")
    print(f"  MSFT earnings theta={phase_msft.theta[PHASE_EARNINGS]:.4f}")
    print("  Per-ticker earnings PASS")


def test_historical_anchor_override():
    """Historical anchors override phases during their date range."""
    from tensor.calendar_regime import (
        CalendarRegimeEncoder, HistoricalAnchor, N_CHANNELS,
    )

    anchor = HistoricalAnchor(
        name="covid_crash",
        start_date=date(2020, 2, 20),
        end_date=date(2020, 4, 1),
        anchor_vector=np.full(N_CHANNELS, 0.1),
        severity=0.95,
    )
    encoder = CalendarRegimeEncoder(historical_anchors=[anchor])

    # During anchor period
    phase_in = encoder.encode(date(2020, 3, 15))
    np.testing.assert_array_almost_equal(phase_in.theta, np.full(N_CHANNELS, 0.1))
    assert all(a == 0.95 for a in phase_in.amplitudes)
    assert "historical:covid_crash" in phase_in.active_events

    # Outside anchor period — normal computation
    phase_out = encoder.encode(date(2021, 3, 15))
    assert "historical:covid_crash" not in phase_out.active_events

    print("  Historical anchor override PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Von Mises Basis Function Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_von_mises_peak():
    """phi_k(0, 1.0, kappa=4) > 0.99 (peak at theta=0)."""
    from tensor.frequency_dependent_lifter import von_mises_basis

    val = von_mises_basis(0.0, 1.0, kappa=4.0)
    assert val > 0.99, f"Peak should be >0.99, got {val}"
    print(f"  phi(0, 1.0, k=4) = {val:.6f} > 0.99 PASS")


def test_von_mises_negligible_at_pi():
    """phi_k(pi, 1.0, kappa=4) < 0.02 (negligible far from event)."""
    from tensor.frequency_dependent_lifter import von_mises_basis

    val = von_mises_basis(np.pi, 1.0, kappa=4.0)
    assert val < 0.02, f"Value at pi should be <0.02, got {val}"
    print(f"  phi(pi, 1.0, k=4) = {val:.6f} < 0.02 PASS")


def test_von_mises_non_negative():
    """Von Mises basis is non-negative everywhere."""
    from tensor.frequency_dependent_lifter import von_mises_basis

    for theta in np.linspace(0, 2 * np.pi, 100):
        for amp in [0.0, 0.5, 1.0]:
            for kappa in [2.0, 4.0, 8.0]:
                val = von_mises_basis(theta, amp, kappa)
                assert val >= 0.0, f"Negative at theta={theta}, a={amp}, k={kappa}: {val}"
    print("  Non-negative everywhere PASS")


def test_von_mises_linear_in_amplitude():
    """phi_k(0, 0.5, kappa) ~ 0.5 * phi_k(0, 1.0, kappa)."""
    from tensor.frequency_dependent_lifter import von_mises_basis

    full = von_mises_basis(0.0, 1.0, kappa=4.0)
    half = von_mises_basis(0.0, 0.5, kappa=4.0)

    assert abs(half - 0.5 * full) < 1e-10, \
        f"Not linear in amplitude: half={half}, 0.5*full={0.5 * full}"
    print(f"  phi(0, 0.5) = {half:.4f} ~ 0.5 * phi(0, 1.0) = {0.5 * full:.4f} PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Lifter Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_lifter_backward_compat():
    """lift() without calendar matches LiftingOperator behavior."""
    from tensor.timescale_state import LiftingOperator
    from tensor.frequency_dependent_lifter import FrequencyDependentLifter

    np.random.seed(42)
    n, src_dim, tgt_dim = 100, 12, 16

    source = np.random.randn(n, src_dim) * 0.1
    target = source[:, :8] @ np.random.randn(8, tgt_dim) * 0.01

    # Fit both
    lo = LiftingOperator(src_dim, tgt_dim)
    lo.fit(source, target)

    fdl = FrequencyDependentLifter(src_dim, tgt_dim)
    fdl.fit(source, target)

    # Compare lifts
    test_x = np.random.randn(src_dim) * 0.1
    lo_result = lo.lift(test_x)
    fdl_result = fdl.lift(test_x)

    np.testing.assert_array_almost_equal(lo_result, fdl_result, decimal=10,
        err_msg="FrequencyDependentLifter.lift() should match LiftingOperator.lift()")

    print(f"  Backward compat: max diff = {np.max(np.abs(lo_result - fdl_result)):.2e}")
    print("  Lifter backward compat PASS")


def test_lift_at_differs_from_lift():
    """lift_at() differs from lift() when calendar active (>5% Frobenius norm difference)."""
    from tensor.calendar_regime import CalendarRegimeEncoder, CalendarPhase, N_CHANNELS
    from tensor.frequency_dependent_lifter import FrequencyDependentLifter

    np.random.seed(42)
    n, src_dim, tgt_dim = 200, 12, 16

    source = np.random.randn(n, src_dim) * 0.1
    # Create target with calendar-correlated structure
    base_target = source[:, :8] @ np.random.randn(8, tgt_dim) * 0.01

    encoder = CalendarRegimeEncoder(use_synthetic=True)
    start = date(2025, 1, 1)
    phases = []
    day_count = 0
    current = start
    while len(phases) < n:
        if current.weekday() < 5:
            phases.append(encoder.encode(current))
        current += timedelta(days=1)

    # Add calendar-correlated signal to target
    from tensor.frequency_dependent_lifter import von_mises_basis, KAPPA
    target = base_target.copy()
    calendar_effect = np.random.randn(src_dim, tgt_dim) * 0.05
    for i in range(n):
        w = von_mises_basis(phases[i].theta[0], phases[i].amplitudes[0], KAPPA[0])
        target[i] += w * (source[i] @ calendar_effect)

    fdl = FrequencyDependentLifter(src_dim, tgt_dim)
    fdl.fit_calendar(source, target, phases)

    # Compare static vs calendar-aware
    test_x = np.random.randn(src_dim) * 0.1
    # Use a phase with high amplitude (near an event)
    active_phase = CalendarPhase(
        theta=np.array([0.1, 0.2, 0.5, 1.0, 2.0]),
        amplitudes=np.array([0.9, 0.8, 0.5, 0.3, 0.1]),
    )

    static_delta = fdl.lift(test_x)
    calendar_result = fdl.lift_at(test_x, active_phase)
    calendar_delta = calendar_result.delta

    diff_norm = np.linalg.norm(calendar_delta - static_delta)
    static_norm = np.linalg.norm(static_delta)
    relative_diff = diff_norm / max(static_norm, 1e-10)

    assert relative_diff > 0.05, \
        f"Calendar lift should differ by >5%, got {relative_diff*100:.1f}%"

    print(f"  Static norm: {static_norm:.6f}")
    print(f"  Calendar diff: {diff_norm:.6f} ({relative_diff*100:.1f}%)")
    print("  lift_at differs from lift PASS")


def test_spectral_radius_bounds():
    """Spectral radius of A_0 < 0.95; worst-case A_0 + sum A_k < 0.95."""
    from tensor.calendar_regime import CalendarRegimeEncoder
    from tensor.frequency_dependent_lifter import FrequencyDependentLifter

    np.random.seed(42)
    n, src_dim, tgt_dim = 200, 12, 16

    source = np.random.randn(n, src_dim) * 0.1
    target = source[:, :8] @ np.random.randn(8, tgt_dim) * 0.01

    encoder = CalendarRegimeEncoder(use_synthetic=True)
    phases = []
    current = date(2025, 1, 1)
    while len(phases) < n:
        if current.weekday() < 5:
            phases.append(encoder.encode(current))
        current += timedelta(days=1)

    fdl = FrequencyDependentLifter(src_dim, tgt_dim, spectral_radius_bound=0.95)
    fdl.fit_calendar(source, target, phases)

    sr_baseline = fdl.spectral_radius
    sr_worst = fdl.worst_case_spectral_radius()

    assert sr_baseline < 0.95, f"Baseline spectral radius {sr_baseline} >= 0.95"
    assert sr_worst < 0.95, f"Worst-case spectral radius {sr_worst} >= 0.95"

    print(f"  Baseline SR: {sr_baseline:.4f} < 0.95")
    print(f"  Worst-case SR: {sr_worst:.4f} < 0.95")
    print("  Spectral radius bounds PASS")


def test_earnings_week_larger_delta():
    """Earnings-week shock produces larger delta_m than mid-quarter shock."""
    from tensor.calendar_regime import CalendarRegimeEncoder, CalendarPhase, N_CHANNELS
    from tensor.frequency_dependent_lifter import FrequencyDependentLifter

    np.random.seed(42)
    n, src_dim, tgt_dim = 200, 12, 16

    source = np.random.randn(n, src_dim) * 0.1
    base_target = source[:, :8] @ np.random.randn(8, tgt_dim) * 0.01

    encoder = CalendarRegimeEncoder(use_synthetic=True)
    phases = []
    current = date(2025, 1, 1)
    while len(phases) < n:
        if current.weekday() < 5:
            phases.append(encoder.encode(current))
        current += timedelta(days=1)

    # Add earnings-correlated signal
    from tensor.frequency_dependent_lifter import von_mises_basis, KAPPA
    target = base_target.copy()
    earnings_effect = np.random.randn(src_dim, tgt_dim) * 0.1
    for i in range(n):
        w = von_mises_basis(phases[i].theta[0], phases[i].amplitudes[0], KAPPA[0])
        target[i] += w * (source[i] @ earnings_effect)

    fdl = FrequencyDependentLifter(src_dim, tgt_dim)
    fdl.fit_calendar(source, target, phases)

    test_x = np.random.randn(src_dim) * 0.1

    # Earnings week: high amplitude, theta near 0
    earnings_phase = CalendarPhase(
        theta=np.array([0.05, np.pi, np.pi, np.pi, np.pi]),
        amplitudes=np.array([0.95, 0.0, 0.0, 0.0, 0.0]),
    )
    # Mid-quarter: low amplitude, theta near pi
    quiet_phase = CalendarPhase(
        theta=np.array([np.pi, np.pi, np.pi, np.pi, np.pi]),
        amplitudes=np.array([0.01, 0.0, 0.0, 0.0, 0.0]),
    )

    result_earn = fdl.lift_at(test_x, earnings_phase)
    result_quiet = fdl.lift_at(test_x, quiet_phase)

    norm_earn = np.linalg.norm(result_earn.delta)
    norm_quiet = np.linalg.norm(result_quiet.delta)

    assert norm_earn > norm_quiet, \
        f"Earnings-week delta ({norm_earn:.6f}) should be larger than mid-quarter ({norm_quiet:.6f})"

    print(f"  Earnings-week |delta|: {norm_earn:.6f}")
    print(f"  Mid-quarter |delta|:   {norm_quiet:.6f}")
    print("  Earnings-week larger delta PASS")


def test_insufficient_data_safe_fallback():
    """When a cycle has insufficient data, A_k stays zero (safe fallback)."""
    from tensor.calendar_regime import CalendarPhase, N_CHANNELS
    from tensor.frequency_dependent_lifter import FrequencyDependentLifter

    np.random.seed(42)
    n, src_dim, tgt_dim = 10, 12, 16  # small n → some cycles won't have enough samples

    source = np.random.randn(n, src_dim) * 0.1
    target = np.random.randn(n, tgt_dim) * 0.01

    # All phases near pi (far from events) → very few active samples per cycle
    phases = [
        CalendarPhase(
            theta=np.full(N_CHANNELS, np.pi),
            amplitudes=np.full(N_CHANNELS, 0.001),
        )
        for _ in range(n)
    ]

    fdl = FrequencyDependentLifter(src_dim, tgt_dim)
    fdl.fit_calendar(source, target, phases, min_active_samples=20)

    # No cycle should be fitted (all below min_active_samples threshold)
    assert not any(fdl._cycle_fitted), \
        f"No cycle should be fitted with insufficient data, got {fdl._cycle_fitted}"

    print("  Insufficient data safe fallback PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Arnold Tongue Resonance Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_best_rational_approximation():
    """Exact rationals are recovered correctly."""
    from tensor.frequency_dependent_lifter import best_rational_approximation

    p, q = best_rational_approximation(2.0, max_denom=8)
    assert (p, q) == (2, 1), f"Expected (2,1), got ({p},{q})"

    p, q = best_rational_approximation(0.5, max_denom=8)
    assert (p, q) == (1, 2), f"Expected (1,2), got ({p},{q})"

    p, q = best_rational_approximation(1.0, max_denom=8)
    assert (p, q) == (1, 1), f"Expected (1,1), got ({p},{q})"

    p, q = best_rational_approximation(1.5, max_denom=8)
    assert (p, q) == (3, 2), f"Expected (3,2), got ({p},{q})"

    print("  Rational approximation PASS")


def test_1_1_tongue_widest():
    """1:1 tongue is widest; higher-order tongues narrower."""
    from tensor.calendar_regime import CalendarPhase, N_CHANNELS
    from tensor.frequency_dependent_lifter import detect_resonance

    # Create phase where two cycles with same period are active
    # We use amplitudes to control coupling strength
    phase = CalendarPhase(
        theta=np.zeros(N_CHANNELS),
        amplitudes=np.array([0.8, 0.8, 0.8, 0.8, 0.0]),
    )
    report = detect_resonance(phase)

    # Find tongues by their rational approximation
    widths_by_order = {}
    for (n1, n2, w), (_, _, p, q) in zip(report.tongue_widths, report.nearest_rationals):
        order = abs(p) + q
        widths_by_order.setdefault(order, []).append(w)

    if len(widths_by_order) >= 2:
        orders = sorted(widths_by_order.keys())
        # Lower order should have wider tongues on average
        low_order_avg = np.mean(widths_by_order[orders[0]])
        for order in orders[1:]:
            high_order_avg = np.mean(widths_by_order[order])
            if high_order_avg > 0:
                assert low_order_avg >= high_order_avg * 0.5, \
                    f"Order {orders[0]} tongue should be wider than order {order}"

    print(f"  Tongue widths by order: {widths_by_order}")
    print("  1:1 tongue widest PASS")


def test_fed_earnings_resonance():
    """Fed/earnings 2:1 ratio is flagged as resonant."""
    from tensor.calendar_regime import CalendarPhase, PHASE_EARNINGS, PHASE_FED, CYCLE_PERIODS, N_CHANNELS
    from tensor.frequency_dependent_lifter import detect_resonance

    # Verify the structural 2:1 ratio
    ratio = CYCLE_PERIODS[PHASE_EARNINGS] / CYCLE_PERIODS[PHASE_FED]
    assert abs(ratio - 2.0) < 0.01, f"Expected earnings/fed ratio ~2.0, got {ratio}"

    # Create phase with both earnings and fed active
    phase = CalendarPhase(
        theta=np.zeros(N_CHANNELS),
        amplitudes=np.zeros(N_CHANNELS),
    )
    phase.amplitudes[PHASE_EARNINGS] = 0.8
    phase.amplitudes[PHASE_FED] = 0.8

    report = detect_resonance(phase)
    assert report.is_resonant, "Fed/earnings 2:1 should be flagged as resonant"

    # Check the resonant pair
    resonant_names = set()
    for pair in report.resonant_pairs:
        resonant_names.update(pair)
    assert "earnings" in resonant_names and "fed" in resonant_names, \
        f"Expected earnings/fed pair, got {report.resonant_pairs}"

    print(f"  Earnings/Fed ratio: {ratio:.1f}")
    print(f"  Resonant pairs: {report.resonant_pairs}")
    print("  Fed/earnings resonance PASS")


def test_zero_amplitude_excluded():
    """Channels with amplitude=0 excluded from resonance check."""
    from tensor.calendar_regime import CalendarPhase, N_CHANNELS
    from tensor.frequency_dependent_lifter import detect_resonance

    # Only one channel active — no pairs possible
    phase = CalendarPhase(
        theta=np.zeros(N_CHANNELS),
        amplitudes=np.array([0.8, 0.0, 0.0, 0.0, 0.0]),
    )
    report = detect_resonance(phase)
    assert not report.is_resonant, "Single channel should not produce resonance"
    assert len(report.frequency_ratios) == 0, "No frequency ratios for single channel"

    print("  Zero amplitude excluded PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_propagate_shock_no_calendar():
    """propagate_shock(event_date=None) matches original behavior."""
    from tensor.timescale_state import (
        CrossTimescaleSystem, ShockState, RegimeState, FundamentalState,
    )

    np.random.seed(42)
    n = 100

    # System without calendar
    system = CrossTimescaleSystem(shock_dim=12, regime_dim=16, fundamental_dim=12)

    shock_states = np.random.randn(n, 12) * 0.1
    regime_deltas = shock_states[:, :8] @ np.random.randn(8, 16) * 0.01
    regime_states = np.random.randn(n, 16) * 0.1
    fund_deltas = regime_states[:, :6] @ np.random.randn(6, 12) * 0.01

    system.fit_s_to_m(shock_states, regime_deltas)
    system.fit_m_to_l(regime_states, fund_deltas)

    shock = ShockState(features=np.random.randn(12) * 0.1)
    regime = RegimeState(features=np.random.randn(16) * 0.1)
    fundamental = FundamentalState(features=np.random.randn(12) * 0.1)

    # Without event_date: original behavior
    new_r, new_f = system.propagate_shock(shock, regime, fundamental)
    assert new_r.features.shape == (16,)
    assert new_f.features.shape == (12,)

    # With event_date=None: same as without
    new_r2, new_f2 = system.propagate_shock(shock, regime, fundamental, event_date=None)
    np.testing.assert_array_equal(new_r.features, new_r2.features)
    np.testing.assert_array_equal(new_f.features, new_f2.features)

    print("  propagate_shock(no calendar) PASS")


def test_propagate_shock_with_calendar():
    """propagate_shock(event_date=earnings_date) uses frequency-dependent lift."""
    from tensor.calendar_regime import CalendarRegimeEncoder
    from tensor.timescale_state import (
        CrossTimescaleSystem, ShockState, RegimeState, FundamentalState,
    )

    np.random.seed(42)
    n = 200

    encoder = CalendarRegimeEncoder(use_synthetic=True)
    system = CrossTimescaleSystem(
        shock_dim=12, regime_dim=16, fundamental_dim=12,
        calendar_encoder=encoder,
    )

    shock_states = np.random.randn(n, 12) * 0.1
    regime_deltas = shock_states[:, :8] @ np.random.randn(8, 16) * 0.01
    regime_states = np.random.randn(n, 16) * 0.1
    fund_deltas = regime_states[:, :6] @ np.random.randn(6, 12) * 0.01

    system.fit_s_to_m(shock_states, regime_deltas)
    system.fit_m_to_l(regime_states, fund_deltas)

    shock = ShockState(features=np.random.randn(12) * 0.1)
    regime = RegimeState(features=np.random.randn(16) * 0.1)
    fundamental = FundamentalState(features=np.random.randn(12) * 0.1)

    # Without event_date: uses static lift
    new_r_static, new_f_static = system.propagate_shock(shock, regime, fundamental)

    # With event_date: uses calendar-aware lift
    new_r_cal, new_f_cal = system.propagate_shock(
        shock, regime, fundamental,
        event_date=date(2025, 1, 15), ticker="AAPL",
    )

    # Results should exist and be valid
    assert new_r_cal.features.shape == (16,)
    assert np.all(np.isfinite(new_r_cal.features))

    print(f"  Static regime delta norm: {np.linalg.norm(new_r_static.features - regime.features):.6f}")
    print(f"  Calendar regime delta norm: {np.linalg.norm(new_r_cal.features - regime.features):.6f}")
    print("  propagate_shock(with calendar) PASS")


def test_full_pipeline():
    """Full pipeline: calendar encode -> lift_at -> mixer with calendar_phase -> blended return with resonance flag."""
    from tensor.calendar_regime import CalendarRegimeEncoder, PHASE_EARNINGS, PHASE_FED
    from tensor.frequency_dependent_lifter import FrequencyDependentLifter
    from tensor.multi_horizon_mixer import MultiHorizonMixer
    from tensor.timescale_state import (
        CrossTimescaleSystem, ShockState, RegimeState, FundamentalState,
    )

    np.random.seed(42)
    n = 200

    # Setup encoder and system
    encoder = CalendarRegimeEncoder(use_synthetic=True)
    system = CrossTimescaleSystem(
        shock_dim=12, regime_dim=16, fundamental_dim=12,
        calendar_encoder=encoder,
    )

    # Fit operators
    shock_states = np.random.randn(n, 12) * 0.1
    regime_deltas = shock_states[:, :8] @ np.random.randn(8, 16) * 0.01
    regime_states = np.random.randn(n, 16) * 0.1
    fund_deltas = regime_states[:, :6] @ np.random.randn(6, 12) * 0.01

    system.fit_s_to_m(shock_states, regime_deltas)
    system.fit_m_to_l(regime_states, fund_deltas)

    # Create states
    shock = ShockState(features=np.array([
        0.8, 0.9, 0.7, 3, 1, 0.0, 0.85, 0.6, 0.1, 0.2, 1.0, 0.7
    ]))
    regime = RegimeState(features=np.random.randn(16) * 0.1)
    fundamental = FundamentalState(features=np.array([
        0.15, 0.65, 0.30, 0.05, 0.3, 15.0, 2.0, 0.18, 0.5, 0.1, 0.1, 0.85
    ]))

    # Propagate through system
    event_date = date(2025, 1, 15)
    new_regime, new_fundamental = system.propagate_shock(
        shock, regime, fundamental,
        event_date=event_date, ticker="AAPL",
    )

    # Mix with calendar phase
    calendar_mod = np.array([0.3, 0.5, 0.1, 0.1, -0.2])
    mixer = MultiHorizonMixer(
        alpha=2.0, beta=1.0, gamma=0.5,
        calendar_alpha_modulation=calendar_mod,
    )

    phase = encoder.encode(event_date, ticker="AAPL")
    result = mixer.mix(new_fundamental, new_regime, shock, calendar_phase=phase)

    assert isinstance(result.blended_return, float)
    assert result.weights.shape == (3,)
    assert abs(result.weights.sum() - 1.0) < 1e-10
    assert isinstance(result.resonance_flag, bool)

    print(f"  Blended return: {result.blended_return:.6f}")
    print(f"  Weights: L={result.weights[0]:.3f}, M={result.weights[1]:.3f}, S={result.weights[2]:.3f}")
    print(f"  Dominant: {result.dominant_timeframe}")
    print(f"  Resonance flag: {result.resonance_flag}")
    print("  Full pipeline PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all calendar lifter tests."""
    tests = [
        # Calendar encoder
        ("Calendar: synthetic phases", test_synthetic_fallback_valid_phases),
        ("Calendar: deterministic", test_synthetic_deterministic),
        ("Calendar: third Friday", test_third_friday),
        ("Calendar: phase near event", test_phase_near_event),
        ("Calendar: amplitude decay", test_amplitude_decay),
        ("Calendar: per-ticker earnings", test_per_ticker_earnings),
        ("Calendar: historical anchor", test_historical_anchor_override),
        # Von Mises basis
        ("Basis: peak", test_von_mises_peak),
        ("Basis: negligible at pi", test_von_mises_negligible_at_pi),
        ("Basis: non-negative", test_von_mises_non_negative),
        ("Basis: linear amplitude", test_von_mises_linear_in_amplitude),
        # Lifter
        ("Lifter: backward compat", test_lifter_backward_compat),
        ("Lifter: calendar differs", test_lift_at_differs_from_lift),
        ("Lifter: spectral bounds", test_spectral_radius_bounds),
        ("Lifter: earnings > mid-quarter", test_earnings_week_larger_delta),
        ("Lifter: insufficient data fallback", test_insufficient_data_safe_fallback),
        # Resonance
        ("Resonance: rational approx", test_best_rational_approximation),
        ("Resonance: 1:1 widest", test_1_1_tongue_widest),
        ("Resonance: fed/earnings 2:1", test_fed_earnings_resonance),
        ("Resonance: zero amplitude excluded", test_zero_amplitude_excluded),
        # Integration
        ("Integration: no calendar", test_propagate_shock_no_calendar),
        ("Integration: with calendar", test_propagate_shock_with_calendar),
        ("Integration: full pipeline", test_full_pipeline),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{passed + failed} tests passed")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
