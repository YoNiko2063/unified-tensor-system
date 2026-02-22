"""Calendar Regime Encoder — date to phase vector mapping.

Maps calendar dates to a 5-channel phase vector locked to known event cycles:
  [earnings, fed, options_expiry, rebalance, holiday]

Each channel encodes:
  - theta: phase angle in [0, 2pi) relative to nearest event
  - amplitude: proximity weight in [0, 1] (von Mises-like decay)

Phase convention: theta=0 means event is NOW, theta=pi means maximally far.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

# Channel indices
PHASE_EARNINGS = 0
PHASE_FED = 1
PHASE_OPTIONS = 2
PHASE_REBALANCE = 3
PHASE_HOLIDAY = 4

CHANNEL_NAMES = ["earnings", "fed", "options_expiry", "rebalance", "holiday"]
N_CHANNELS = 5

# Cycle periods in trading days
CYCLE_PERIODS = np.array([63.0, 31.5, 21.0, 63.0, 252.0])

# Amplitude halflife in trading days per cycle
HALFLIFE = np.array([5.0, 2.0, 3.0, 3.0, 2.0])


@dataclass
class CalendarPhase:
    """Phase vector for a single date."""
    theta: np.ndarray = field(default_factory=lambda: np.zeros(N_CHANNELS))
    amplitudes: np.ndarray = field(default_factory=lambda: np.zeros(N_CHANNELS))
    active_events: List[str] = field(default_factory=list)
    regime_prior: np.ndarray = field(default_factory=lambda: np.zeros(N_CHANNELS))


@dataclass
class HistoricalAnchor:
    """Fixed HDV anchor for a structural historical event."""
    name: str = ""
    start_date: date = date(2020, 1, 1)
    end_date: date = date(2020, 1, 1)
    anchor_vector: np.ndarray = field(default_factory=lambda: np.zeros(N_CHANNELS))
    severity: float = 0.0


def third_friday(year: int, month: int) -> date:
    """Compute the 3rd Friday of the given month (options expiry / rebalance)."""
    # First day of the month
    first_day = date(year, month, 1)
    # Day of week: 0=Monday, 4=Friday
    first_dow = first_day.weekday()
    # Days until the first Friday
    days_to_friday = (4 - first_dow) % 7
    first_friday = first_day + timedelta(days=days_to_friday)
    # Third Friday = first Friday + 14 days
    return first_friday + timedelta(days=14)


def _trading_days_between(d1: date, d2: date) -> float:
    """Approximate trading days between two dates (weekdays only)."""
    if d1 == d2:
        return 0.0
    sign = 1.0
    if d2 < d1:
        d1, d2 = d2, d1
        sign = -1.0
    count = 0
    current = d1
    while current < d2:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            count += 1
    return sign * count


class CalendarRegimeEncoder:
    """Encodes calendar dates into phase vectors for frequency-dependent lifting.

    Parameters:
        fed_dates: List of FOMC meeting dates.
        holiday_dates: List of market holiday dates.
        historical_anchors: List of HistoricalAnchor for structural events.
        earnings_dates: Dict[ticker, List[date]] for per-ticker earnings.
        use_synthetic: If True, generate deterministic periodic cycles for testing.
    """

    def __init__(
        self,
        fed_dates: Optional[List[date]] = None,
        holiday_dates: Optional[List[date]] = None,
        historical_anchors: Optional[List[HistoricalAnchor]] = None,
        earnings_dates: Optional[Dict[str, List[date]]] = None,
        use_synthetic: bool = False,
    ) -> None:
        self._fed_dates = sorted(fed_dates) if fed_dates else []
        self._holiday_dates = sorted(holiday_dates) if holiday_dates else []
        self._historical_anchors = historical_anchors or []
        self._earnings_dates = earnings_dates or {}
        self._use_synthetic = use_synthetic

    def encode(self, query_date: date, ticker: Optional[str] = None) -> CalendarPhase:
        """Encode a single date into a CalendarPhase vector.

        Args:
            query_date: The date to encode.
            ticker: Optional ticker for per-ticker earnings overlay.

        Returns:
            CalendarPhase with theta, amplitudes, active_events, regime_prior.
        """
        if self._use_synthetic:
            return self._encode_synthetic(query_date)

        theta = np.full(N_CHANNELS, np.pi)
        amplitudes = np.zeros(N_CHANNELS)
        active_events: List[str] = []

        # Per-channel computation
        t_e, a_e = self._compute_earnings_phase(query_date, ticker)
        t_f, a_f = self._compute_fed_phase(query_date)
        t_o, a_o = self._compute_options_phase(query_date)
        t_r, a_r = self._compute_rebalance_phase(query_date)
        t_h, a_h = self._compute_holiday_phase(query_date)

        theta[PHASE_EARNINGS] = t_e
        theta[PHASE_FED] = t_f
        theta[PHASE_OPTIONS] = t_o
        theta[PHASE_REBALANCE] = t_r
        theta[PHASE_HOLIDAY] = t_h

        amplitudes[PHASE_EARNINGS] = a_e
        amplitudes[PHASE_FED] = a_f
        amplitudes[PHASE_OPTIONS] = a_o
        amplitudes[PHASE_REBALANCE] = a_r
        amplitudes[PHASE_HOLIDAY] = a_h

        # Active events: amplitude > 0.3 threshold
        for i, name in enumerate(CHANNEL_NAMES):
            if amplitudes[i] > 0.3:
                active_events.append(name)

        # Historical anchor override
        for anchor in self._historical_anchors:
            if anchor.start_date <= query_date <= anchor.end_date:
                # Override with anchor phases and amplitudes
                theta = anchor.anchor_vector.copy()
                amplitudes = np.full(N_CHANNELS, anchor.severity)
                active_events = [f"historical:{anchor.name}"]
                break

        # Regime prior: amplitude-weighted prior for elevated vol
        regime_prior = amplitudes * np.array([0.6, 0.8, 0.3, 0.2, 0.1])

        return CalendarPhase(
            theta=theta,
            amplitudes=amplitudes,
            active_events=active_events,
            regime_prior=regime_prior,
        )

    def encode_range(
        self,
        start_date: date,
        end_date: date,
        ticker: Optional[str] = None,
    ) -> List[CalendarPhase]:
        """Encode a date range into a list of CalendarPhase vectors."""
        phases = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # trading days only
                phases.append(self.encode(current, ticker))
            current += timedelta(days=1)
        return phases

    def _encode_synthetic(self, query_date: date) -> CalendarPhase:
        """Deterministic periodic cycles for testing.

        Uses day-of-year as base, wraps into each cycle period.
        """
        # Days since epoch (deterministic)
        epoch = date(2020, 1, 1)
        day_offset = (query_date - epoch).days

        theta = np.zeros(N_CHANNELS)
        amplitudes = np.zeros(N_CHANNELS)

        for i in range(N_CHANNELS):
            period = CYCLE_PERIODS[i]
            # Phase wraps with cycle period (convert calendar days to approx trading days)
            trading_approx = day_offset * 5.0 / 7.0
            phase = (trading_approx % period) / period * 2.0 * np.pi
            theta[i] = phase
            # Amplitude peaks at theta=0 (start of cycle)
            amplitudes[i] = float(np.exp(-abs(np.cos(phase / 2.0) - 1.0) / 0.5))

        active_events = [
            CHANNEL_NAMES[i] for i in range(N_CHANNELS) if amplitudes[i] > 0.3
        ]

        regime_prior = amplitudes * np.array([0.6, 0.8, 0.3, 0.2, 0.1])

        return CalendarPhase(
            theta=theta,
            amplitudes=amplitudes,
            active_events=active_events,
            regime_prior=regime_prior,
        )

    def _compute_phase_from_dates(
        self,
        query_date: date,
        event_dates: List[date],
        cycle_period: float,
        halflife: float,
    ) -> tuple:
        """Generic phase/amplitude computation from a list of event dates.

        Returns (theta, amplitude).
        """
        if not event_dates:
            return np.pi, 0.0

        # Find nearest event (past or future)
        min_dist = float("inf")
        for ed in event_dates:
            dist = _trading_days_between(query_date, ed)
            if abs(dist) < abs(min_dist):
                min_dist = dist

        # Phase: 0 at event, pi at maximally far
        theta = 2.0 * np.pi * min_dist / cycle_period
        # Wrap to [0, 2pi)
        theta = theta % (2.0 * np.pi)

        # Amplitude: exponential decay from event
        amplitude = float(np.exp(-abs(min_dist) / halflife))

        return theta, amplitude

    def _compute_earnings_phase(self, query_date: date, ticker: Optional[str] = None) -> tuple:
        """Earnings cycle phase. Uses per-ticker dates if available."""
        dates = []
        if ticker and ticker in self._earnings_dates:
            dates = self._earnings_dates[ticker]
        else:
            # Default: quarterly from standard reporting months (Jan, Apr, Jul, Oct)
            year = query_date.year
            for m in [1, 4, 7, 10]:
                # Approximate: 3rd week of reporting month
                dates.append(date(year, m, 15))
                if year > 2020:
                    dates.append(date(year - 1, m, 15))
                dates.append(date(year + 1, m, 15))

        return self._compute_phase_from_dates(
            query_date, dates, CYCLE_PERIODS[PHASE_EARNINGS], HALFLIFE[PHASE_EARNINGS]
        )

    def _compute_fed_phase(self, query_date: date) -> tuple:
        """FOMC meeting cycle phase."""
        if self._fed_dates:
            return self._compute_phase_from_dates(
                query_date, self._fed_dates, CYCLE_PERIODS[PHASE_FED], HALFLIFE[PHASE_FED]
            )
        # Default: ~8 meetings per year, roughly every 6 weeks
        year = query_date.year
        default_months = [1, 3, 5, 6, 7, 9, 11, 12]
        dates = []
        for y in [year - 1, year, year + 1]:
            for m in default_months:
                try:
                    dates.append(date(y, m, 15))
                except ValueError:
                    pass
        return self._compute_phase_from_dates(
            query_date, dates, CYCLE_PERIODS[PHASE_FED], HALFLIFE[PHASE_FED]
        )

    def _compute_options_phase(self, query_date: date) -> tuple:
        """Monthly options expiry cycle (3rd Friday)."""
        year = query_date.year
        dates = []
        for y in [year - 1, year, year + 1]:
            for m in range(1, 13):
                dates.append(third_friday(y, m))
        return self._compute_phase_from_dates(
            query_date, dates, CYCLE_PERIODS[PHASE_OPTIONS], HALFLIFE[PHASE_OPTIONS]
        )

    def _compute_rebalance_phase(self, query_date: date) -> tuple:
        """Quarterly rebalance cycle (3rd Friday of Mar, Jun, Sep, Dec)."""
        year = query_date.year
        dates = []
        for y in [year - 1, year, year + 1]:
            for m in [3, 6, 9, 12]:
                dates.append(third_friday(y, m))
        return self._compute_phase_from_dates(
            query_date, dates, CYCLE_PERIODS[PHASE_REBALANCE], HALFLIFE[PHASE_REBALANCE]
        )

    def _compute_holiday_phase(self, query_date: date) -> tuple:
        """Holiday proximity phase."""
        if self._holiday_dates:
            return self._compute_phase_from_dates(
                query_date, self._holiday_dates, CYCLE_PERIODS[PHASE_HOLIDAY], HALFLIFE[PHASE_HOLIDAY]
            )
        # Default: major US holidays (approximate)
        year = query_date.year
        dates = []
        for y in [year - 1, year, year + 1]:
            dates.extend([
                date(y, 1, 1),    # New Year
                date(y, 1, 20),   # MLK (approx)
                date(y, 2, 17),   # Presidents Day (approx)
                date(y, 5, 26),   # Memorial Day (approx)
                date(y, 7, 4),    # Independence Day
                date(y, 9, 1),    # Labor Day (approx)
                date(y, 11, 27),  # Thanksgiving (approx)
                date(y, 12, 25),  # Christmas
            ])
        return self._compute_phase_from_dates(
            query_date, dates, CYCLE_PERIODS[PHASE_HOLIDAY], HALFLIFE[PHASE_HOLIDAY]
        )
