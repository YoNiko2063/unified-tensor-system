"""
GET /api/v1/calendar/phase?date=YYYY-MM-DD  -> CalendarPhase for a single date
GET /api/v1/calendar/range?start=YYYY-MM-DD&end=YYYY-MM-DD -> phase series
"""
import datetime
from typing import List
import numpy as np
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from tensor.calendar_regime import CalendarRegimeEncoder

router = APIRouter()

# Channel labels in order matching the encoder's amplitude array
CHANNEL_LABELS = ["earnings", "fed", "options", "rebalance", "holiday"]

_encoder: CalendarRegimeEncoder | None = None


def _get_encoder() -> CalendarRegimeEncoder:
    global _encoder
    if _encoder is None:
        _encoder = CalendarRegimeEncoder(use_synthetic=True)
    return _encoder


class PhaseVector(BaseModel):
    channels: List[str]
    amplitudes: List[float]
    active_events: List[str]
    dominant_cycle: str
    regime_label: str
    resonance_detected: bool
    vol_multiplier: float
    date: str


def _phase_for_date(d: datetime.date) -> PhaseVector:
    enc = _get_encoder()
    cp = enc.encode(d)

    amps = cp.amplitudes.tolist()
    dominant_idx = int(np.argmax(amps))
    dominant = CHANNEL_LABELS[dominant_idx] if dominant_idx < len(CHANNEL_LABELS) else "unknown"

    # Resonance: top-2 amplitudes ratio <= 2:1
    sorted_amps = sorted(amps, reverse=True)
    resonance = False
    if len(sorted_amps) >= 2 and sorted_amps[1] > 0.05:
        ratio = sorted_amps[0] / (sorted_amps[1] + 1e-12)
        resonance = ratio <= 2.5

    # Vol multiplier: proportional to top amplitude above baseline
    vol_mult = 1.0 + float(sorted_amps[0]) * 0.20

    # Regime label based on active events
    if cp.active_events:
        label = "+".join(cp.active_events[:2]).upper()
    else:
        label = dominant.upper()

    return PhaseVector(
        channels=CHANNEL_LABELS,
        amplitudes=amps,
        active_events=cp.active_events,
        dominant_cycle=dominant,
        regime_label=label,
        resonance_detected=resonance,
        vol_multiplier=round(vol_mult, 3),
        date=d.isoformat(),
    )


@router.get("/phase", response_model=PhaseVector)
def get_calendar_phase(date: str = Query(default=None)) -> PhaseVector:
    """Return 5-channel calendar phase vector for a given date (YYYY-MM-DD)."""
    if date is None:
        d = datetime.date.today()
    else:
        try:
            d = datetime.date.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid date format: {date!r}. Use YYYY-MM-DD.")
    return _phase_for_date(d)


class PhaseRange(BaseModel):
    dates: List[str]
    series: List[PhaseVector]


@router.get("/range", response_model=PhaseRange)
def get_calendar_range(
    start: str = Query(...),
    end: str = Query(...),
) -> PhaseRange:
    """Return phase vector series for a date range (inclusive)."""
    try:
        d_start = datetime.date.fromisoformat(start)
        d_end = datetime.date.fromisoformat(end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if d_end < d_start:
        raise HTTPException(status_code=422, detail="end must be >= start")

    delta = (d_end - d_start).days
    if delta > 366:
        raise HTTPException(status_code=422, detail="Range too large (max 366 days)")

    dates = [d_start + datetime.timedelta(days=i) for i in range(delta + 1)]
    series = [_phase_for_date(d) for d in dates]

    return PhaseRange(
        dates=[d.isoformat() for d in dates],
        series=series,
    )
