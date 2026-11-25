"""Linke turbidity climatology.

The bundled table ``linke_turbidity.bin.gz`` is the SoDa / Meteotest monthly
Linke turbidity climatology (Remund et al., 2003) as distributed with pvlib
(``LinkeTurbidities.h5``), block-averaged from 1/12 deg to 0.5 deg.

Layout: uint8, C order, shape (360 lat rows, 720 lon cols, 12 months).
Row 0 is the band 90N..89.5N, column 0 is 180W..179.5W. Stored values are
TL * 20.
"""

import calendar
from datetime import date
import gzip
from pathlib import Path

_TABLE_PATH = Path(__file__).with_name("linke_turbidity.bin.gz")
_ROWS = 360
_COLS = 720
_MONTHS = 12
_CELL = 0.5  # degrees


def _read_cell(raw: bytes, row: int, col: int) -> list[float]:
    offset = (row * _COLS + col) * _MONTHS
    return [b / 20.0 for b in raw[offset : offset + _MONTHS]]


def load_monthly_linke_turbidity(latitude: float, longitude: float) -> list[float]:
    """Return 12 monthly Linke turbidity values for a location.

    Values are bilinearly interpolated between 0.5 deg cell centres. This does
    blocking file I/O and decompression (~3 MB) and must be run in an
    executor when called from the event loop.
    """
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude out of range")
    longitude = ((longitude + 180.0) % 360.0) - 180.0

    raw = gzip.decompress(_TABLE_PATH.read_bytes())
    if len(raw) != _ROWS * _COLS * _MONTHS:
        raise ValueError("Linke turbidity table is corrupt")

    # Continuous index where integer values are cell centres.
    fr = (90.0 - latitude) / _CELL - 0.5
    fc = (longitude + 180.0) / _CELL - 0.5

    r0 = int(fr // 1)
    c0 = int(fc // 1)
    tr = fr - r0
    tc = fc - c0
    r1 = r0 + 1
    c1 = c0 + 1
    # Clamp rows at the poles, wrap columns around the dateline.
    r0 = min(max(r0, 0), _ROWS - 1)
    r1 = min(max(r1, 0), _ROWS - 1)
    c0 %= _COLS
    c1 %= _COLS

    v00 = _read_cell(raw, r0, c0)
    v01 = _read_cell(raw, r0, c1)
    v10 = _read_cell(raw, r1, c0)
    v11 = _read_cell(raw, r1, c1)
    return [
        (1 - tr) * ((1 - tc) * v00[m] + tc * v01[m]) + tr * ((1 - tc) * v10[m] + tc * v11[m])
        for m in range(_MONTHS)
    ]


def _month_middles(year: int) -> list[float]:
    """Mid-month day-of-year positions with Dec(prev) and Jan(next) padding."""
    mdays = [calendar.monthrange(year, m)[1] for m in range(1, 13)]
    ydays = 366 if calendar.isleap(year) else 365
    middles: list[float] = [-calendar.monthrange(year - 1, 12)[1] / 2.0]
    cum = 0
    for days in mdays:
        middles.append(cum + days / 2.0)
        cum += days
    middles.append(ydays + calendar.monthrange(year + 1, 1)[1] / 2.0)
    return middles


def interpolate_linke_turbidity(
    monthly: list[float], day: date, day_fraction: float = 0.5
) -> float:
    """Interpolate monthly values (treated as mid-month) to a given day.

    ``day_fraction`` is the fraction of the day elapsed (0..1); the position
    on the year axis is ``(day_of_year - 1) + day_fraction`` so that Jan 1
    00:00 is 0.0 and the January value sits at 15.5. This is the same scheme
    as pvlib's ``lookup_linke_turbidity(interp_turbidity=True)`` to within
    half a day, far below the precision of a monthly climatology.
    """
    if len(monthly) != _MONTHS:
        raise ValueError("expected 12 monthly values")
    x = day.timetuple().tm_yday - 1 + day_fraction
    xs = _month_middles(day.year)
    ys = [monthly[11], *monthly, monthly[0]]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    # Unreachable for valid dates, but stay safe.
    return monthly[day.month - 1]
