"""Site calibration from measured PV production.

Inputs are hourly records: the forecast weather (cloud cover, temperature) as
it stood for the hour (near-nowcast), optionally the forecast weather for the
same hour as it stood at the end of the previous day (day-ahead), and the
measured AC energy.

All comparisons are made on the DC side: measured AC energy is converted to
DC-equivalent with the inverter model, and the model's DC output is used, so
the factors can be applied before the inverter model at forecast time.

1. **System factor and horizon shading.** From hours forecast as clear, the
   ratio ``measured / clear-sky model`` is collected. A recency-weighted
   upper-middle quantile (robust to hours that were cloudier than forecast)
   gives the system factor ``g`` (real losses, soiling, orientation and
   rating errors). In sun-position bins the same statistic gives ``S``; the
   shortfall relative to ``g`` is attributed to obstruction of the *beam*
   only: ``s = 1 - (1 - S/g) / b`` where ``b`` is the beam share of the
   effective irradiance. At forecast time the system factor is
   ``g * (1 - b + b * s(az, el))`` with ``b`` from the forecast conditions,
   so shading matters on clear hours and vanishes under overcast.

2. **Cloud response.** In 10 % cloud-cover bands, the energy-weighted ratio
   ``measured / (model x system factor)`` corrects the provider's cloud-cover
   bias and the generic cloud relation for this site, including the average
   effect of the provider's forecast misses. Only anomalies are rejected:
   likely snow cover (cold hours far below the overall median ratio) and
   metering glitches (far above the band median); outages are caught by the
   minimum ratio to clear sky. Each band is shrunk towards the
   kernel-smoothed ratio of its neighbouring bands, itself shrunk to 1.
   Two curves are fitted: one on near-nowcast weather and one on day-ahead
   weather; at forecast time they are blended by lead time.

The model output for training hours is recomputed from the stored weather
with the *current* model, so changing the configuration does not corrupt the
calibration, and calibrated output never feeds back into training.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from typing import TYPE_CHECKING

from .pvsystem import inverter_dc_from_ac
from .solar import solar_position

if TYPE_CHECKING:
    from .forecast import PVModel

AZ_BIN = 15.0
EL_BIN = 10.0
N_AZ = 24
N_EL = 9
N_CLOUD = 10

CLEAR_CLOUD_MAX = 0.25
"""Forecast cloud fraction at or below which an hour counts as clear."""
CLEAR_QUANTILE = 0.7
MIN_CLEAR_FRACTION = 0.05
"""Hours whose clear-sky DC energy is below this fraction of nameplate x 1 h
are ignored (low sun: noisy ratios, timing errors dominate)."""
RATIO_MIN = 0.02
"""measured / clear-sky below this is treated as an outage, not weather."""
RATIO_MAX = 1.6
"""measured / clear-sky above this is treated as a metering glitch."""
OUTLIER_HIGH = 4.0
"""Stage 2 rejects hours above this x the band's median ratio (glitches)."""
SNOW_TEMPERATURE = 4.0
SNOW_RATIO = 0.5
"""Hours at or below SNOW_TEMPERATURE degC with a ratio below SNOW_RATIO x
the median over all bands are treated as snow-covered and excluded."""
HALF_LIFE_DAYS = 30.0
GLOBAL_PRIOR_WEIGHT = 5.0
BIN_PRIOR_WEIGHT = 3.0
CLOUD_PRIOR_HOURS = 6.0
"""Prior weight for cloud bands, in (recency-weighted) pseudo-hours."""
CLOUD_KERNEL_SIGMA = 1.0
"""Width (in bands) of the Gaussian smoothing across cloud bands."""
MIN_BEAM_FOR_SHADING = 0.3
"""Bins whose clear-sky beam share is below this cannot attribute a
shortfall to beam obstruction."""
FACTOR_LIMITS = (0.3, 1.5)
SHADING_LIMITS = (0.0, 1.2)
CLOUD_FACTOR_LIMITS = (0.2, 4.0)
NOWCAST_LEAD_HOURS = 3.0
DAY_AHEAD_LEAD_HOURS = 12.0
"""The cloud response blends linearly from the nowcast curve (lead <= 3 h)
to the day-ahead curve (lead >= 12 h)."""


@dataclass(frozen=True, slots=True)
class HourRecord:
    """One elapsed hour: forecast weather as it stood and measured energy."""

    start: datetime
    cloud_cover: float
    temperature: float
    actual_wh: float | None
    cloud_cover_day_ahead: float | None = None
    temperature_day_ahead: float | None = None


@dataclass(frozen=True, slots=True)
class Calibration:
    """Learnt correction factors."""

    global_factor: float = 1.0
    shading: dict[tuple[int, int], float] = field(default_factory=dict)
    """Beam transmission per (azimuth bin, elevation bin); missing = 1."""
    cloud_factors: tuple[float, ...] = (1.0,) * N_CLOUD
    """Cloud response for near-nowcast weather."""
    cloud_factors_day_ahead: tuple[float, ...] = (1.0,) * N_CLOUD
    """Cloud response for day-ahead weather."""
    clear_hours: float = 0.0
    """Recency-weighted number of clear hours used for stage 1."""
    hours: int = 0
    """Number of hours used for the nowcast cloud response."""
    hours_day_ahead: int = 0
    rejected_hours: int = 0
    """Hours rejected as outliers, snow, outage or glitch."""
    fitted_at: datetime | None = None

    @property
    def active(self) -> bool:
        """True when fitted from at least one usable hour."""
        return self.hours > 0 or self.clear_hours > 0

    def beam_transmission(self, azimuth: float, elevation: float) -> float:
        """Bilinearly interpolated beam transmission (horizon shading)."""
        if not self.shading:
            return 1.0
        fa = (azimuth % 360.0) / AZ_BIN - 0.5
        fe = min(max(elevation / EL_BIN - 0.5, 0.0), N_EL - 1.0)
        a0 = math.floor(fa)
        ta = fa - a0
        e0 = math.floor(fe)
        e1 = min(e0 + 1, N_EL - 1)
        te = fe - e0

        def get(ai: int, ei: int) -> float:
            return self.shading.get((ai % N_AZ, ei), 1.0)

        top = (1 - ta) * get(a0, e0) + ta * get(a0 + 1, e0)
        bottom = (1 - ta) * get(a0, e1) + ta * get(a0 + 1, e1)
        return (1 - te) * top + te * bottom

    def system_factor(self, azimuth: float, elevation: float, beam_fraction: float) -> float:
        """System factor including beam-only horizon shading."""
        b = min(max(beam_fraction, 0.0), 1.0)
        s = self.beam_transmission(azimuth, elevation)
        return self.global_factor * (1.0 - b + b * s)

    @staticmethod
    def _curve(factors: tuple[float, ...], cloud_cover: float) -> float:
        x = min(max(cloud_cover, 0.0), 1.0) * N_CLOUD - 0.5
        if x <= 0:
            return factors[0]
        if x >= N_CLOUD - 1:
            return factors[-1]
        i = int(x)
        t = x - i
        return (1 - t) * factors[i] + t * factors[i + 1]

    def cloud_factor(self, cloud_cover: float, lead_hours: float = 0.0) -> float:
        """Cloud response, blended from nowcast to day-ahead by lead time."""
        now = self._curve(self.cloud_factors, cloud_cover)
        ahead = self._curve(self.cloud_factors_day_ahead, cloud_cover)
        w = (lead_hours - NOWCAST_LEAD_HOURS) / (DAY_AHEAD_LEAD_HOURS - NOWCAST_LEAD_HOURS)
        w = min(max(w, 0.0), 1.0)
        return (1 - w) * now + w * ahead

    def factor(
        self,
        azimuth: float,
        elevation: float,
        beam_fraction: float,
        cloud_cover: float,
        lead_hours: float = 0.0,
    ) -> float:
        """Total multiplicative DC correction."""
        return self.system_factor(azimuth, elevation, beam_fraction) * self.cloud_factor(
            cloud_cover, lead_hours
        )

    @property
    def clear_sky_factor(self) -> float:
        """Effective correction for clear sky at an unshaded sun position."""
        return self.global_factor * self.cloud_factors[0]

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable summary for attributes/diagnostics."""
        return {
            "clear_sky_factor": round(self.clear_sky_factor, 4),
            "system_factor": round(self.global_factor, 4),
            "cloud_factors": [round(f, 4) for f in self.cloud_factors],
            "cloud_factors_day_ahead": [round(f, 4) for f in self.cloud_factors_day_ahead],
            "shaded_bins": sum(1 for v in self.shading.values() if v < 0.95),
            "clear_hours": round(self.clear_hours, 1),
            "hours": self.hours,
            "hours_day_ahead": self.hours_day_ahead,
            "rejected_hours": self.rejected_hours,
            "fitted_at": self.fitted_at.isoformat() if self.fitted_at else None,
        }


IDENTITY = Calibration()


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Weighted quantile.

    Each value is placed at the midpoint of its span on the cumulative weight
    axis and the quantile is linearly interpolated between those points
    (for equal weights this is the Hazen definition).
    """
    if not values:
        raise ValueError("no values")
    pairs = sorted(zip(values, weights, strict=True))
    total = sum(w for _, w in pairs)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    mids: list[float] = []
    cum = 0.0
    for _, w in pairs:
        mids.append(cum + w / 2.0)
        cum += w
    target = q * total
    if target <= mids[0]:
        return pairs[0][0]
    if target >= mids[-1]:
        return pairs[-1][0]
    for i in range(len(mids) - 1):
        if mids[i] <= target <= mids[i + 1]:
            span = mids[i + 1] - mids[i]
            t = (target - mids[i]) / span if span > 0 else 0.0
            return pairs[i][0] + (pairs[i + 1][0] - pairs[i][0]) * t
    return pairs[-1][0]


@dataclass(frozen=True, slots=True)
class HourModel:
    """Model output for one hour with constant weather (DC side)."""

    dc_wh: float
    clear_dc_wh: float
    beam_fraction: float
    """DC-energy-weighted beam share under the given cloud."""
    clear_beam_fraction: float
    clipped: bool
    """Inverter limit reached under the given cloud."""
    clear_clipped: bool


def hour_model(model: "PVModel", start: datetime, cloud: float, temp: float) -> HourModel:
    """Integrate the model over one hour (13 points, trapezoid) on the DC side."""
    step = timedelta(minutes=5)
    dc = clear_dc = beam = clear_beam = 0.0
    clipped = clear_clipped = False
    for k in range(13):
        w = 0.5 if k in (0, 12) else 1.0
        r = model.evaluate(start + step * k, cloud, temp)
        dc += w * r.dc_total
        clear_dc += w * r.clear_dc_total
        beam += w * r.dc_total * r.beam_fraction
        clear_beam += w * r.clear_dc_total * r.clear_beam_fraction
        clipped = clipped or model.is_clipping(r.total)
        clear_clipped = clear_clipped or model.is_clipping(r.clear_sky_total)
    return HourModel(
        dc_wh=dc / 12.0,
        clear_dc_wh=clear_dc / 12.0,
        beam_fraction=beam / dc if dc > 0 else 0.0,
        clear_beam_fraction=clear_beam / clear_dc if clear_dc > 0 else 0.0,
        clipped=clipped,
        clear_clipped=clear_clipped,
    )


@dataclass(frozen=True, slots=True)
class _Prepared:
    weight: float
    actual_dc: float
    temperature: float
    azimuth: float
    elevation: float
    now: HourModel
    cloud: float
    ahead: HourModel | None
    cloud_ahead: float | None


def _clamp(value: float, limits: tuple[float, float]) -> float:
    return min(max(value, limits[0]), limits[1])


def _cloud_curve(
    items: list[tuple[float, float, float, float, float]],
    fallback: tuple[float, ...] | None,
) -> tuple[tuple[float, ...], int, int]:
    """Fit a cloud response curve.

    ``items`` are ``(weight, actual_dc, expected_dc, cloud, temperature)``.
    Returns ``(factors, used, rejected)``. ``fallback`` (if given) is the
    curve sparse bands shrink towards instead of 1 (used to tie the
    day-ahead curve to the nowcast curve).
    """
    # Robust screening against each band's median ratio (pooled with
    # neighbours when a band has few hours).
    by_band: list[list[tuple[float, float]]] = [[] for _ in range(N_CLOUD)]
    for w, a, e, c, _t in items:
        by_band[min(int(c * N_CLOUD), N_CLOUD - 1)].append((a / e, w))
    medians: list[float | None] = []
    for i in range(N_CLOUD):
        pool = list(by_band[i])
        radius = 1
        while len(pool) < 8 and radius < N_CLOUD:
            for j in (i - radius, i + radius):
                if 0 <= j < N_CLOUD:
                    pool.extend(by_band[j])
            radius += 1
        medians.append(
            weighted_quantile([r for r, _ in pool], [w for _, w in pool], 0.5) if pool else None
        )

    # A band whose hours mostly come from one bad spell (e.g. a snowy week)
    # has a contaminated median, so gross outliers and snow are also judged
    # against the median over all bands.
    global_median = (
        weighted_quantile([a / e for _, a, e, _, _ in items], [w for w, *_ in items], 0.5)
        if items
        else None
    )

    sums_a = [0.0] * N_CLOUD
    sums_f = [0.0] * N_CLOUD
    counts = [0.0] * N_CLOUD
    used = rejected = 0
    for w, a, e, c, t in items:
        band = min(int(c * N_CLOUD), N_CLOUD - 1)
        med = medians[band]
        ratio = a / e
        reject = False
        # Only genuine anomalies are rejected. Ordinary forecast misses (a
        # clear forecast that turned cloudy, or vice versa) are exactly what
        # the conditional mean must learn, so low/high ratios alone are not
        # grounds for rejection.
        if med is not None and med > 0 and ratio / med > OUTLIER_HIGH:
            reject = True  # metering glitch
        if (
            global_median is not None
            and global_median > 0
            and t <= SNOW_TEMPERATURE
            and ratio / global_median < SNOW_RATIO
        ):
            reject = True  # snow-covered panels
        if reject:
            rejected += 1
            continue
        sums_a[band] += w * a
        sums_f[band] += w * e
        counts[band] += w
        used += 1

    total_f = sum(sums_f)
    total_n = sum(counts)
    typical = total_f / total_n if total_n > 0 else 1.0
    factors: list[float] = []
    for i in range(N_CLOUD):
        num = den = n_nb = 0.0
        for j in range(N_CLOUD):
            if j == i:
                continue
            k = math.exp(-0.5 * ((i - j) / CLOUD_KERNEL_SIGMA) ** 2)
            num += k * sums_a[j]
            den += k * sums_f[j]
            n_nb += k * counts[j]
        base = 1.0 if fallback is None else fallback[i]
        e_nb = CLOUD_PRIOR_HOURS * (den / n_nb if n_nb > 0 else typical)
        neighbour_prior = (num + e_nb * base) / (den + e_nb)
        e_own = CLOUD_PRIOR_HOURS * (sums_f[i] / counts[i] if counts[i] > 0 else typical)
        value = (sums_a[i] + e_own * neighbour_prior) / (sums_f[i] + e_own)
        factors.append(_clamp(value, CLOUD_FACTOR_LIMITS))
    return tuple(factors), used, rejected


def fit_calibration(model: "PVModel", records: Sequence[HourRecord], now: datetime) -> Calibration:
    """Fit the calibration from history records (blocking; run in executor)."""
    nameplate = model.nameplate
    if nameplate <= 0:
        return IDENTITY
    min_clear = MIN_CLEAR_FRACTION * nameplate
    rejected = 0

    prepared: list[_Prepared] = []
    for rec in records:
        if rec.actual_wh is None or not math.isfinite(rec.actual_wh) or rec.actual_wh < 0:
            continue
        mid = rec.start + timedelta(minutes=30)
        pos = solar_position(mid, model.site.latitude, model.site.longitude, model.site.pressure)
        if pos.apparent_elevation <= 0:
            continue
        hm = hour_model(model, rec.start, rec.cloud_cover, rec.temperature)
        if hm.clear_dc_wh < min_clear or hm.dc_wh <= 0 or hm.clipped:
            continue
        # Hourly mean AC power -> DC equivalent through the inverter model.
        actual_dc = inverter_dc_from_ac(model.inverter, rec.actual_wh, nameplate)
        if actual_dc is None:
            continue  # measured output at the AC limit: not informative
        ratio = actual_dc / hm.clear_dc_wh
        if not RATIO_MIN <= ratio <= RATIO_MAX:
            rejected += 1
            continue
        ahead = None
        if rec.cloud_cover_day_ahead is not None and rec.temperature_day_ahead is not None:
            ahead = hour_model(
                model, rec.start, rec.cloud_cover_day_ahead, rec.temperature_day_ahead
            )
            if ahead.dc_wh <= 0 or ahead.clipped:
                ahead = None
        age_days = max((now - rec.start).total_seconds() / 86400.0, 0.0)
        prepared.append(
            _Prepared(
                weight=0.5 ** (age_days / HALF_LIFE_DAYS),
                actual_dc=actual_dc,
                temperature=rec.temperature,
                azimuth=pos.azimuth,
                elevation=pos.apparent_elevation,
                now=hm,
                cloud=rec.cloud_cover,
                ahead=ahead,
                cloud_ahead=rec.cloud_cover_day_ahead if ahead is not None else None,
            )
        )

    # --- Stage 1: system factor and beam shading from clear hours -----------
    clear = [
        p
        for p in prepared
        if p.cloud <= CLEAR_CLOUD_MAX
        and not p.now.clear_clipped
        and not (p.temperature <= SNOW_TEMPERATURE and p.actual_dc / p.now.clear_dc_wh < 0.3)
    ]
    w_total = sum(p.weight for p in clear)
    if clear:
        q = weighted_quantile(
            [p.actual_dc / p.now.clear_dc_wh for p in clear],
            [p.weight for p in clear],
            CLEAR_QUANTILE,
        )
        global_factor = (w_total * q + GLOBAL_PRIOR_WEIGHT * 1.0) / (w_total + GLOBAL_PRIOR_WEIGHT)
    else:
        global_factor = 1.0
    global_factor = _clamp(global_factor, FACTOR_LIMITS)

    bins: dict[tuple[int, int], list[_Prepared]] = {}
    for p in clear:
        key = (int(p.azimuth // AZ_BIN) % N_AZ, min(int(p.elevation // EL_BIN), N_EL - 1))
        bins.setdefault(key, []).append(p)
    shading: dict[tuple[int, int], float] = {}
    for key, items in bins.items():
        wb = sum(p.weight for p in items)
        qb = weighted_quantile(
            [p.actual_dc / p.now.clear_dc_wh for p in items],
            [p.weight for p in items],
            CLEAR_QUANTILE,
        )
        s_bin = (wb * qb + BIN_PRIOR_WEIGHT * global_factor) / (wb + BIN_PRIOR_WEIGHT)
        beam = sum(p.weight * p.now.clear_beam_fraction for p in items) / wb
        if beam < MIN_BEAM_FOR_SHADING:
            continue
        transmission = 1.0 - (1.0 - s_bin / global_factor) / beam
        shading[key] = _clamp(transmission, SHADING_LIMITS)

    partial = Calibration(global_factor=global_factor, shading=shading)

    # --- Stage 2: cloud response (nowcast, then day-ahead) -------------------
    now_items = [
        (
            p.weight,
            p.actual_dc,
            p.now.dc_wh * partial.system_factor(p.azimuth, p.elevation, p.now.beam_fraction),
            p.cloud,
            p.temperature,
        )
        for p in prepared
    ]
    cloud_now, used_now, rej_now = _cloud_curve(now_items, None)
    ahead_items = [
        (
            p.weight,
            p.actual_dc,
            p.ahead.dc_wh * partial.system_factor(p.azimuth, p.elevation, p.ahead.beam_fraction),
            p.cloud_ahead,
            p.temperature,
        )
        for p in prepared
        if p.ahead is not None and p.cloud_ahead is not None
    ]
    cloud_ahead, used_ahead, _ = _cloud_curve(ahead_items, cloud_now)

    return Calibration(
        global_factor=global_factor,
        shading=shading,
        cloud_factors=cloud_now,
        cloud_factors_day_ahead=cloud_ahead,
        clear_hours=w_total,
        hours=used_now,
        hours_day_ahead=used_ahead,
        rejected_hours=rejected + rej_now,
        fitted_at=now,
    )
