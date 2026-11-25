"""PV forecast engine: weather series + site + arrays -> power and energy."""

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import NamedTuple

from .calibration import Calibration

from .irradiance import ZERO_IRRADIANCE, Irradiance, cloudy_sky_irradiance, ineichen_clearsky
from .linke import interpolate_linke_turbidity
from .pvsystem import ArrayConfig, InverterConfig, array_dc_power, inverter_ac_power, plane_of_array
from .solar import (
    altitude_to_pressure,
    extraterrestrial_irradiance,
    relative_airmass,
    solar_position,
)
from .weather import EnsembleWeather

STEP = timedelta(minutes=5)
_STEPS_PER_HOUR = 12


@dataclass(frozen=True, slots=True)
class Site:
    """Location of the PV system."""

    latitude: float
    longitude: float
    altitude: float
    """Metres above sea level."""
    monthly_linke_turbidity: tuple[float, ...]
    """12 monthly Linke turbidity values (Jan..Dec)."""
    linke_turbidity_scale: float = 1.0
    """Multiplier applied to the climatology (site-specific tuning)."""

    @property
    def pressure(self) -> float:
        """Standard-atmosphere surface pressure (Pa)."""
        return altitude_to_pressure(self.altitude)


@dataclass(frozen=True, slots=True)
class PowerSample:
    """Modelled AC power at an instant (W).

    ``total`` and ``arrays`` include the site calibration (if any);
    ``raw_total`` and ``clear_sky_total`` are uncalibrated model output.
    ``cloud_cover`` / ``temperature`` are the weather inputs used.
    """

    time: datetime
    total: float
    arrays: tuple[float, ...]
    clear_sky_total: float
    raw_total: float = 0.0
    cloud_cover: float = 0.0
    temperature: float = 0.0


class Evaluation(NamedTuple):
    """Result of :meth:`PVModel.evaluate`."""

    total: float
    """AC power (W) under the given cloud cover."""
    arrays: tuple[float, ...]
    clear_sky_total: float
    azimuth: float
    """Solar azimuth (deg)."""
    elevation: float
    """Apparent solar elevation (deg)."""
    dc_total: float = 0.0
    """DC power (W) under the given cloud cover, before the inverter."""
    clear_dc_total: float = 0.0
    beam_fraction: float = 0.0
    """DC-weighted share of the effective in-plane irradiance that is beam."""
    clear_beam_fraction: float = 0.0


@dataclass(slots=True)
class ForecastResult:
    """Output of :func:`compute_forecast`.

    ``samples`` are 5 minute instantaneous AC power values. ``hourly_wh`` maps
    each hour start (UTC) to total energy in Wh, ``hourly_wh_arrays`` holds the
    same per array.
    """

    samples: list[PowerSample] = field(default_factory=list)
    hourly_wh: dict[datetime, float] = field(default_factory=dict)
    hourly_wh_arrays: dict[datetime, tuple[float, ...]] = field(default_factory=dict)
    _time_index: list[datetime] | None = field(default=None, repr=False, compare=False)

    @property
    def _times(self) -> list[datetime]:
        cached = self._time_index
        if cached is None or len(cached) != len(self.samples):
            cached = [s.time for s in self.samples]
            self._time_index = cached
        return cached

    def power_at(self, when: datetime) -> PowerSample | None:
        """Linearly interpolate power at ``when`` from the 5 minute samples."""
        samples = self.samples
        if not samples or when < samples[0].time or when > samples[-1].time:
            return None
        # Samples may have gaps where weather was missing; search by time.
        idx = bisect_right(self._times, when) - 1
        s0 = samples[idx]
        if s0.time == when:
            return s0
        if idx + 1 >= len(samples):
            return None
        s1 = samples[idx + 1]
        if s1.time - s0.time != STEP:
            return None  # inside a gap
        frac = (when - s0.time) / STEP

        def lerp(a: float, b: float) -> float:
            return a + (b - a) * frac

        return PowerSample(
            time=when,
            total=lerp(s0.total, s1.total),
            arrays=tuple(lerp(a, b) for a, b in zip(s0.arrays, s1.arrays, strict=True)),
            clear_sky_total=lerp(s0.clear_sky_total, s1.clear_sky_total),
            raw_total=lerp(s0.raw_total, s1.raw_total),
            cloud_cover=lerp(s0.cloud_cover, s1.cloud_cover),
            temperature=lerp(s0.temperature, s1.temperature),
        )


def _floor_hour(when: datetime) -> datetime:
    return when.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


class PVModel:
    """Evaluate the PV system at single instants."""

    def __init__(self, site: Site, arrays: list[ArrayConfig], inverter: InverterConfig) -> None:
        if not arrays:
            raise ValueError("at least one array is required")
        self.site = site
        self.arrays = arrays
        self.inverter = inverter
        self._nameplate = sum(a.dc_power_w for a in arrays)

    def _linke(self, when: datetime) -> float:
        utc = when.astimezone(timezone.utc)
        frac = (utc.hour * 3600 + utc.minute * 60 + utc.second) / 86400.0
        tl = interpolate_linke_turbidity(list(self.site.monthly_linke_turbidity), utc.date(), frac)
        return tl * self.site.linke_turbidity_scale

    @property
    def nameplate(self) -> float:
        """Total DC nameplate power (W)."""
        return self._nameplate

    def evaluate(self, when: datetime, cloud_cover: float, temperature: float) -> Evaluation:
        """Evaluate uncalibrated AC power at ``when``."""
        n = len(self.arrays)
        pressure = self.site.pressure
        pos = solar_position(when, self.site.latitude, self.site.longitude, pressure, temperature)
        elevation = pos.apparent_elevation
        if pos.apparent_zenith >= 90.0:
            return Evaluation(0.0, (0.0,) * n, 0.0, pos.azimuth, elevation)

        dni_extra = extraterrestrial_irradiance(when)
        am_rel = relative_airmass(pos.apparent_zenith)
        am_abs = am_rel * pressure / 101325.0 if am_rel is not None else None

        clear = ineichen_clearsky(
            pos.apparent_zenith, am_abs, self._linke(when), self.site.altitude, dni_extra
        )
        if clear.ghi <= 0.0:
            return Evaluation(0.0, (0.0,) * n, 0.0, pos.azimuth, elevation)
        doy = when.astimezone(timezone.utc).timetuple().tm_yday
        cloudy = cloudy_sky_irradiance(
            clear, cloud_cover, pos.apparent_zenith, doy, pressure, true_zenith=pos.zenith
        )

        def system_ac(irr: Irradiance) -> tuple[float, tuple[float, ...], float, float]:
            """(AC total, AC per array, DC total, beam fraction)."""
            if irr is ZERO_IRRADIANCE or irr.ghi <= 0.0:
                return 0.0, (0.0,) * n, 0.0, 0.0
            dcs = []
            beam_weighted = 0.0
            for array in self.arrays:
                poa = plane_of_array(
                    array, irr, dni_extra, pos.apparent_zenith, pos.azimuth, am_rel
                )
                dc = array_dc_power(array, poa, temperature)
                dcs.append(dc)
                if poa.effective > 0.0:
                    beam_weighted += dc * poa.beam_effective / poa.effective
            dc_total = sum(dcs)
            ac_total = inverter_ac_power(self.inverter, dc_total, self._nameplate)
            if dc_total <= 0.0:
                return 0.0, (0.0,) * n, 0.0, 0.0
            # Inverter losses and clipping are shared pro rata by DC share.
            return (
                ac_total,
                tuple(ac_total * dc / dc_total for dc in dcs),
                dc_total,
                beam_weighted / dc_total,
            )

        total, per_array, dc_total, beam = system_ac(cloudy)
        clear_total, _, clear_dc, clear_beam = system_ac(clear)
        return Evaluation(
            total,
            per_array,
            clear_total,
            pos.azimuth,
            elevation,
            dc_total,
            clear_dc,
            beam,
            clear_beam,
        )

    def is_clipping(self, ac_power: float) -> bool:
        """True if ``ac_power`` is at (or within 2 % of) the inverter limit."""
        limit = self.inverter.ac_power_w
        return bool(limit) and ac_power >= 0.98 * limit

    def apply_calibration(
        self,
        ev: Evaluation,
        cloud_cover: float,
        calibration: Calibration | None,
        lead_hours: float = 0.0,
    ) -> tuple[float, tuple[float, ...]]:
        """Apply the site calibration to an evaluation.

        The factor scales the DC power, which is then passed through the
        inverter model again, so clipping and part-load efficiency are
        respected. The calibrated output is capped at the calibrated
        clear-sky output (plus 10 %). Per-array values keep their share.
        """
        if calibration is None or ev.total <= 0.0 or ev.dc_total <= 0.0:
            return ev.total, ev.arrays
        factor = calibration.factor(
            ev.azimuth, ev.elevation, ev.beam_fraction, cloud_cover, lead_hours
        )
        dc = ev.dc_total * factor
        # Cloud corrections above 1 may not lift output beyond the calibrated
        # clear-sky output (+10 % for cloud enhancement). The cap never goes
        # below the system/shading-corrected value itself.
        system_only = ev.dc_total * calibration.system_factor(
            ev.azimuth, ev.elevation, ev.beam_fraction
        )
        clear_cap = (
            1.1
            * ev.clear_dc_total
            * calibration.system_factor(ev.azimuth, ev.elevation, ev.clear_beam_fraction)
        )
        dc = min(dc, max(clear_cap, system_only))
        total = inverter_ac_power(self.inverter, dc, self._nameplate)
        scale = total / ev.total
        return total, tuple(a * scale for a in ev.arrays)


def compute_forecast(
    model: PVModel,
    weather: EnsembleWeather,
    start: datetime,
    end: datetime,
    hold_before: datetime | None = None,
    calibration: Calibration | None = None,
    issued: datetime | None = None,
) -> ForecastResult:
    """Compute power samples and hourly energy between ``start`` and ``end``.

    ``start`` is floored to the hour. Times before the first weather sample
    are only modelled if they are at or after ``hold_before``; for those the
    first available forecast value is held (used to fill the already elapsed
    part of today on a fresh start). An hour is only emitted in
    ``hourly_wh`` when weather is available for all of its 13 sample points.
    ``calibration`` (optional) is applied to ``total`` / ``arrays``; the
    forecast lead time used to pick its cloud response is measured from
    ``issued`` (defaults to ``start``).
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone aware")
    t = _floor_hour(start)
    end = end.astimezone(timezone.utc)
    result = ForecastResult()
    n = len(model.arrays)
    issued_utc = (issued or start).astimezone(timezone.utc)

    while t <= end:
        hold = hold_before is not None and t >= hold_before
        wx = weather.at(t, hold_before_start=hold)
        if wx is not None:
            ev = model.evaluate(t, wx.cloud_cover, wx.temperature)
            lead = (t - issued_utc) / timedelta(hours=1)
            total, arrays = model.apply_calibration(ev, wx.cloud_cover, calibration, lead)
            result.samples.append(
                PowerSample(
                    time=t,
                    total=total,
                    arrays=arrays,
                    clear_sky_total=ev.clear_sky_total,
                    raw_total=ev.total,
                    cloud_cover=wx.cloud_cover,
                    temperature=wx.temperature,
                )
            )
        t += STEP

    # Integrate each hour with the trapezoidal rule over 12 x 5 min steps.
    by_time = {s.time: s for s in result.samples}
    if result.samples:
        hour = _floor_hour(result.samples[0].time)
        last = result.samples[-1].time
        while hour + timedelta(hours=1) <= last:
            pts = [by_time.get(hour + STEP * k) for k in range(_STEPS_PER_HOUR + 1)]
            if all(p is not None for p in pts):
                weights = [0.5] + [1.0] * (_STEPS_PER_HOUR - 1) + [0.5]
                step_h = STEP / timedelta(hours=1)
                wh = sum(w * p.total for w, p in zip(weights, pts, strict=True)) * step_h  # type: ignore[union-attr]
                wh_arr = tuple(
                    sum(w * p.arrays[i] for w, p in zip(weights, pts, strict=True)) * step_h  # type: ignore[union-attr]
                    for i in range(n)
                )
                if math.isfinite(wh):
                    result.hourly_wh[hour] = wh
                    result.hourly_wh_arrays[hour] = wh_arr
            hour += timedelta(hours=1)
    return result
