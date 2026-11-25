"""Data update coordinator for Solar Forecast."""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ELEVATION, CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    CONF_ALBEDO,
    CONF_AZIMUTH,
    CONF_DC_POWER,
    CONF_INVERTER_AC_POWER,
    CONF_INVERTER_EFFICIENCY,
    CONF_LOSSES,
    CONF_MODULE_EFFICIENCY,
    CONF_MOUNTING,
    CONF_PRODUCTION_SENSOR,
    CONF_TEMP_COEFFICIENT,
    CONF_TILT,
    CONF_TURBIDITY_SCALE,
    CONF_WEATHER_ENTITIES,
    DEFAULT_ALBEDO,
    DEFAULT_INVERTER_EFFICIENCY,
    DEFAULT_LOSSES,
    DEFAULT_MODULE_EFFICIENCY,
    DEFAULT_MOUNTING,
    DEFAULT_TEMP_COEFFICIENT,
    DEFAULT_TURBIDITY_SCALE,
    DOMAIN,
    FORECAST_DAYS,
    HISTORY_DAYS,
    STORAGE_VERSION,
    SUBENTRY_TYPE_ARRAY,
    UPDATE_INTERVAL,
)
from .calibration_manager import CalibrationManager
from .model.forecast import ForecastResult, PowerSample, PVModel, Site, compute_forecast
from .model.linke import load_monthly_linke_turbidity
from .model.pvsystem import ArrayConfig, InverterConfig
from .model.weather import EnsembleWeather, WeatherAt
from .weather_input import SourceReport, async_fetch_weather

_LOGGER = logging.getLogger(__name__)

type SolarForecastConfigEntry = ConfigEntry[SolarForecastCoordinator]


def build_array_config(name: str, data: dict[str, Any] | Any) -> ArrayConfig:
    """Translate stored subentry data (user units) into a model ArrayConfig."""
    return ArrayConfig(
        name=name,
        tilt=float(data[CONF_TILT]),
        azimuth=float(data[CONF_AZIMUTH]) % 360.0,
        dc_power_w=float(data[CONF_DC_POWER]),
        temp_coefficient=float(data.get(CONF_TEMP_COEFFICIENT, DEFAULT_TEMP_COEFFICIENT)) / 100.0,
        mounting=str(data.get(CONF_MOUNTING, DEFAULT_MOUNTING)),
        losses=float(data.get(CONF_LOSSES, DEFAULT_LOSSES)) / 100.0,
        albedo=float(data.get(CONF_ALBEDO, DEFAULT_ALBEDO)),
        module_efficiency=float(data.get(CONF_MODULE_EFFICIENCY, DEFAULT_MODULE_EFFICIENCY))
        / 100.0,
    )


@dataclass(slots=True)
class SolarForecastData:
    """Forecast state exposed to entities.

    ``hourly_wh`` / ``hourly_wh_arrays`` merge frozen history (hours that have
    already ended) with the live forecast; keys are UTC hour starts.
    """

    array_ids: list[str]
    hourly_wh: dict[datetime, float]
    hourly_wh_arrays: dict[str, dict[datetime, float]]
    forecast: ForecastResult
    weather_now: WeatherAt | None
    sources: list[SourceReport] = field(default_factory=list)
    updated: datetime = field(default_factory=dt_util.utcnow)
    calibration: dict[str, Any] | None = None
    """Calibration summary (None when no production sensor is configured)."""
    accuracy: dict[str, Any] | None = None
    """Forecast error versus measured production (None when disabled)."""

    def _series(self, array_id: str | None) -> dict[datetime, float]:
        if array_id is None:
            return self.hourly_wh
        return self.hourly_wh_arrays.get(array_id, {})

    def energy_between(
        self, start: datetime, end: datetime, array_id: str | None = None
    ) -> float | None:
        """Energy (Wh) between two instants; None if any hour is missing.

        Whole UTC hours use the hourly values directly. A partially covered
        hour (non-aligned bounds, e.g. "now" or local midnight in a UTC+5:30
        zone) contributes its hourly value times the fraction of that hour's
        energy falling inside the window, taken from the 5 minute power curve
        (or the time fraction when the curve is unavailable or zero). This
        keeps partial values consistent with frozen hourly history.
        """
        series = self._series(array_id)
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        if end <= start:
            return 0.0
        one_hour = timedelta(hours=1)
        total = 0.0
        hour = start.replace(minute=0, second=0, microsecond=0)
        while hour < end:
            value = series.get(hour)
            if value is None:
                return None
            a = max(start, hour)
            b = min(end, hour + one_hour)
            if a == hour and b == hour + one_hour:
                total += value
            else:
                total += value * self._hour_fraction(hour, a, b, array_id)
            hour += one_hour
        return total

    def _hour_fraction(
        self, hour: datetime, a: datetime, b: datetime, array_id: str | None
    ) -> float:
        """Fraction of the energy of ``hour`` that falls within [a, b)."""
        whole = self.energy_in_window(hour, hour + timedelta(hours=1), array_id)
        part = self.energy_in_window(a, b, array_id)
        if whole is not None and part is not None and whole > 0.0:
            return min(max(part / whole, 0.0), 1.0)
        return (b - a) / timedelta(hours=1)

    def energy_day(self, day: date, array_id: str | None = None) -> float | None:
        """Total energy (Wh) for a local calendar day; None if incomplete."""
        start = dt_util.start_of_local_day(day)
        end = dt_util.start_of_local_day(day + timedelta(days=1))
        return self.energy_between(start, end, array_id)

    def energy_remaining_today(self, now: datetime, array_id: str | None = None) -> float | None:
        """Energy (Wh) from ``now`` until local midnight."""
        tomorrow = dt_util.start_of_local_day(dt_util.as_local(now).date() + timedelta(days=1))
        return self.energy_between(now, tomorrow, array_id)

    def energy_in_window(
        self, start: datetime, end: datetime, array_id: str | None = None
    ) -> float | None:
        """Integrate the power curve between two instants (Wh)."""
        if end <= start:
            return 0.0
        idx = None if array_id is None else self._array_index(array_id)
        if array_id is not None and idx is None:
            return None
        points: list[datetime] = [start]
        t = start.astimezone(timezone.utc)
        step = timedelta(minutes=5)
        # Align to the 5 minute grid.
        aligned = t.replace(second=0, microsecond=0) - timedelta(minutes=t.minute % 5)
        if aligned <= t:
            aligned += step
        while aligned < end:
            points.append(aligned)
            aligned += step
        points.append(end)
        values: list[float] = []
        for p in points:
            sample = self.forecast.power_at(p)
            if sample is None:
                return None
            values.append(sample.total if idx is None else sample.arrays[idx])
        energy = 0.0
        for i in range(len(points) - 1):
            dt_h = (points[i + 1] - points[i]) / timedelta(hours=1)
            energy += 0.5 * (values[i] + values[i + 1]) * dt_h
        return energy

    def _array_index(self, array_id: str) -> int | None:
        try:
            return self.array_ids.index(array_id)
        except ValueError:
            return None

    def power_now(self, now: datetime, array_id: str | None = None) -> float | None:
        """Forecast AC power (W) at ``now``."""
        sample = self.forecast.power_at(now)
        if sample is None:
            return None
        if array_id is None:
            return sample.total
        idx = self._array_index(array_id)
        return None if idx is None else sample.arrays[idx]

    def clear_sky_power_now(self, now: datetime) -> float | None:
        """Clear-sky AC power (W) at ``now``."""
        sample = self.forecast.power_at(now)
        return None if sample is None else sample.clear_sky_total

    def peak(self, day: date) -> PowerSample | None:
        """Highest forecast power sample during a local day."""
        start = dt_util.start_of_local_day(day)
        end = dt_util.start_of_local_day(day + timedelta(days=1))
        best: PowerSample | None = None
        covered = False
        for sample in self.forecast.samples:
            if start <= sample.time < end:
                covered = True
                if best is None or sample.total > best.total:
                    best = sample
        return best if covered else None

    def hourly_list(self, day: date, array_id: str | None = None) -> list[dict[str, Any]]:
        """Hourly forecast for a local day, one entry per local clock hour.

        Hours are stepped in UTC from local midnight, so DST days have 23 or
        25 entries; in UTC+x:30 zones the local hours straddle UTC hours and
        are apportioned like the daily total, so the entries sum to it.
        """
        start = dt_util.start_of_local_day(day).astimezone(timezone.utc)
        end = dt_util.start_of_local_day(day + timedelta(days=1)).astimezone(timezone.utc)
        out: list[dict[str, Any]] = []
        hour = start
        while hour < end:
            nxt = min(hour + timedelta(hours=1), end)
            value = self.energy_between(hour, nxt, array_id)
            if value is not None:
                out.append(
                    {
                        "period_start": dt_util.as_local(hour).isoformat(),
                        "energy_wh": round(value, 1),
                    }
                )
            hour = nxt
        return out


class SolarForecastCoordinator(DataUpdateCoordinator[SolarForecastData]):
    """Fetch weather forecasts and run the PV model."""

    config_entry: SolarForecastConfigEntry

    def __init__(self, hass: HomeAssistant, entry: SolarForecastConfigEntry) -> None:
        """Initialise the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=UPDATE_INTERVAL,
        )
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}.history"
        )
        # hour start (UTC) -> {array subentry id: Wh}. Site total is the sum
        # (AC is split between arrays pro rata, so the sum is exact).
        self._history: dict[datetime, dict[str, float]] = {}
        self._array_ids: list[str] = []
        self._model: PVModel | None = None
        self.calibration = CalibrationManager(
            hass, entry.entry_id, entry.options.get(CONF_PRODUCTION_SENSOR) or None
        )
        self._live_since: datetime | None = None

    @property
    def model(self) -> PVModel | None:
        """The PV model (available after setup)."""
        return self._model

    @property
    def array_ids(self) -> list[str]:
        """Subentry ids of the configured arrays, in model order."""
        return self._array_ids

    async def _async_setup(self) -> None:
        """Build the model and load stored history (runs once before first refresh)."""
        entry = self.config_entry
        data = entry.data
        latitude = float(data[CONF_LATITUDE])
        longitude = float(data[CONF_LONGITUDE])
        monthly_tl = await self.hass.async_add_executor_job(
            load_monthly_linke_turbidity, latitude, longitude
        )
        site = Site(
            latitude=latitude,
            longitude=longitude,
            altitude=float(data.get(CONF_ELEVATION, 0.0)),
            monthly_linke_turbidity=tuple(monthly_tl),
            linke_turbidity_scale=float(
                entry.options.get(CONF_TURBIDITY_SCALE, DEFAULT_TURBIDITY_SCALE)
            ),
        )
        arrays: list[ArrayConfig] = []
        self._array_ids = []
        for subentry_id, subentry in entry.subentries.items():
            if subentry.subentry_type != SUBENTRY_TYPE_ARRAY:
                continue
            arrays.append(build_array_config(subentry.title, subentry.data))
            self._array_ids.append(subentry_id)
        if not arrays:
            # async_setup_entry refuses to set up without arrays.
            raise UpdateFailed("No PV arrays configured")
        ac_power = float(data.get(CONF_INVERTER_AC_POWER) or 0.0)
        inverter = InverterConfig(
            ac_power_w=ac_power if ac_power > 0 else None,
            efficiency_nominal=float(
                data.get(CONF_INVERTER_EFFICIENCY, DEFAULT_INVERTER_EFFICIENCY)
            )
            / 100.0,
        )
        self._model = PVModel(site, arrays, inverter)
        await self._async_load_history()
        await self.calibration.async_load()

    async def _async_load_history(self) -> None:
        stored = await self._store.async_load()
        self._history = {}
        if not stored:
            return
        wanted = set(self._array_ids)
        for key, values in (stored.get("hours") or {}).items():
            when = dt_util.parse_datetime(key)
            if when is None or not isinstance(values, dict):
                continue
            # Keep an hour only if every current array has a value for it, so
            # adding an array doesn't mix arrays with and without history.
            if not wanted.issubset(values):
                continue
            try:
                self._history[when.astimezone(timezone.utc)] = {
                    aid: float(values[aid]) for aid in self._array_ids
                }
            except TypeError, ValueError:
                continue

    def _history_data(self) -> dict[str, Any]:
        return {
            "hours": {k.isoformat(): dict(v) for k, v in sorted(self._history.items())},
        }

    async def _async_update_data(self) -> SolarForecastData:
        """Fetch weather and recompute the forecast."""
        assert self._model is not None
        entity_ids: list[str] = list(self.config_entry.data.get(CONF_WEATHER_ENTITIES, []))
        series, reports = await async_fetch_weather(self.hass, entity_ids)
        if not series:
            raise UpdateFailed(
                translation_domain=DOMAIN,
                translation_key="no_weather_data",
                translation_placeholders={
                    "details": "; ".join(
                        f"{r.entity_id}: {', '.join(r.errors) or 'no data'}" for r in reports
                    )
                },
            )
        weather = EnsembleWeather(series)
        now = dt_util.utcnow()
        today = dt_util.as_local(now).date()
        start_today = dt_util.start_of_local_day(today)
        horizon = dt_util.start_of_local_day(today + timedelta(days=FORECAST_DAYS))
        end = min(horizon, weather.end) if weather.end else horizon

        model = self._model
        cal = self.calibration
        await cal.async_update_actuals(now)
        await cal.async_maybe_fit(model, now)
        result: ForecastResult = await self.hass.async_add_executor_job(
            compute_forecast,
            model,
            weather,
            start_today,
            end,
            start_today,
            cal.calibration if cal.enabled else None,
            now,
        )
        result = self._keep_elapsed_samples(result, now, start_today)
        data = self._merge(result, now)
        data.weather_now = weather.at(now, hold_before_start=True)
        data.sources = reports

        if self._live_since is None:
            self._live_since = now
        cal.record_elapsed_hours(result, now, self._live_since)
        tomorrow = today + timedelta(days=1)
        cal.record_day_ahead(result, data.energy_day(tomorrow), today)
        if cal.enabled:
            data.calibration = cal.summary()
            data.accuracy = cal.accuracy(data.hourly_wh, today)
        cal.schedule_save()
        self._store.async_delay_save(self._history_data, 30)
        return data

    def _keep_elapsed_samples(
        self, result: ForecastResult, now: datetime, start_today: datetime
    ) -> ForecastResult:
        """Keep the previous power curve for the elapsed part of today.

        Weather providers drop past hours, so the fresh result fills the
        elapsed part of today by holding the current first forecast value.
        Reuse the samples computed earlier instead, so that the peak of
        today and partial-hour energy stay as forecast at the time.
        """
        previous = self.data.forecast.samples if self.data is not None else []
        if not previous:
            return result
        split = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 5)
        # Keep the last hour of yesterday too, so that hour can be recorded
        # for calibration after midnight (matters at midnight-sun sites).
        kept = [s for s in previous if start_today - timedelta(hours=1) <= s.time < split]
        if not kept:
            return result
        fresh = [s for s in result.samples if s.time >= split]
        return ForecastResult(
            samples=kept + fresh,
            hourly_wh=result.hourly_wh,
            hourly_wh_arrays=result.hourly_wh_arrays,
        )

    def _merge(self, result: ForecastResult, now: datetime) -> SolarForecastData:
        """Combine fresh forecast with frozen history of elapsed hours."""
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        ids = self._array_ids

        # Future hours always reflect the latest forecast only; drop any that
        # the new forecast no longer covers (e.g. a shorter weather horizon).
        for hour in [h for h in self._history if h >= current_hour]:
            if hour not in result.hourly_wh:
                del self._history[hour]
        for hour, per_array in result.hourly_wh_arrays.items():
            if hour < current_hour and hour in self._history:
                continue  # elapsed hour: keep the forecast as it stood
            self._history[hour] = dict(zip(ids, per_array, strict=True))

        cutoff = current_hour - timedelta(days=HISTORY_DAYS)
        self._history = {h: v for h, v in self._history.items() if h >= cutoff}

        hourly: dict[datetime, float] = {}
        per_array_series: dict[str, dict[datetime, float]] = {aid: {} for aid in ids}
        for hour, values in sorted(self._history.items()):
            hourly[hour] = sum(values[aid] for aid in ids)
            for aid in ids:
                per_array_series[aid][hour] = values[aid]

        return SolarForecastData(
            array_ids=list(ids),
            hourly_wh=hourly,
            hourly_wh_arrays=per_array_series,
            forecast=result,
            weather_now=None,
        )

    async def async_shutdown(self) -> None:
        """Persist history on unload."""
        await super().async_shutdown()
        if self._history:
            await self._store.async_save(self._history_data())
        await self.calibration.async_save()
