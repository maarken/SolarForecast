"""Collect training data, fetch measured production and fit the calibration."""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import partial
import logging
import math
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DOMAIN, STORAGE_VERSION
from .model.calibration import IDENTITY, Calibration, HourRecord, fit_calibration
from .model.forecast import STEP, ForecastResult, PVModel

_LOGGER = logging.getLogger(__name__)

RECORD_DAYS = 90
"""Training records kept (hourly)."""
REFRESH_ACTUALS_DAYS = 3
"""Missing actuals are retried for this long (statistics can arrive late)."""
STATS_SETTLE = timedelta(minutes=15)
"""Wait this long after an hour ends before reading its statistics."""
QUERY_INTERVAL = timedelta(hours=1)
"""Minimum time between statistics queries unless new hours need data."""
QUERY_TIMEOUT = 60.0
"""Seconds before a recorder query is abandoned (forecast continues)."""
REFIT_INTERVAL = timedelta(hours=1)
ACCURACY_DAYS = 7
HOUR = timedelta(hours=1)

_ENERGY_UNITS = {"Wh", "kWh", "MWh", "GWh"}
_POWER_UNITS = {"W", "kW", "MW", "GW"}


@dataclass(slots=True)
class _Record:
    cloud: float
    temperature: float
    actual: float | None
    cloud_ahead: float | None = None
    temperature_ahead: float | None = None


def _hour_weather(by_time: dict[datetime, Any], hour: datetime) -> tuple[float, float] | None:
    """Trapezoid-weighted hourly mean (cloud, temperature) from 5 min samples."""
    pts = [by_time.get(hour + STEP * k) for k in range(13)]
    if any(p is None for p in pts):
        return None
    weights = [0.5] + [1.0] * 11 + [0.5]
    cloud = sum(w * p.cloud_cover for w, p in zip(weights, pts, strict=True)) / 12
    temp = sum(w * p.temperature for w, p in zip(weights, pts, strict=True)) / 12
    return cloud, temp


class CalibrationManager:
    """Owns calibration records, measured production and the fitted factors."""

    def __init__(self, hass: HomeAssistant, entry_id: str, sensor_id: str | None) -> None:
        """Initialise; call :meth:`async_load` before use."""
        self.hass = hass
        self.sensor_id = sensor_id
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, f"{DOMAIN}.{entry_id}.calibration"
        )
        self._records: dict[datetime, _Record] = {}
        # Hour (UTC) -> (cloud, temperature) forecast for that hour as it
        # stood at the last update of the previous local day.
        self._pending_ahead: dict[datetime, tuple[float, float]] = {}
        # Local date -> energy forecast for that day as it stood at the end of
        # the previous day (day-ahead), Wh.
        self._day_ahead: dict[date, float] = {}
        self._full_refetch = False
        self._dirty_actuals = False
        self._last_query: datetime | None = None
        self._last_query_missing: frozenset[datetime] = frozenset()
        self.calibration: Calibration = IDENTITY
        self.last_fit: datetime | None = None
        self.last_error: str | None = None
        self.statistic_kind: str | None = None

    @property
    def enabled(self) -> bool:
        """True when a production sensor is configured."""
        return bool(self.sensor_id)

    # ------------------------------------------------------------------ storage

    async def async_load(self) -> None:
        """Load stored records."""
        stored = await self._store.async_load() or {}
        same_sensor = stored.get("sensor") == self.sensor_id
        for key, values in (stored.get("hours") or {}).items():
            when = dt_util.parse_datetime(key)
            if when is None or not isinstance(values, list) or len(values) not in (3, 5):
                continue
            try:
                floats = [None if v is None else float(v) for v in values]
            except TypeError, ValueError:
                continue
            cloud, temp = floats[0], floats[1]
            if cloud is None or temp is None or not (math.isfinite(cloud) and math.isfinite(temp)):
                continue
            # Measurements from a different sensor are not reused.
            actual = floats[2] if same_sensor else None
            cloud_ahead = floats[3] if len(floats) == 5 else None
            temp_ahead = floats[4] if len(floats) == 5 else None
            self._records[when.astimezone(timezone.utc)] = _Record(
                cloud, temp, actual, cloud_ahead, temp_ahead
            )
        for key, values in (stored.get("pending_ahead") or {}).items():
            when = dt_util.parse_datetime(key)
            try:
                if when is not None and isinstance(values, list) and len(values) == 2:
                    self._pending_ahead[when.astimezone(timezone.utc)] = (
                        float(values[0]),
                        float(values[1]),
                    )
            except TypeError, ValueError:
                continue
        for key, value in (stored.get("day_ahead") or {}).items():
            try:
                self._day_ahead[date.fromisoformat(key)] = float(value)
            except TypeError, ValueError:
                continue
        if not same_sensor:
            self._full_refetch = True
            self._dirty_actuals = True

    def _data(self) -> dict[str, Any]:
        def r4(v: float | None) -> float | None:
            return None if v is None else round(v, 4)

        return {
            "sensor": self.sensor_id,
            "hours": {
                k.isoformat(): [
                    round(r.cloud, 4),
                    round(r.temperature, 2),
                    r.actual,
                    r4(r.cloud_ahead),
                    r4(r.temperature_ahead),
                ]
                for k, r in sorted(self._records.items())
            },
            "pending_ahead": {
                k.isoformat(): [round(c, 4), round(t, 2)]
                for k, (c, t) in sorted(self._pending_ahead.items())
            },
            "day_ahead": {k.isoformat(): round(v, 1) for k, v in sorted(self._day_ahead.items())},
        }

    def schedule_save(self) -> None:
        """Save soon (debounced)."""
        self._store.async_delay_save(self._data, 60)

    async def async_save(self) -> None:
        """Save now."""
        await self._store.async_save(self._data())

    # ------------------------------------------------------------- recording

    def record_elapsed_hours(
        self, result: ForecastResult, now: datetime, live_since: datetime
    ) -> None:
        """Store weather inputs of hours that have just ended.

        The samples for elapsed time are the ones computed while that time
        was current (see the coordinator), so this is the forecast as it
        stood. Only hours starting at or after ``live_since`` (the first
        update of this run) qualify: before that, elapsed samples were filled
        by holding the first forecast value and do not reflect the weather
        forecast for that hour. Night hours are skipped; an hour is recorded
        once. The day-ahead weather captured for the hour (if any) is
        attached.
        """
        by_time = {s.time: s for s in result.samples}
        if by_time:
            first = max(min(by_time), live_since)
            hour = first.replace(minute=0, second=0, microsecond=0)
            if hour < first:
                hour += HOUR
            while hour + HOUR <= now:
                if hour not in self._records:
                    pts = [by_time.get(hour + STEP * k) for k in range(13)]
                    weather = _hour_weather(by_time, hour)
                    if weather is not None and any(
                        p.clear_sky_total > 0
                        for p in pts  # type: ignore[union-attr]
                    ):
                        ahead = self._pending_ahead.get(hour)
                        self._records[hour] = _Record(
                            weather[0],
                            weather[1],
                            None,
                            ahead[0] if ahead else None,
                            ahead[1] if ahead else None,
                        )
                hour += HOUR
        cutoff = now - timedelta(days=RECORD_DAYS)
        self._records = {h: r for h, r in self._records.items() if h >= cutoff}
        self._pending_ahead = {
            h: v for h, v in self._pending_ahead.items() if h >= now - timedelta(days=2)
        }

    def record_day_ahead(
        self, result: ForecastResult, tomorrow_energy: float | None, today: date
    ) -> None:
        """Capture tomorrow's forecast as it stands (the last update of the day wins).

        Stores tomorrow's total energy (for the day-ahead accuracy metric)
        and tomorrow's hourly weather (for the day-ahead cloud response).
        """
        tomorrow = today + timedelta(days=1)
        if tomorrow_energy is not None:
            self._day_ahead[tomorrow] = tomorrow_energy
        start = dt_util.start_of_local_day(tomorrow).astimezone(timezone.utc)
        end = dt_util.start_of_local_day(tomorrow + timedelta(days=1)).astimezone(timezone.utc)
        by_time = {s.time: s for s in result.samples if start <= s.time <= end}
        hour = start
        while hour < end:
            weather = _hour_weather(by_time, hour)
            if weather is not None:
                self._pending_ahead[hour] = weather
            hour += HOUR
        cutoff = today - timedelta(days=RECORD_DAYS)
        self._day_ahead = {d: v for d, v in self._day_ahead.items() if d >= cutoff}

    # ----------------------------------------------------------- measurements

    async def async_update_actuals(self, now: datetime) -> None:
        """Fill in measured energy for recorded hours from recorder statistics."""
        if not self.enabled:
            return
        settled = now - STATS_SETTLE
        oldest_retry = now - timedelta(days=REFRESH_ACTUALS_DAYS)
        missing = frozenset(
            h
            for h, r in self._records.items()
            if r.actual is None
            and h + HOUR <= settled
            and (self._full_refetch or h >= oldest_retry)
        )
        if not missing:
            self._full_refetch = False
            return
        # Throttle: query when new hours need data, otherwise at most hourly.
        if (
            self._last_query is not None
            and missing <= self._last_query_missing
            and now - self._last_query < QUERY_INTERVAL
        ):
            return
        self._last_query = now
        self._last_query_missing = missing
        try:
            async with asyncio.timeout(QUERY_TIMEOUT):
                values = await self._async_query_statistics(min(missing), settled)
        except Exception as err:  # noqa: BLE001 - never break the forecast
            self.last_error = f"{type(err).__name__}: {err}"
            _LOGGER.debug("Fetching production statistics failed: %s", err)
            return
        self._full_refetch = False
        for hour in missing:
            value = values.get(hour)
            if value is not None:
                self._records[hour].actual = value
                self._dirty_actuals = True

    async def _async_query_statistics(
        self, start: datetime, end: datetime
    ) -> dict[datetime, float]:
        """Hourly measured energy (Wh) keyed by UTC hour start.

        An hour is only returned if the statistics row of the preceding hour
        exists too. For energy sensors the hourly ``change`` is the
        difference to the previous row, so after a gap (sensor unavailable)
        the first row would carry the energy of the whole gap; for power
        sensors the first row after a gap is a mean over part of the hour.
        """
        if "recorder" not in self.hass.config.components:
            raise RuntimeError("recorder is not available")
        # Imported lazily so the integration still loads without recorder.
        from homeassistant.components.recorder import get_instance  # noqa: PLC0415
        from homeassistant.components.recorder.models import StatisticMeanType  # noqa: PLC0415
        from homeassistant.components.recorder.statistics import (  # noqa: PLC0415
            get_metadata,
            statistics_during_period,
        )

        sensor_id = self.sensor_id
        assert sensor_id is not None
        instance = get_instance(self.hass)
        metadata = await instance.async_add_executor_job(
            partial(get_metadata, self.hass, statistic_ids={sensor_id})
        )
        if sensor_id not in metadata:
            raise RuntimeError(f"{sensor_id} has no long-term statistics (it needs a state_class)")
        meta = metadata[sensor_id][1]
        unit = meta.get("unit_of_measurement")
        unit_class = meta.get("unit_class")
        if meta.get("has_sum"):
            kind = "change"
            if unit_class != "energy" and unit not in _ENERGY_UNITS:
                raise RuntimeError(f"unsupported energy unit {unit!r}")
            units = {"energy": "Wh"}
        elif meta.get("mean_type", StatisticMeanType.NONE) == StatisticMeanType.ARITHMETIC:
            kind = "mean"  # hourly mean power in W equals energy in Wh
            if unit_class != "power" and unit not in _POWER_UNITS:
                raise RuntimeError(f"unsupported power unit {unit!r}")
            units = {"power": "W"}
        else:
            raise RuntimeError(f"{sensor_id} statistics have neither sum nor mean")
        self.statistic_kind = kind

        stats = await instance.async_add_executor_job(
            statistics_during_period,
            self.hass,
            start - HOUR,  # include the preceding row for the gap check
            end,
            {sensor_id},
            "hour",
            units,
            {kind},
        )
        rows: dict[datetime, float] = {}
        for row in stats.get(sensor_id, []):
            value = row.get(kind)
            ts = row.get("start")
            if value is None or ts is None:
                continue
            value = float(value)
            if math.isfinite(value):
                rows[datetime.fromtimestamp(ts, tz=timezone.utc)] = value

        result: dict[datetime, float] = {}
        negative = nonzero = 0
        for hour, energy in rows.items():
            if hour < start or (hour - HOUR) not in rows:
                continue
            if energy != 0:
                nonzero += 1
            if energy < 0:
                negative += 1
                continue  # meter reset, export sensor or bad data
            result[hour] = energy
        if nonzero and negative > nonzero / 2:
            self.last_error = (
                f"{sensor_id} reports mostly negative production values; "
                "select a sensor where production is positive"
            )
        else:
            self.last_error = None
        return result

    # ----------------------------------------------------------------- fitting

    async def async_maybe_fit(self, model: PVModel, now: datetime) -> None:
        """Refit when new measurements arrived (at most once per interval)."""
        if not self.enabled:
            self.calibration = IDENTITY
            return
        due = self.last_fit is None or now - self.last_fit >= REFIT_INTERVAL
        if not (due and (self._dirty_actuals or self.last_fit is None)):
            return
        records = [
            HourRecord(h, r.cloud, r.temperature, r.actual, r.cloud_ahead, r.temperature_ahead)
            for h, r in sorted(self._records.items())
            if r.actual is not None
        ]
        self.calibration = await self.hass.async_add_executor_job(
            fit_calibration, model, records, now
        )
        self.last_fit = now
        self._dirty_actuals = False

    # ---------------------------------------------------------------- accuracy

    def accuracy(self, frozen_hourly: dict[datetime, float], today: date) -> dict[str, Any] | None:
        """Forecast error over the last complete days.

        ``day_ahead``: forecast made the previous evening vs measured.
        ``same_day``: sum of each hour's forecast as it stood when the hour
        ended (nowcast) vs measured. A day counts only if every recorded
        daylight hour has a measurement and HA ran the whole day.
        """
        if not self.enabled:
            return None
        days: list[dict[str, Any]] = []
        for offset in range(1, ACCURACY_DAYS + 1):
            day = today - timedelta(days=offset)
            start = dt_util.start_of_local_day(day).astimezone(timezone.utc)
            end = dt_util.start_of_local_day(day + timedelta(days=1)).astimezone(timezone.utc)
            hours = [h for h in self._records if start <= h < end]
            if not hours or any(self._records[h].actual is None for h in hours):
                continue
            actual = sum(self._records[h].actual or 0.0 for h in hours)
            # Every daylight hour in the frozen forecast must be recorded.
            forecast_hours = [h for h, v in frozen_hourly.items() if start <= h < end and v > 0]
            if any(h not in self._records for h in forecast_hours):
                continue
            same_day = sum(v for h, v in frozen_hourly.items() if start <= h < end)
            days.append(
                {
                    "date": day.isoformat(),
                    "actual_kwh": round(actual / 1000, 3),
                    "same_day_kwh": round(same_day / 1000, 3),
                    "day_ahead_kwh": (
                        round(self._day_ahead[day] / 1000, 3) if day in self._day_ahead else None
                    ),
                }
            )
        if not days:
            return {"days": []}

        def stats(key: str) -> tuple[float | None, float | None]:
            pairs = [(d[key], d["actual_kwh"]) for d in days if d[key] is not None]
            total = sum(a for _, a in pairs)
            if not pairs or total <= 0:
                return None, None
            mae = sum(abs(f - a) for f, a in pairs) / total * 100
            bias = sum(f - a for f, a in pairs) / total * 100
            return round(mae, 1), round(bias, 1)

        da_mae, da_bias = stats("day_ahead_kwh")
        sd_mae, sd_bias = stats("same_day_kwh")
        return {
            "day_ahead_error": da_mae,
            "day_ahead_bias": da_bias,
            "same_day_error": sd_mae,
            "same_day_bias": sd_bias,
            "days": days,
        }

    def summary(self) -> dict[str, Any]:
        """Attributes / diagnostics."""
        with_actual = sum(1 for r in self._records.values() if r.actual is not None)
        return {
            "production_sensor": self.sensor_id,
            "statistic": self.statistic_kind,
            "recorded_hours": len(self._records),
            "measured_hours": with_actual,
            "last_error": self.last_error,
            **self.calibration.as_dict(),
        }
