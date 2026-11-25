"""Fetch forecasts from Home Assistant weather entities and convert to model input."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import math
from typing import Any

from homeassistant.components.weather import (
    ATTR_FORECAST_CLOUD_COVERAGE,
    ATTR_FORECAST_TEMP,
    ATTR_FORECAST_TEMP_LOW,
    ATTR_FORECAST_TIME,
    DOMAIN as WEATHER_DOMAIN,
    WeatherEntityFeature,
)
from homeassistant.const import ATTR_SUPPORTED_FEATURES, STATE_UNAVAILABLE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import TemperatureConverter

from .model.weather import WeatherSample, WeatherSeries, daily_to_hourly

_LOGGER = logging.getLogger(__name__)

ATTR_TEMPERATURE_UNIT = "temperature_unit"
SERVICE_GET_FORECASTS = "get_forecasts"


@dataclass(slots=True)
class SourceReport:
    """Diagnostics about one weather source."""

    entity_id: str
    hourly_points: int = 0
    daily_points: int = 0
    first: datetime | None = None
    last: datetime | None = None
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "entity_id": self.entity_id,
            "hourly_points": self.hourly_points,
            "daily_points": self.daily_points,
            "first": self.first.isoformat() if self.first else None,
            "last": self.last.isoformat() if self.last else None,
            "errors": list(self.errors),
        }


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except TypeError, ValueError:
        return None
    return result if math.isfinite(result) else None


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed: datetime | None = value
    elif isinstance(value, str):
        parsed = dt_util.parse_datetime(value)
    else:
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt_util.get_default_time_zone())
    return parsed


def _cloud_fraction(item: dict[str, Any]) -> float | None:
    value = _to_float(item.get(ATTR_FORECAST_CLOUD_COVERAGE))
    if value is None:
        return None
    return min(max(value / 100.0, 0.0), 1.0)


def _celsius(value: Any, unit: str) -> float | None:
    temp = _to_float(value)
    if temp is None:
        return None
    if unit == UnitOfTemperature.CELSIUS:
        return temp
    try:
        return TemperatureConverter.convert(temp, unit, UnitOfTemperature.CELSIUS)
    except HomeAssistantError:
        return None


async def _get_forecast(
    hass: HomeAssistant, entity_id: str, forecast_type: str
) -> list[dict[str, Any]]:
    response = await hass.services.async_call(
        WEATHER_DOMAIN,
        SERVICE_GET_FORECASTS,
        {"type": forecast_type},
        target={"entity_id": entity_id},
        blocking=True,
        return_response=True,
    )
    if not isinstance(response, dict):
        return []
    entity_resp = response.get(entity_id)
    if not isinstance(entity_resp, dict):
        return []
    forecast = entity_resp.get("forecast")
    if not isinstance(forecast, list):
        return []
    return [f for f in forecast if isinstance(f, dict)]


def hourly_samples(forecast: list[dict[str, Any]], temp_unit: str) -> list[WeatherSample]:
    """Convert a ``get_forecasts`` hourly list into model samples."""
    samples: list[WeatherSample] = []
    for item in forecast:
        when = _parse_time(item.get(ATTR_FORECAST_TIME))
        if when is None:
            continue
        samples.append(
            WeatherSample(
                time=when,
                cloud_cover=_cloud_fraction(item),
                temperature=_celsius(item.get(ATTR_FORECAST_TEMP), temp_unit),
            )
        )
    samples.sort(key=lambda s: s.time)
    return samples


def _forecast_day(when: datetime) -> Any:
    """Calendar day a daily forecast entry refers to.

    Most providers stamp daily entries at local midnight (or another local
    time of that day). Some (e.g. Open-Meteo in core) stamp the calendar date
    at 00:00 UTC, which west of Greenwich would fall on the previous local
    evening; treat an exact UTC midnight as that UTC date.
    """
    utc = dt_util.as_utc(when)
    if utc.hour == 0 and utc.minute == 0 and utc.second == 0 and utc.microsecond == 0:
        return utc.date()
    return dt_util.as_local(when).date()


def daily_samples(
    forecast: list[dict[str, Any]], temp_unit: str, after: datetime | None
) -> list[WeatherSample]:
    """Expand a ``get_forecasts`` daily list into hourly samples.

    Only samples strictly later than ``after`` (the end of the hourly
    forecast) are returned, so hourly data always takes precedence.
    """
    samples: list[WeatherSample] = []
    seen_days: set[Any] = set()
    for item in forecast:
        when = _parse_time(item.get(ATTR_FORECAST_TIME))
        cloud = _cloud_fraction(item)
        if when is None or cloud is None:
            continue
        local_day = _forecast_day(when)
        if local_day in seen_days:
            continue
        seen_days.add(local_day)
        day_start = dt_util.start_of_local_day(local_day)
        for sample in daily_to_hourly(
            day_start,
            cloud,
            _celsius(item.get(ATTR_FORECAST_TEMP), temp_unit),
            _celsius(item.get(ATTR_FORECAST_TEMP_LOW), temp_unit),
        ):
            if after is None or sample.time > after:
                samples.append(sample)
    samples.sort(key=lambda s: s.time)
    # Drop duplicate instants (day boundaries) keeping the later day's value.
    deduped: dict[datetime, WeatherSample] = {}
    for sample in samples:
        deduped[sample.time] = sample
    return list(deduped.values())


async def async_fetch_weather(
    hass: HomeAssistant, entity_ids: list[str]
) -> tuple[list[WeatherSeries], list[SourceReport]]:
    """Fetch hourly (and daily as fallback) forecasts from each weather entity."""
    series: list[WeatherSeries] = []
    reports: list[SourceReport] = []
    for entity_id in entity_ids:
        report = SourceReport(entity_id)
        reports.append(report)
        state = hass.states.get(entity_id)
        if state is None or state.state == STATE_UNAVAILABLE:
            report.errors.append("entity unavailable")
            continue
        features = int(state.attributes.get(ATTR_SUPPORTED_FEATURES) or 0)
        temp_unit = str(
            state.attributes.get(ATTR_TEMPERATURE_UNIT) or hass.config.units.temperature_unit
        )

        hourly: list[WeatherSample] = []
        if features & WeatherEntityFeature.FORECAST_HOURLY:
            try:
                hourly = hourly_samples(await _get_forecast(hass, entity_id, "hourly"), temp_unit)
            except Exception as err:  # noqa: BLE001 - isolate each source
                report.errors.append(f"hourly forecast failed: {err!r}")
        else:
            report.errors.append("no hourly forecast support")

        hourly_with_cloud = [s for s in hourly if s.cloud_cover is not None]
        if hourly and not hourly_with_cloud:
            report.errors.append("hourly forecast has no cloud_coverage")
        report.hourly_points = len(hourly_with_cloud)
        hourly_end = hourly_with_cloud[-1].time if hourly_with_cloud else None

        daily: list[WeatherSample] = []
        if features & WeatherEntityFeature.FORECAST_DAILY:
            try:
                daily = daily_samples(
                    await _get_forecast(hass, entity_id, "daily"), temp_unit, hourly_end
                )
            except Exception as err:  # noqa: BLE001 - isolate each source
                report.errors.append(f"daily forecast failed: {err!r}")
        if hourly_end is not None and daily:
            # Bridge only if the daily data continues the hourly data closely.
            if daily[0].time - hourly_end > timedelta(hours=3):
                daily = []
        report.daily_points = len(daily)

        combined = hourly_with_cloud + daily
        if not combined:
            if not report.errors:
                report.errors.append("no usable forecast data")
            continue
        report.first = combined[0].time
        report.last = combined[-1].time
        series.append(WeatherSeries(combined))
        for err in report.errors:
            _LOGGER.debug("%s: %s", entity_id, err)
    return series, reports
