"""Diagnostics support for Solar Forecast."""

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.core import HomeAssistant

from .coordinator import SolarForecastConfigEntry

TO_REDACT = {CONF_LATITUDE, CONF_LONGITUDE}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: SolarForecastConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    result: dict[str, Any] = {
        "entry": {
            "data": async_redact_data(dict(entry.data), TO_REDACT),
            "options": dict(entry.options),
            "arrays": {
                sid: {"title": s.title, "data": dict(s.data)} for sid, s in entry.subentries.items()
            },
        }
    }
    coordinator = getattr(entry, "runtime_data", None)
    if coordinator is None or coordinator.data is None:
        return result
    data = coordinator.data
    model = coordinator.model
    result["model"] = {
        "altitude_m": model.site.altitude if model else None,
        "monthly_linke_turbidity": list(model.site.monthly_linke_turbidity) if model else None,
        "linke_turbidity_scale": model.site.linke_turbidity_scale if model else None,
    }
    result["updated"] = data.updated.isoformat()
    result["weather_sources"] = [r.as_dict() for r in data.sources]
    result["weather_now"] = (
        None
        if data.weather_now is None
        else {
            "cloud_cover": data.weather_now.cloud_cover,
            "temperature": data.weather_now.temperature,
        }
    )
    result["calibration"] = data.calibration
    result["accuracy"] = data.accuracy
    result["hourly_wh"] = {h.isoformat(): round(v, 1) for h, v in sorted(data.hourly_wh.items())}
    return result
