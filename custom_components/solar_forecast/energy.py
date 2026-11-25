"""Energy dashboard solar forecast platform."""

from typing import Any

from homeassistant.core import HomeAssistant

from .const import DOMAIN


async def async_get_solar_forecast(
    hass: HomeAssistant, config_entry_id: str
) -> dict[str, dict[str, float]] | None:
    """Return hourly forecast energy (Wh) keyed by hour start, for the energy dashboard."""
    entry = hass.config_entries.async_get_entry(config_entry_id)
    if entry is None or entry.domain != DOMAIN:
        return None
    coordinator: Any = getattr(entry, "runtime_data", None)
    if coordinator is None or coordinator.data is None:
        return None
    return {
        "wh_hours": {
            hour.isoformat(): round(wh, 1)
            for hour, wh in sorted(coordinator.data.hourly_wh.items())
        }
    }
