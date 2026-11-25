"""The Solar Forecast integration."""

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.storage import Store

from .const import DOMAIN, STORAGE_VERSION, SUBENTRY_TYPE_ARRAY
from .coordinator import SolarForecastConfigEntry, SolarForecastCoordinator

PLATFORMS: list[Platform] = [Platform.SENSOR]


def _issue_id(entry: SolarForecastConfigEntry) -> str:
    return f"no_arrays_{entry.entry_id}"


async def async_setup_entry(hass: HomeAssistant, entry: SolarForecastConfigEntry) -> bool:
    """Set up Solar Forecast from a config entry."""
    # Registered first so that adding an array to an empty entry reloads it.
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    if not any(s.subentry_type == SUBENTRY_TYPE_ARRAY for s in entry.subentries.values()):
        ir.async_create_issue(
            hass,
            DOMAIN,
            _issue_id(entry),
            is_fixable=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key="no_arrays",
            translation_placeholders={"name": entry.title},
        )
        return True
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry))

    coordinator = SolarForecastCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: SolarForecastConfigEntry) -> bool:
    """Unload a config entry."""
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry))
    if getattr(entry, "runtime_data", None) is None:
        return True  # set up without arrays; no platforms were forwarded
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: SolarForecastConfigEntry) -> None:
    """Clean up when the entry is deleted."""
    ir.async_delete_issue(hass, DOMAIN, _issue_id(entry))
    for name in ("history", "calibration"):
        await Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}.{name}").async_remove()


async def _async_update_listener(hass: HomeAssistant, entry: SolarForecastConfigEntry) -> None:
    """Reload when data, options or array subentries change."""
    hass.config_entries.async_schedule_reload(entry.entry_id)
