"""Config flow tests."""

from homeassistant import config_entries
from homeassistant.const import CONF_ELEVATION, CONF_LATITUDE, CONF_LONGITUDE, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.solar_forecast.const import (
    CONF_ALBEDO,
    CONF_AZIMUTH,
    CONF_DC_POWER,
    CONF_INVERTER_AC_POWER,
    CONF_INVERTER_EFFICIENCY,
    CONF_LOSSES,
    CONF_MODULE_EFFICIENCY,
    CONF_MOUNTING,
    CONF_TEMP_COEFFICIENT,
    CONF_TILT,
    CONF_TURBIDITY_SCALE,
    CONF_WEATHER_ENTITIES,
    DOMAIN,
    SUBENTRY_TYPE_ARRAY,
)

from .conftest import WEATHER_ID

SITE = {
    CONF_NAME: "Home",
    CONF_LATITUDE: 48.75,
    CONF_LONGITUDE: -122.48,
    CONF_ELEVATION: 30,
    CONF_WEATHER_ENTITIES: [WEATHER_ID],
    CONF_INVERTER_AC_POWER: 5000,
    CONF_INVERTER_EFFICIENCY: 96.5,
}
ARRAY = {
    CONF_NAME: "South roof",
    CONF_DC_POWER: 6000,
    CONF_TILT: 25,
    CONF_AZIMUTH: 180,
    CONF_MOUNTING: "roof_gap",
    CONF_TEMP_COEFFICIENT: -0.34,
    CONF_LOSSES: 9,
    CONF_MODULE_EFFICIENCY: 21,
    CONF_ALBEDO: 0.2,
}


async def test_user_flow_creates_entry_with_array(hass: HomeAssistant, mock_weather) -> None:
    """Site step then array step creates an entry with one array subentry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(result["flow_id"], SITE)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "array"

    result = await hass.config_entries.flow.async_configure(result["flow_id"], ARRAY)
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Home"
    entry = result["result"]
    assert CONF_NAME not in entry.data
    assert entry.data[CONF_WEATHER_ENTITIES] == [WEATHER_ID]
    subentries = list(entry.subentries.values())
    assert len(subentries) == 1
    assert subentries[0].subentry_type == SUBENTRY_TYPE_ARRAY
    assert subentries[0].title == "South roof"
    assert subentries[0].data[CONF_DC_POWER] == 6000
    assert CONF_NAME not in subentries[0].data


async def test_user_flow_rejects_weather_without_cloud(hass: HomeAssistant, mock_weather) -> None:
    """A weather entity with no cloud coverage is rejected."""
    for item in mock_weather["hourly"]:
        item.pop("cloud_coverage")
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(result["flow_id"], SITE)
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_WEATHER_ENTITIES: "no_cloud_forecast"}


async def test_user_flow_rejects_missing_entity(hass: HomeAssistant, mock_weather) -> None:
    """An unknown weather entity is rejected."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {**SITE, CONF_WEATHER_ENTITIES: ["weather.missing"]}
    )
    assert result["errors"] == {CONF_WEATHER_ENTITIES: "no_cloud_forecast"}


def _entry() -> MockConfigEntry:
    site = {k: v for k, v in SITE.items() if k != CONF_NAME}
    array = {k: v for k, v in ARRAY.items() if k != CONF_NAME}
    return MockConfigEntry(
        domain=DOMAIN,
        title="Home",
        data=site,
        subentries_data=[
            {
                "data": array,
                "subentry_type": SUBENTRY_TYPE_ARRAY,
                "title": "South roof",
                "unique_id": None,
            }
        ],
    )


async def test_add_and_reconfigure_array(hass: HomeAssistant, mock_weather) -> None:
    """Arrays can be added and edited through subentry flows."""
    entry = _entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, SUBENTRY_TYPE_ARRAY), context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {**ARRAY, CONF_NAME: "West roof", CONF_AZIMUTH: 270}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert len(entry.subentries) == 2
    west = next(s for s in entry.subentries.values() if s.title == "West roof")
    assert west.data[CONF_AZIMUTH] == 270
    # Reloaded with two arrays.
    assert len(entry.runtime_data.array_ids) == 2

    result = await entry.start_subentry_reconfigure_flow(hass, west.subentry_id)
    assert result["type"] is FlowResultType.FORM
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {**ARRAY, CONF_NAME: "West roof", CONF_AZIMUTH: 260}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.subentries[west.subentry_id].data[CONF_AZIMUTH] == 260


async def test_reconfigure_site(hass: HomeAssistant, mock_weather) -> None:
    """The site can be reconfigured."""
    entry = _entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await entry.start_reconfigure_flow(hass)
    assert result["type"] is FlowResultType.FORM
    new = {k: v for k, v in SITE.items() if k != CONF_NAME}
    new[CONF_ELEVATION] = 500
    result = await hass.config_entries.flow.async_configure(result["flow_id"], new)
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data[CONF_ELEVATION] == 500
    assert entry.runtime_data.model.site.altitude == 500


async def test_options_flow(hass: HomeAssistant, mock_weather) -> None:
    """Turbidity scale option is stored and applied."""
    entry = _entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_TURBIDITY_SCALE: 1.2}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.runtime_data.model.site.linke_turbidity_scale == 1.2
