"""Config flow for Solar Forecast."""

from typing import Any

import voluptuous as vol  # resolves to probatio on Home Assistant >= 2026.9

from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryData,
    ConfigSubentryFlow,
    OptionsFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_ELEVATION, CONF_LATITUDE, CONF_LONGITUDE, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
)

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
    DEFAULT_AZIMUTH,
    DEFAULT_INVERTER_EFFICIENCY,
    DEFAULT_LOSSES,
    DEFAULT_MODULE_EFFICIENCY,
    DEFAULT_MOUNTING,
    DEFAULT_TEMP_COEFFICIENT,
    DEFAULT_TILT,
    DEFAULT_TURBIDITY_SCALE,
    DOMAIN,
    MOUNTING_OPTIONS,
    SUBENTRY_TYPE_ARRAY,
)
from .weather_input import async_fetch_weather


def _number(
    min_value: float,
    max_value: float,
    step: float | str,
    unit: str | None = None,
    mode: NumberSelectorMode = NumberSelectorMode.BOX,
) -> NumberSelector:
    config = NumberSelectorConfig(min=min_value, max=max_value, step=step, mode=mode)
    if unit is not None:
        config["unit_of_measurement"] = unit
    return NumberSelector(config)


def _site_schema(hass: HomeAssistant, defaults: dict[str, Any], include_name: bool) -> vol.Schema:
    fields: dict[Any, Any] = {}
    if include_name:
        fields[vol.Required(CONF_NAME, default=defaults.get(CONF_NAME, "Solar Forecast"))] = (
            TextSelector()
        )
    fields.update(
        {
            vol.Required(
                CONF_LATITUDE, default=defaults.get(CONF_LATITUDE, hass.config.latitude)
            ): _number(-90, 90, "any", "°"),
            vol.Required(
                CONF_LONGITUDE, default=defaults.get(CONF_LONGITUDE, hass.config.longitude)
            ): _number(-180, 180, "any", "°"),
            vol.Required(
                CONF_ELEVATION, default=defaults.get(CONF_ELEVATION, hass.config.elevation)
            ): _number(-500, 9000, 1, "m"),
            vol.Required(
                CONF_WEATHER_ENTITIES, default=defaults.get(CONF_WEATHER_ENTITIES, [])
            ): EntitySelector(EntitySelectorConfig(domain="weather", multiple=True)),
            vol.Required(
                CONF_INVERTER_AC_POWER, default=defaults.get(CONF_INVERTER_AC_POWER, 0)
            ): _number(0, 10_000_000, 1, "W"),
            vol.Required(
                CONF_INVERTER_EFFICIENCY,
                default=defaults.get(CONF_INVERTER_EFFICIENCY, DEFAULT_INVERTER_EFFICIENCY),
            ): _number(80, 100, 0.1, "%"),
        }
    )
    return vol.Schema(fields)


def _array_schema(defaults: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_NAME, default=defaults.get(CONF_NAME, "Array")): TextSelector(),
            (
                vol.Required(CONF_DC_POWER, default=defaults[CONF_DC_POWER])
                if CONF_DC_POWER in defaults
                else vol.Required(CONF_DC_POWER)
            ): _number(1, 10_000_000, 1, "W"),
            vol.Required(CONF_TILT, default=defaults.get(CONF_TILT, DEFAULT_TILT)): _number(
                0, 90, 0.1, "°"
            ),
            vol.Required(
                CONF_AZIMUTH, default=defaults.get(CONF_AZIMUTH, DEFAULT_AZIMUTH)
            ): _number(0, 360, 0.1, "°"),
            vol.Required(
                CONF_MOUNTING, default=defaults.get(CONF_MOUNTING, DEFAULT_MOUNTING)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=MOUNTING_OPTIONS,
                    mode=SelectSelectorMode.DROPDOWN,
                    translation_key=CONF_MOUNTING,
                )
            ),
            vol.Required(
                CONF_TEMP_COEFFICIENT,
                default=defaults.get(CONF_TEMP_COEFFICIENT, DEFAULT_TEMP_COEFFICIENT),
            ): _number(-1.0, 0.0, 0.01, "%/°C"),
            vol.Required(CONF_LOSSES, default=defaults.get(CONF_LOSSES, DEFAULT_LOSSES)): _number(
                0, 50, 0.1, "%"
            ),
            vol.Required(
                CONF_MODULE_EFFICIENCY,
                default=defaults.get(CONF_MODULE_EFFICIENCY, DEFAULT_MODULE_EFFICIENCY),
            ): _number(5, 30, 0.1, "%"),
            vol.Required(CONF_ALBEDO, default=defaults.get(CONF_ALBEDO, DEFAULT_ALBEDO)): _number(
                0, 1, 0.01
            ),
        }
    )


async def _validate_site(hass: HomeAssistant, user_input: dict[str, Any]) -> dict[str, str]:
    """Check that at least one selected weather entity yields cloud cover."""
    errors: dict[str, str] = {}
    entity_ids = list(user_input.get(CONF_WEATHER_ENTITIES) or [])
    if not entity_ids:
        errors[CONF_WEATHER_ENTITIES] = "no_weather_entities"
        return errors
    series, _reports = await async_fetch_weather(hass, entity_ids)
    if not series:
        errors[CONF_WEATHER_ENTITIES] = "no_cloud_forecast"
    return errors


def _split_array_input(user_input: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    data = dict(user_input)
    title = str(data.pop(CONF_NAME)).strip() or "Array"
    return title, data


class SolarForecastConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the initial setup: site + weather, then the first array."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialise the flow."""
        self._site: dict[str, Any] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Configure the site and weather sources."""
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = await _validate_site(self.hass, user_input)
            if not errors:
                self._site = user_input
                return await self.async_step_array()
        return self.async_show_form(
            step_id="user",
            data_schema=_site_schema(self.hass, user_input or {}, include_name=True),
            errors=errors,
        )

    async def async_step_array(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Configure the first PV array."""
        if user_input is not None:
            title, array_data = _split_array_input(user_input)
            site = dict(self._site)
            name = str(site.pop(CONF_NAME)).strip() or "Solar Forecast"
            subentry: ConfigSubentryData = {
                "data": array_data,
                "subentry_type": SUBENTRY_TYPE_ARRAY,
                "title": title,
                "unique_id": None,
            }
            return self.async_create_entry(title=name, data=site, subentries=[subentry])
        return self.async_show_form(step_id="array", data_schema=_array_schema({}))

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change site or weather settings."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = await _validate_site(self.hass, user_input)
            if not errors:
                # The update listener reloads the entry.
                return self.async_update_and_abort(entry, data_updates=user_input)
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=_site_schema(self.hass, user_input or dict(entry.data), include_name=False),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return the options flow."""
        return SolarForecastOptionsFlow()

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return the subentry types supported by this integration."""
        return {SUBENTRY_TYPE_ARRAY: ArraySubentryFlow}


def _validate_production_sensor(hass: HomeAssistant, entity_id: str) -> str | None:
    """Return an error key if the entity cannot provide hourly statistics."""
    state = hass.states.get(entity_id)
    if state is None:
        return "production_sensor_missing"
    device_class = state.attributes.get("device_class")
    state_class = state.attributes.get("state_class")
    if device_class == SensorDeviceClass.ENERGY and state_class in (
        SensorStateClass.TOTAL,
        SensorStateClass.TOTAL_INCREASING,
    ):
        return None
    if device_class == SensorDeviceClass.POWER and state_class == SensorStateClass.MEASUREMENT:
        return None
    return "production_sensor_invalid"


class SolarForecastOptionsFlow(OptionsFlow):
    """Model tuning and calibration options."""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}
        if user_input is not None:
            sensor = user_input.get(CONF_PRODUCTION_SENSOR)
            if sensor and (err := _validate_production_sensor(self.hass, sensor)):
                errors[CONF_PRODUCTION_SENSOR] = err
            else:
                return self.async_create_entry(data=user_input)
        options = user_input or self.config_entry.options
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        vol.Optional(CONF_PRODUCTION_SENSOR): EntitySelector(
                            EntitySelectorConfig(
                                domain="sensor",
                                device_class=[SensorDeviceClass.ENERGY, SensorDeviceClass.POWER],
                            )
                        ),
                        vol.Required(
                            CONF_TURBIDITY_SCALE, default=DEFAULT_TURBIDITY_SCALE
                        ): _number(0.5, 2.0, 0.01),
                    }
                ),
                {
                    CONF_PRODUCTION_SENSOR: options.get(CONF_PRODUCTION_SENSOR),
                    CONF_TURBIDITY_SCALE: options.get(
                        CONF_TURBIDITY_SCALE, DEFAULT_TURBIDITY_SCALE
                    ),
                },
            ),
            errors=errors,
        )


class ArraySubentryFlow(ConfigSubentryFlow):
    """Add or edit a PV array (one orientation)."""

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Add an array."""
        if user_input is not None:
            title, data = _split_array_input(user_input)
            return self.async_create_entry(title=title, data=data)
        return self.async_show_form(step_id="user", data_schema=_array_schema({}))

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Edit an existing array."""
        subentry = self._get_reconfigure_subentry()
        if user_input is not None:
            title, data = _split_array_input(user_input)
            return self.async_update_and_abort(self._get_entry(), subentry, title=title, data=data)
        defaults = {**subentry.data, CONF_NAME: subentry.title}
        return self.async_show_form(step_id="reconfigure", data_schema=_array_schema(defaults))
