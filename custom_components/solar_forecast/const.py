"""Constants for the Solar Forecast integration."""

from datetime import timedelta
from typing import Final

DOMAIN: Final = "solar_forecast"

UPDATE_INTERVAL: Final = timedelta(minutes=5)
HISTORY_DAYS: Final = 8
STORAGE_VERSION: Final = 1

SUBENTRY_TYPE_ARRAY: Final = "array"

# Main entry
CONF_WEATHER_ENTITIES: Final = "weather_entities"
CONF_INVERTER_AC_POWER: Final = "inverter_ac_power"
CONF_INVERTER_EFFICIENCY: Final = "inverter_efficiency"
CONF_TURBIDITY_SCALE: Final = "turbidity_scale"
CONF_PRODUCTION_SENSOR: Final = "production_sensor"

# Array subentry
CONF_TILT: Final = "tilt"
CONF_AZIMUTH: Final = "azimuth"
CONF_DC_POWER: Final = "dc_power"
CONF_TEMP_COEFFICIENT: Final = "temp_coefficient"
CONF_MOUNTING: Final = "mounting"
CONF_LOSSES: Final = "losses"
CONF_ALBEDO: Final = "albedo"
CONF_MODULE_EFFICIENCY: Final = "module_efficiency"

MOUNTING_OPTIONS: Final = ["open_rack", "roof_gap", "integrated"]

DEFAULT_INVERTER_EFFICIENCY: Final = 96.0  # %
DEFAULT_TURBIDITY_SCALE: Final = 1.0
DEFAULT_TILT: Final = 30.0
DEFAULT_AZIMUTH: Final = 180.0
DEFAULT_TEMP_COEFFICIENT: Final = -0.37  # %/K
DEFAULT_MOUNTING: Final = "roof_gap"
DEFAULT_LOSSES: Final = 10.0  # %
DEFAULT_ALBEDO: Final = 0.20
DEFAULT_MODULE_EFFICIENCY: Final = 20.0  # %

FORECAST_DAYS: Final = 7  # today + 6
