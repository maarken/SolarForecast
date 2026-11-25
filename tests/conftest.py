"""Fixtures for Solar Forecast tests."""

from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

pytest_plugins = "pytest_homeassistant_custom_component"

WEATHER_ID = "weather.test"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integrations in all tests."""


def make_hourly(
    start: datetime, hours: int, cloud: float, temp: float = 20.0
) -> list[dict[str, Any]]:
    """Build a get_forecasts style hourly list."""
    start = start.replace(minute=0, second=0, microsecond=0)
    return [
        {
            "datetime": (start + timedelta(hours=h)).isoformat(),
            "cloud_coverage": cloud,
            "temperature": temp,
            "condition": "sunny",
        }
        for h in range(hours)
    ]


@pytest.fixture
def forecast_data() -> dict[str, list[dict[str, Any]]]:
    """Mutable forecasts returned by the patched weather fetch, by type."""
    now = dt_util.utcnow()
    return {"hourly": make_hourly(now, 48, 0.0), "daily": []}


@pytest.fixture
def mock_weather(
    hass: HomeAssistant, forecast_data: dict[str, list[dict[str, Any]]]
) -> Generator[dict[str, list[dict[str, Any]]]]:
    """Provide a weather entity state and patch the forecast service call."""
    hass.states.async_set(WEATHER_ID, "sunny", {"supported_features": 3, "temperature_unit": "°C"})

    async def _fake(_hass: HomeAssistant, entity_id: str, forecast_type: str):
        assert entity_id == WEATHER_ID
        return list(forecast_data[forecast_type])

    with patch("custom_components.solar_forecast.weather_input._get_forecast", side_effect=_fake):
        yield forecast_data
