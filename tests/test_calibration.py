"""End-to-end calibration tests with the real recorder and imported statistics."""

from datetime import datetime, timedelta

from freezegun.api import FrozenDateTimeFactory
import pytest

from homeassistant.components.recorder.models import StatisticMeanType
from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.recorder.common import (
    async_wait_recording_done,
)

from custom_components.solar_forecast.const import CONF_PRODUCTION_SENSOR, DOMAIN

from .conftest import make_hourly
from .test_config_flow import _entry

SENSOR = "sensor.pv_energy"
DAY1 = datetime.fromisoformat("2026-06-20T06:00:00-07:00")
TRUE_FACTOR = 0.8


@pytest.fixture
def mock_recorder_before_hass(async_setup_recorder_instance) -> None:
    """Set up the recorder before hass so statistics are available."""


def _sensor_state(hass: HomeAssistant, entry: MockConfigEntry, key: str):
    reg = er.async_get(hass)
    entity_id = reg.async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_{key}")
    assert entity_id, key
    return hass.states.get(entity_id)


def _calibrated_entry() -> MockConfigEntry:
    base = _entry()
    return MockConfigEntry(
        domain=DOMAIN,
        title="Home",
        data=dict(base.data),
        options={CONF_PRODUCTION_SENSOR: SENSOR},
        subentries_data=[
            {
                "data": dict(s.data),
                "subentry_type": s.subentry_type,
                "title": s.title,
                "unique_id": None,
            }
            for s in base.subentries.values()
        ],
    )


async def _import(hass: HomeAssistant, hourly_wh: dict[datetime, float]) -> None:
    total = 0.0
    stats = []
    for hour in sorted(hourly_wh):
        total += hourly_wh[hour] / 1000.0
        stats.append({"start": hour, "state": total, "sum": total})
    async_import_statistics(
        hass,
        {
            "has_sum": True,
            "mean_type": StatisticMeanType.NONE,
            "name": "PV energy",
            "source": "recorder",
            "statistic_id": SENSOR,
            "unit_of_measurement": "kWh",
            "unit_class": "energy",
        },
        stats,
    )
    await async_wait_recording_done(hass)


async def test_calibration_learns_system_factor(
    recorder_mock, hass: HomeAssistant, freezer: FrozenDateTimeFactory, mock_weather
) -> None:
    """With production at 80 % of the model, calibration lowers the forecast."""
    await hass.config.async_set_time_zone("America/Los_Angeles")
    freezer.move_to(DAY1)
    hass.states.async_set(
        SENSOR,
        "0",
        {"device_class": "energy", "state_class": "total_increasing", "unit_of_measurement": "kWh"},
    )
    start = dt_util.start_of_local_day(DAY1.date())
    mock_weather["hourly"] = make_hourly(start, 24 * 5, 0.0, 20.0)

    entry = _calibrated_entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    coordinator = entry.runtime_data
    uncalibrated_d3 = coordinator.data.energy_day(DAY1.date() + timedelta(days=3))
    assert _sensor_state(hass, entry, "calibration_factor").state == "1.0"

    # Run two days with updates every 30 minutes; production is 80 % of the
    # (uncalibrated) model, recorded into long-term statistics.
    actual: dict[datetime, float] = {}
    t = DAY1
    end = DAY1 + timedelta(days=2)
    while t < end:
        t += timedelta(minutes=30)
        freezer.move_to(t)
        await coordinator.async_refresh()
        hour = dt_util.as_utc(t).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        if hour not in actual and hour >= dt_util.as_utc(DAY1):
            raw = (
                sum(
                    (0.5 if k in (0, 12) else 1.0)
                    * coordinator.model.evaluate(hour + timedelta(minutes=5 * k), 0.0, 20.0).total
                    for k in range(13)
                )
                / 12
            )
            actual[hour] = TRUE_FACTOR * raw
            await _import(hass, actual)

    # Force a refit and refresh.
    coordinator.calibration.last_fit = None
    freezer.move_to(end + timedelta(minutes=20))
    await coordinator.async_refresh()
    await hass.async_block_till_done()

    summary = coordinator.data.calibration
    assert summary["measured_hours"] > 20
    factor = float(_sensor_state(hass, entry, "calibration_factor").state)
    # Shrinkage towards 1 with ~30 clear hours: between truth and 1.
    assert TRUE_FACTOR <= factor < 0.86
    # The shrunk system factor plus the clear-band cloud correction together
    # recover the true ratio.
    calibrated_d3 = coordinator.data.energy_day(DAY1.date() + timedelta(days=3))
    assert calibrated_d3 / uncalibrated_d3 == pytest.approx(TRUE_FACTOR, rel=0.03)

    # Accuracy: day 1 is complete with measurements; day-ahead for day 2 was
    # recorded during day 1.
    acc = _sensor_state(hass, entry, "forecast_error_day_ahead")
    days = {d["date"]: d for d in acc.attributes["days"]}
    day2 = (DAY1.date() + timedelta(days=1)).isoformat()
    assert day2 in days
    assert days[day2]["day_ahead_kwh"] is not None
    # Day 2's day-ahead forecast was made on the evening of day 1, after the
    # calibration had learnt from day 1, so the error is small (it would be
    # ~25 % uncalibrated). Day 1 is excluded: HA started after sunrise.
    assert float(acc.state) < 6
    assert DAY1.date().isoformat() not in days


async def test_invalid_production_sensor_rejected(hass: HomeAssistant, mock_weather) -> None:
    """The options flow rejects sensors that cannot provide statistics."""
    entry = _entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    hass.states.async_set("sensor.temp", "20", {"device_class": "temperature"})
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_PRODUCTION_SENSOR: "sensor.temp", "turbidity_scale": 1.0}
    )
    assert result["errors"] == {CONF_PRODUCTION_SENSOR: "production_sensor_invalid"}


async def test_no_recorder_does_not_break_forecast(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory, mock_weather
) -> None:
    """Without recorder, calibration reports an error but the forecast works."""
    await hass.config.async_set_time_zone("America/Los_Angeles")
    freezer.move_to(DAY1 + timedelta(hours=6))
    start = dt_util.start_of_local_day(DAY1.date())
    mock_weather["hourly"] = make_hourly(start, 72, 0.0, 20.0)
    entry = _calibrated_entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    coordinator = entry.runtime_data
    for _ in range(4):
        freezer.tick(timedelta(minutes=30))
        await coordinator.async_refresh()
    assert coordinator.last_update_success
    assert coordinator.data.calibration["last_error"]
    assert float(_sensor_state(hass, entry, "power_production_now").state) > 0


async def test_statistics_gap_and_sign_handling(
    recorder_mock, hass: HomeAssistant, freezer: FrozenDateTimeFactory, mock_weather
) -> None:
    """Regression: a gap's backlog landed in one hour; negative sensors were silent."""
    await hass.config.async_set_time_zone("America/Los_Angeles")
    freezer.move_to(DAY1 + timedelta(hours=10))
    hass.states.async_set(
        SENSOR,
        "0",
        {"device_class": "energy", "state_class": "total_increasing", "unit_of_measurement": "kWh"},
    )
    start = dt_util.start_of_local_day(DAY1.date())
    mock_weather["hourly"] = make_hourly(start, 72, 0.0, 20.0)
    entry = _calibrated_entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    manager = entry.runtime_data.calibration

    h0 = dt_util.as_utc(DAY1)
    hours = [h0 + timedelta(hours=i) for i in (0, 1, 2, 6, 7)]  # gap: 3, 4, 5 missing
    total = 0.0
    stats = []
    for h in hours:
        total += 1.0
        stats.append({"start": h, "state": total, "sum": total})
    async_import_statistics(
        hass,
        {
            "has_sum": True,
            "mean_type": StatisticMeanType.NONE,
            "name": "PV energy",
            "source": "recorder",
            "statistic_id": SENSOR,
            "unit_of_measurement": "kWh",
            "unit_class": "energy",
        },
        stats,
    )
    await async_wait_recording_done(hass)
    values = await manager._async_query_statistics(h0 + timedelta(hours=1), h0 + timedelta(hours=8))
    # Hours 1, 2 and 7 have a preceding row; hour 6 follows the gap and would
    # otherwise carry 4 kWh of backlog.
    assert values == {
        h0 + timedelta(hours=1): pytest.approx(1000.0),
        h0 + timedelta(hours=2): pytest.approx(1000.0),
        h0 + timedelta(hours=7): pytest.approx(1000.0),
    }
    assert manager.last_error is None


async def test_negative_power_sensor_reports_error(
    recorder_mock, hass: HomeAssistant, freezer: FrozenDateTimeFactory, mock_weather
) -> None:
    """A power sensor with negative production values is flagged."""
    power = "sensor.pv_power"
    await hass.config.async_set_time_zone("America/Los_Angeles")
    freezer.move_to(DAY1 + timedelta(hours=10))
    start = dt_util.start_of_local_day(DAY1.date())
    mock_weather["hourly"] = make_hourly(start, 72, 0.0, 20.0)
    base = _calibrated_entry()
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home",
        data=dict(base.data),
        options={CONF_PRODUCTION_SENSOR: power},
        subentries_data=[
            {
                "data": dict(s.data),
                "subentry_type": s.subentry_type,
                "title": s.title,
                "unique_id": None,
            }
            for s in base.subentries.values()
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    manager = entry.runtime_data.calibration

    h0 = dt_util.as_utc(DAY1)
    async_import_statistics(
        hass,
        {
            "has_sum": False,
            "mean_type": StatisticMeanType.ARITHMETIC,
            "name": "PV power",
            "source": "recorder",
            "statistic_id": power,
            "unit_of_measurement": "kW",
            "unit_class": "power",
        },
        [
            {"start": h0 + timedelta(hours=i), "mean": -3.0, "min": -3.0, "max": -3.0}
            for i in range(5)
        ],
    )
    await async_wait_recording_done(hass)
    values = await manager._async_query_statistics(h0, h0 + timedelta(hours=5))
    assert values == {}
    assert "negative" in manager.last_error
