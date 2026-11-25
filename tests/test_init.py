"""Setup, sensor, energy platform and history tests."""

from datetime import datetime, timedelta

from freezegun.api import FrozenDateTimeFactory
import pytest

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er, issue_registry as ir
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_fire_time_changed

from custom_components.solar_forecast.const import DOMAIN
from custom_components.solar_forecast.energy import async_get_solar_forecast

from .conftest import make_hourly
from .test_config_flow import _entry

NOON_PDT = datetime.fromisoformat("2026-06-21T12:00:00-07:00")


def _state(hass: HomeAssistant, entry: MockConfigEntry, key: str, unique_prefix: str | None = None):
    reg = er.async_get(hass)
    uid = f"{unique_prefix or entry.entry_id}_{key}"
    entity_id = reg.async_get_entity_id("sensor", DOMAIN, uid)
    assert entity_id, uid
    return hass.states.get(entity_id)


@pytest.fixture
async def setup_entry(hass: HomeAssistant, freezer: FrozenDateTimeFactory, mock_weather):
    """Set up an entry at local noon on the June solstice with clear skies."""
    await hass.config.async_set_time_zone("America/Los_Angeles")
    freezer.move_to(NOON_PDT)
    start = dt_util.start_of_local_day(NOON_PDT.date())
    mock_weather["hourly"] = make_hourly(start, 72, 0.0, 20.0)
    entry = _entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_setup_and_sensors(hass: HomeAssistant, setup_entry: MockConfigEntry) -> None:
    """Sensors are created with physically sensible clear-sky values."""
    entry = setup_entry
    assert entry.state is ConfigEntryState.LOADED

    power = float(_state(hass, entry, "power_production_now").state)
    # 6 kWp, 25 deg south, clear solstice noon at 48.75N, 5 kW inverter limit.
    assert 4000 < power <= 5000

    today = float(_state(hass, entry, "energy_production_today").state)
    tomorrow = float(_state(hass, entry, "energy_production_tomorrow").state)
    assert 30 < today < 48
    assert abs(today - tomorrow) / today < 0.02  # same weather, one day apart

    remaining = float(_state(hass, entry, "energy_production_today_remaining").state)
    assert 0.3 * today < remaining < 0.7 * today

    attrs = _state(hass, entry, "energy_production_today").attributes
    hourly = attrs["forecast"]
    assert len(hourly) == 24
    assert sum(h["energy_wh"] for h in hourly) / 1000 == pytest.approx(today, rel=1e-3)

    peak_time = dt_util.parse_datetime(_state(hass, entry, "power_highest_peak_time_today").state)
    local_peak = dt_util.as_local(peak_time)
    # Solar noon at -122.48 in PDT is ~13:11.
    assert 12 <= local_peak.hour <= 14

    clear = float(_state(hass, entry, "clear_sky_power_now").state)
    assert clear == pytest.approx(power, rel=1e-6)
    assert float(_state(hass, entry, "cloud_cover_now").state) == 0

    # Array-level sensors sum to the site (single array here).
    array_id = entry.runtime_data.array_ids[0]
    arr_today = float(_state(hass, entry, "energy_production_today", array_id).state)
    assert arr_today == pytest.approx(today, rel=1e-6)


async def test_overcast_reduces_output(
    hass: HomeAssistant, setup_entry: MockConfigEntry, mock_weather, freezer
) -> None:
    """Full overcast reduces output to roughly a quarter (Kasten-Czeplak)."""
    entry = setup_entry
    clear_tomorrow = float(_state(hass, entry, "energy_production_tomorrow").state)
    start = dt_util.start_of_local_day(NOON_PDT.date())
    mock_weather["hourly"] = make_hourly(start, 72, 100.0, 20.0)
    await entry.runtime_data.async_refresh()
    await hass.async_block_till_done()
    cloudy_tomorrow = float(_state(hass, entry, "energy_production_tomorrow").state)
    assert 0.15 < cloudy_tomorrow / clear_tomorrow < 0.4


async def test_elapsed_hours_are_frozen(
    hass: HomeAssistant, setup_entry: MockConfigEntry, mock_weather, freezer
) -> None:
    """Hours that have ended keep their forecast; future hours follow updates."""
    entry = setup_entry
    coordinator = entry.runtime_data
    past_hour = dt_util.as_utc(NOON_PDT - timedelta(hours=2))
    before = coordinator.data.hourly_wh[past_hour]
    assert before > 0

    # Weather now says overcast for the whole period, including the past.
    start = dt_util.start_of_local_day(NOON_PDT.date())
    mock_weather["hourly"] = make_hourly(start, 72, 100.0, 20.0)
    await coordinator.async_refresh()
    after = coordinator.data.hourly_wh
    assert after[past_hour] == pytest.approx(before)
    future_hour = dt_util.as_utc(NOON_PDT + timedelta(hours=2))
    assert after[future_hour] < 0.5 * before


async def test_energy_platform(hass: HomeAssistant, setup_entry: MockConfigEntry) -> None:
    """The energy dashboard receives hour-start keyed Wh values."""
    result = await async_get_solar_forecast(hass, setup_entry.entry_id)
    assert result is not None
    wh_hours = result["wh_hours"]
    assert len(wh_hours) >= 48
    for key in wh_hours:
        when = dt_util.parse_datetime(key)
        assert when is not None and when.tzinfo is not None
        assert when.minute == 0 and when.second == 0
    today = float(_state(hass, setup_entry, "energy_production_today").state)
    day_start = dt_util.start_of_local_day(NOON_PDT.date())
    day_sum = sum(
        v
        for k, v in wh_hours.items()
        if day_start <= dt_util.parse_datetime(k) < day_start + timedelta(days=1)
    )
    assert day_sum / 1000 == pytest.approx(today, rel=1e-3)
    assert await async_get_solar_forecast(hass, "missing") is None


async def test_daily_fallback_extends_horizon(
    hass: HomeAssistant, setup_entry: MockConfigEntry, mock_weather
) -> None:
    """Days beyond the hourly forecast are filled from the daily forecast."""
    entry = setup_entry
    coordinator = entry.runtime_data
    day3 = NOON_PDT.date() + timedelta(days=3)
    assert coordinator.data.energy_day(day3) is None

    start = dt_util.start_of_local_day(NOON_PDT.date())
    mock_weather["daily"] = [
        {
            "datetime": dt_util.start_of_local_day(NOON_PDT.date() + timedelta(days=d)).isoformat(),
            "cloud_coverage": 0,
            "temperature": 24,
            "templow": 12,
        }
        for d in range(7)
    ]
    del start
    await coordinator.async_refresh()
    day1 = coordinator.data.energy_day(NOON_PDT.date() + timedelta(days=1))
    day3_energy = coordinator.data.energy_day(day3)
    assert day3_energy is not None
    assert day3_energy == pytest.approx(day1, rel=0.03)


async def test_unavailable_weather_fails_update(
    hass: HomeAssistant, setup_entry: MockConfigEntry, mock_weather
) -> None:
    """With no usable weather the coordinator reports failure."""
    coordinator = setup_entry.runtime_data
    mock_weather["hourly"] = []
    await coordinator.async_refresh()
    assert not coordinator.last_update_success


async def test_minute_tick_updates_power(
    hass: HomeAssistant, setup_entry: MockConfigEntry, freezer: FrozenDateTimeFactory
) -> None:
    """Power now is re-evaluated between coordinator updates."""
    first = float(_state(hass, setup_entry, "power_production_now").state)
    freezer.tick(timedelta(hours=4))
    # Prevent the coordinator from refreshing so only the tick runs.
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    later = float(_state(hass, setup_entry, "power_production_now").state)
    assert later < first


async def test_no_arrays_creates_issue(hass: HomeAssistant, mock_weather) -> None:
    """An entry without arrays loads with a repair issue and no sensors."""
    entry = MockConfigEntry(domain=DOMAIN, title="Empty", data=_entry().data)
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    issues = ir.async_get(hass)
    assert issues.async_get_issue(DOMAIN, f"no_arrays_{entry.entry_id}") is not None
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_unload(hass: HomeAssistant, setup_entry: MockConfigEntry) -> None:
    """The entry unloads cleanly."""
    assert await hass.config_entries.async_unload(setup_entry.entry_id)
    await hass.async_block_till_done()
    assert setup_entry.state is ConfigEntryState.NOT_LOADED


async def test_half_hour_timezone(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory, mock_weather
) -> None:
    """Regression: daily totals were always unknown in UTC+5:30 zones."""
    await hass.config.async_set_time_zone("Asia/Kolkata")
    noon = datetime.fromisoformat("2026-03-20T12:00:00+05:30")
    freezer.move_to(noon)
    start = dt_util.start_of_local_day(noon.date())
    mock_weather["hourly"] = make_hourly(start - timedelta(hours=1), 74, 0.0, 30.0)
    base = _entry()
    data = {**base.data, "latitude": 28.61, "longitude": 77.21, "elevation": 216}
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Delhi",
        data=data,
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

    today = float(_state(hass, entry, "energy_production_today").state)
    tomorrow = float(_state(hass, entry, "energy_production_tomorrow").state)
    remaining = float(_state(hass, entry, "energy_production_today_remaining").state)
    assert today > 20
    assert today == pytest.approx(tomorrow, rel=0.02)
    assert 0.3 * today < remaining < 0.7 * today
    assert float(_state(hass, entry, "energy_current_hour").state) > 0

    # Day total equals the power-curve integral over the local day.
    data_ = entry.runtime_data.data
    day_start = dt_util.start_of_local_day(noon.date())
    curve = data_.energy_in_window(day_start, day_start + timedelta(days=1))
    assert today * 1000 == pytest.approx(curve, rel=0.005)


async def test_elapsed_power_curve_preserved(
    hass: HomeAssistant, setup_entry: MockConfigEntry, mock_weather, freezer
) -> None:
    """Regression: elapsed part of today was recomputed from held current weather."""
    coordinator = setup_entry.runtime_data
    morning = dt_util.as_utc(NOON_PDT - timedelta(hours=2))
    before = coordinator.data.forecast.power_at(morning).total
    freezer.tick(timedelta(minutes=30))
    # Provider drops the past and now says overcast from here on.
    now = dt_util.utcnow()
    mock_weather["hourly"] = make_hourly(now, 60, 100.0, 20.0)
    await coordinator.async_refresh()
    assert coordinator.data.forecast.power_at(morning).total == pytest.approx(before)


async def test_failing_source_is_isolated(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory, mock_weather
) -> None:
    """Regression: a non-HA exception in one source aborted all sources."""
    from unittest.mock import patch

    from custom_components.solar_forecast.const import CONF_WEATHER_ENTITIES

    from .conftest import WEATHER_ID

    freezer.move_to(NOON_PDT)
    mock_weather["hourly"] = make_hourly(NOON_PDT - timedelta(hours=13), 72, 0.0)
    hass.states.async_set("weather.bad", "sunny", {"supported_features": 3})
    good = list(mock_weather["hourly"])

    async def _fake(_hass, entity_id, forecast_type):
        if entity_id == "weather.bad":
            raise TimeoutError
        return good if forecast_type == "hourly" else []

    base = _entry()
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home",
        data={**base.data, CONF_WEATHER_ENTITIES: [WEATHER_ID, "weather.bad"]},
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
    with patch("custom_components.solar_forecast.weather_input._get_forecast", side_effect=_fake):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    reports = {r.entity_id: r for r in entry.runtime_data.data.sources}
    assert reports["weather.bad"].errors


async def test_removing_array_keeps_history(
    hass: HomeAssistant, setup_entry: MockConfigEntry, mock_weather
) -> None:
    """Removing an array keeps the frozen history of the remaining arrays."""
    from types import MappingProxyType

    from homeassistant.config_entries import ConfigSubentry

    from custom_components.solar_forecast.const import SUBENTRY_TYPE_ARRAY

    entry = setup_entry
    first_id = entry.runtime_data.array_ids[0]
    past_hour = dt_util.as_utc(NOON_PDT - timedelta(hours=2))

    west = ConfigSubentry(
        data=MappingProxyType({**entry.subentries[first_id].data, "azimuth": 270}),
        subentry_type=SUBENTRY_TYPE_ARRAY,
        title="West",
        unique_id=None,
    )
    hass.config_entries.async_add_subentry(entry, west)
    await hass.async_block_till_done()
    assert len(entry.runtime_data.array_ids) == 2
    first_value = entry.runtime_data.data.hourly_wh_arrays[first_id][past_hour]

    hass.config_entries.async_remove_subentry(entry, west.subentry_id)
    await hass.async_block_till_done()
    assert entry.runtime_data.array_ids == [first_id]
    # History was flushed on unload and reused for the remaining array.
    assert entry.runtime_data.data.hourly_wh[past_hour] == pytest.approx(first_value)


def test_daily_utc_midnight_maps_to_calendar_date(hass: HomeAssistant) -> None:
    """Daily entries stamped at 00:00 UTC belong to that calendar date."""
    from custom_components.solar_forecast.weather_input import daily_samples

    import zoneinfo

    dt_util.set_default_time_zone(zoneinfo.ZoneInfo("America/Los_Angeles"))
    samples = daily_samples(
        [{"datetime": "2026-11-25T00:00:00+00:00", "cloud_coverage": 50, "temperature": 10}],
        "°C",
        None,
    )
    first_local = dt_util.as_local(samples[0].time)
    assert first_local.date().isoformat() == "2026-11-25"
    assert first_local.hour == 0
