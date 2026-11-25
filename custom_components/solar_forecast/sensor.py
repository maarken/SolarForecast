"""Sensors for Solar Forecast."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfEnergy, UnitOfPower
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .coordinator import SolarForecastConfigEntry, SolarForecastCoordinator, SolarForecastData

PARALLEL_UPDATES = 0

type ValueFn = Callable[[SolarForecastData, datetime, str | None], Any]


@dataclass(frozen=True, kw_only=True)
class SolarForecastSensorDescription(SensorEntityDescription):
    """Describes a Solar Forecast sensor."""

    value_fn: ValueFn
    attr_fn: Callable[[SolarForecastData, datetime, str | None], dict[str, Any]] | None = None
    time_dependent: bool = False
    """Re-evaluate every minute (value depends on the current time)."""


def _wh_to_kwh(value: float | None) -> float | None:
    return None if value is None else value / 1000.0


def _local_day(now: datetime, offset: int):
    return dt_util.as_local(now).date() + timedelta(days=offset)


def _day_energy(offset: int) -> ValueFn:
    return lambda d, now, aid: _wh_to_kwh(d.energy_day(_local_day(now, offset), aid))


def _day_attrs(offset: int) -> Callable[[SolarForecastData, datetime, str | None], dict[str, Any]]:
    return lambda d, now, aid: {"forecast": d.hourly_list(_local_day(now, offset), aid)}


def _hour_window(now: datetime, offset: int) -> tuple[datetime, datetime]:
    """UTC bounds of the local clock hour ``offset`` hours from the current one.

    The local hour start handles UTC+x:30 zones; the arithmetic is done in
    UTC so DST transitions do not produce 2 hour windows.
    """
    start = dt_util.as_utc(dt_util.as_local(now).replace(minute=0, second=0, microsecond=0))
    start += timedelta(hours=offset)
    return start, start + timedelta(hours=1)


def _peak_power(offset: int) -> ValueFn:
    def fn(d: SolarForecastData, now: datetime, aid: str | None) -> float | None:
        peak = d.peak(_local_day(now, offset))
        return None if peak is None else round(peak.total)

    return fn


def _peak_time(offset: int) -> ValueFn:
    def fn(d: SolarForecastData, now: datetime, aid: str | None) -> datetime | None:
        peak = d.peak(_local_day(now, offset))
        if peak is None or peak.total <= 0:
            return None
        return peak.time

    return fn


def _energy_desc(key: str, value_fn: ValueFn, **kwargs: Any) -> SolarForecastSensorDescription:
    return SolarForecastSensorDescription(
        key=key,
        translation_key=key,
        device_class=SensorDeviceClass.ENERGY,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        suggested_display_precision=2,
        value_fn=value_fn,
        **kwargs,
    )


SITE_SENSORS: tuple[SolarForecastSensorDescription, ...] = (
    SolarForecastSensorDescription(
        key="power_production_now",
        translation_key="power_production_now",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        suggested_display_precision=0,
        value_fn=lambda d, now, aid: d.power_now(now, aid),
        time_dependent=True,
    ),
    _energy_desc(
        "energy_production_today",
        _day_energy(0),
        attr_fn=_day_attrs(0),
    ),
    _energy_desc(
        "energy_production_today_remaining",
        lambda d, now, aid: _wh_to_kwh(d.energy_remaining_today(now, aid)),
        time_dependent=True,
    ),
    _energy_desc(
        "energy_production_tomorrow",
        _day_energy(1),
        attr_fn=_day_attrs(1),
    ),
    *(
        _energy_desc(
            f"energy_production_d{offset}",
            _day_energy(offset),
            attr_fn=_day_attrs(offset),
            entity_registry_enabled_default=False,
        )
        for offset in range(2, 7)
    ),
    SolarForecastSensorDescription(
        key="energy_current_hour",
        translation_key="energy_current_hour",
        device_class=SensorDeviceClass.ENERGY,
        native_unit_of_measurement=UnitOfEnergy.WATT_HOUR,
        suggested_display_precision=0,
        value_fn=lambda d, now, aid: d.energy_between(*_hour_window(now, 0), aid),
        time_dependent=True,
    ),
    SolarForecastSensorDescription(
        key="energy_next_hour",
        translation_key="energy_next_hour",
        device_class=SensorDeviceClass.ENERGY,
        native_unit_of_measurement=UnitOfEnergy.WATT_HOUR,
        suggested_display_precision=0,
        value_fn=lambda d, now, aid: d.energy_between(*_hour_window(now, 1), aid),
        time_dependent=True,
    ),
    SolarForecastSensorDescription(
        key="power_highest_peak_today",
        translation_key="power_highest_peak_today",
        device_class=SensorDeviceClass.POWER,
        native_unit_of_measurement=UnitOfPower.WATT,
        suggested_display_precision=0,
        value_fn=_peak_power(0),
    ),
    SolarForecastSensorDescription(
        key="power_highest_peak_time_today",
        translation_key="power_highest_peak_time_today",
        device_class=SensorDeviceClass.TIMESTAMP,
        value_fn=_peak_time(0),
    ),
    SolarForecastSensorDescription(
        key="power_highest_peak_tomorrow",
        translation_key="power_highest_peak_tomorrow",
        device_class=SensorDeviceClass.POWER,
        native_unit_of_measurement=UnitOfPower.WATT,
        suggested_display_precision=0,
        value_fn=_peak_power(1),
    ),
    SolarForecastSensorDescription(
        key="power_highest_peak_time_tomorrow",
        translation_key="power_highest_peak_time_tomorrow",
        device_class=SensorDeviceClass.TIMESTAMP,
        value_fn=_peak_time(1),
    ),
    SolarForecastSensorDescription(
        key="clear_sky_power_now",
        translation_key="clear_sky_power_now",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        suggested_display_precision=0,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda d, now, aid: d.clear_sky_power_now(now),
        time_dependent=True,
    ),
    SolarForecastSensorDescription(
        key="cloud_cover_now",
        translation_key="cloud_cover_now",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=0,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda d, now, aid: (
            None if d.weather_now is None else d.weather_now.cloud_cover * 100.0
        ),
    ),
)

CALIBRATION_SENSORS: tuple[SolarForecastSensorDescription, ...] = (
    SolarForecastSensorDescription(
        key="calibration_factor",
        translation_key="calibration_factor",
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=3,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda d, now, aid: (
            None if d.calibration is None else d.calibration.get("clear_sky_factor")
        ),
        attr_fn=lambda d, now, aid: dict(d.calibration or {}),
    ),
    SolarForecastSensorDescription(
        key="forecast_error_day_ahead",
        translation_key="forecast_error_day_ahead",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=1,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda d, now, aid: (
            None if d.accuracy is None else d.accuracy.get("day_ahead_error")
        ),
        attr_fn=lambda d, now, aid: dict(d.accuracy or {}),
    ),
)

ARRAY_SENSORS: tuple[SolarForecastSensorDescription, ...] = (
    SolarForecastSensorDescription(
        key="power_production_now",
        translation_key="power_production_now",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        suggested_display_precision=0,
        value_fn=lambda d, now, aid: d.power_now(now, aid),
        time_dependent=True,
    ),
    _energy_desc("energy_production_today", _day_energy(0), attr_fn=_day_attrs(0)),
    _energy_desc(
        "energy_production_today_remaining",
        lambda d, now, aid: _wh_to_kwh(d.energy_remaining_today(now, aid)),
        time_dependent=True,
    ),
    _energy_desc("energy_production_tomorrow", _day_energy(1), attr_fn=_day_attrs(1)),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: SolarForecastConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up sensors for the site and for each array subentry."""
    coordinator = entry.runtime_data
    site_device = DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=entry.title,
        manufacturer="Solar Forecast",
        model="PV forecast model",
        entry_type=DeviceEntryType.SERVICE,
    )
    async_add_entities(
        SolarForecastSensor(coordinator, desc, f"{entry.entry_id}_{desc.key}", site_device, None)
        for desc in SITE_SENSORS
    )
    if coordinator.calibration.enabled:
        async_add_entities(
            SolarForecastSensor(
                coordinator, desc, f"{entry.entry_id}_{desc.key}", site_device, None
            )
            for desc in CALIBRATION_SENSORS
        )
    for subentry_id in coordinator.array_ids:
        subentry = entry.subentries[subentry_id]
        device = DeviceInfo(
            identifiers={(DOMAIN, subentry_id)},
            name=subentry.title,
            manufacturer="Solar Forecast",
            model="PV array",
            entry_type=DeviceEntryType.SERVICE,
            via_device=(DOMAIN, entry.entry_id),
        )
        async_add_entities(
            (
                SolarForecastSensor(
                    coordinator, desc, f"{subentry_id}_{desc.key}", device, subentry_id
                )
                for desc in ARRAY_SENSORS
            ),
            config_subentry_id=subentry_id,
        )


class SolarForecastSensor(CoordinatorEntity[SolarForecastCoordinator], SensorEntity):
    """A Solar Forecast sensor."""

    entity_description: SolarForecastSensorDescription
    _attr_has_entity_name = True
    _unrecorded_attributes = frozenset(
        {
            "forecast",
            "days",
            "cloud_factors",
            "cloud_factors_day_ahead",
            "production_sensor",
            "fitted_at",
            "last_error",
        }
    )

    def __init__(
        self,
        coordinator: SolarForecastCoordinator,
        description: SolarForecastSensorDescription,
        unique_id: str,
        device_info: DeviceInfo,
        array_id: str | None,
    ) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator)
        self.entity_description = description
        self._attr_unique_id = unique_id
        self._attr_device_info = device_info
        self._array_id = array_id

    async def async_added_to_hass(self) -> None:
        """Refresh time-dependent values every minute between data updates."""
        await super().async_added_to_hass()
        if self.entity_description.time_dependent:
            self.async_on_remove(
                async_track_time_interval(self.hass, self._async_tick, timedelta(minutes=1))
            )

    @callback
    def _async_tick(self, _now: datetime) -> None:
        if self.coordinator.data is not None:
            self.async_write_ha_state()

    @property
    def native_value(self) -> Any:
        """Return the forecast value."""
        data = self.coordinator.data
        if data is None:
            return None
        return self.entity_description.value_fn(data, dt_util.utcnow(), self._array_id)

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the hourly forecast where applicable."""
        data = self.coordinator.data
        if data is None or self.entity_description.attr_fn is None:
            return None
        return self.entity_description.attr_fn(data, dt_util.utcnow(), self._array_id)
