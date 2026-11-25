"""Weather time series used to drive the PV model (Home Assistant independent)."""

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import math

# Interpolating across a larger hole than this is not trusted.
MAX_INTERPOLATION_GAP = timedelta(hours=6)


@dataclass(frozen=True, slots=True)
class WeatherSample:
    """A forecast point. ``cloud_cover`` is a fraction 0..1, temperature degC."""

    time: datetime
    cloud_cover: float | None
    temperature: float | None


@dataclass(frozen=True, slots=True)
class WeatherAt:
    """Weather conditions interpolated to an instant."""

    cloud_cover: float
    temperature: float


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class _Channel:
    """One interpolated quantity, ignoring samples where it is missing."""

    __slots__ = ("_times", "_values")

    def __init__(self, pairs: Iterable[tuple[datetime, float]]) -> None:
        # Normalise to UTC: arithmetic between datetimes sharing a ZoneInfo
        # tzinfo is wall-clock based and breaks across DST transitions.
        merged: dict[datetime, float] = {}
        for when, value in pairs:
            merged[when.astimezone(timezone.utc)] = value
        ordered = sorted(merged.items(), key=lambda p: p[0])
        self._times = [p[0] for p in ordered]
        self._values = [p[1] for p in ordered]

    @property
    def first(self) -> datetime | None:
        return self._times[0] if self._times else None

    @property
    def last(self) -> datetime | None:
        return self._times[-1] if self._times else None

    def at(self, when: datetime, hold_before_start: bool) -> float | None:
        when = when.astimezone(timezone.utc)
        times = self._times
        if not times:
            return None
        if when < times[0]:
            return self._values[0] if hold_before_start else None
        if when == times[-1]:
            return self._values[-1]
        if when > times[-1]:
            return None
        i = bisect_right(times, when)
        t0, t1 = times[i - 1], times[i]
        if when == t0:
            return self._values[i - 1]
        if t1 - t0 > MAX_INTERPOLATION_GAP:
            return None
        frac = (when - t0) / (t1 - t0)
        return _lerp(self._values[i - 1], self._values[i], frac)


class WeatherSeries:
    """Linearly interpolated weather forecast from one source."""

    def __init__(self, samples: Sequence[WeatherSample]) -> None:
        self._cloud = _Channel(
            (s.time, min(max(s.cloud_cover, 0.0), 1.0))
            for s in samples
            if s.cloud_cover is not None and math.isfinite(s.cloud_cover)
        )
        self._temp = _Channel(
            (s.time, s.temperature)
            for s in samples
            if s.temperature is not None and math.isfinite(s.temperature)
        )

    @property
    def start(self) -> datetime | None:
        """First time with cloud data (cloud cover is the essential input)."""
        return self._cloud.first

    @property
    def end(self) -> datetime | None:
        """Last time with cloud data."""
        return self._cloud.last

    def cloud_at(self, when: datetime, hold_before_start: bool = False) -> float | None:
        """Cloud cover fraction at ``when``."""
        return self._cloud.at(when, hold_before_start)

    def temperature_at(self, when: datetime, hold_before_start: bool = False) -> float | None:
        """Air temperature at ``when``; held at the ends if cloud data exists."""
        value = self._temp.at(when, hold_before_start)
        if value is None and self._temp.first is not None and self._temp.last is not None:
            # Temperature is a second-order input; hold the nearest endpoint
            # rather than dropping a time step that has cloud data.
            if when > self._temp.last:
                value = self._temp.at(self._temp.last, False)
            elif when < self._temp.first:
                value = self._temp.at(self._temp.first, False)
        return value


class EnsembleWeather:
    """Average of several weather sources (e.g. Pirate Weather + OpenWeatherMap).

    At each instant the mean is taken over the sources that have data, so a
    source with a shorter horizon does not truncate the forecast.
    """

    def __init__(self, sources: Sequence[WeatherSeries], default_temperature: float = 15.0) -> None:
        self._sources = [s for s in sources if s.start is not None]
        self._default_temperature = default_temperature

    @property
    def start(self) -> datetime | None:
        starts = [s.start for s in self._sources if s.start is not None]
        return min(starts) if starts else None

    @property
    def end(self) -> datetime | None:
        ends = [s.end for s in self._sources if s.end is not None]
        return max(ends) if ends else None

    def at(self, when: datetime, hold_before_start: bool = False) -> WeatherAt | None:
        clouds: list[float] = []
        temps: list[float] = []
        for src in self._sources:
            cloud = src.cloud_at(when, hold_before_start)
            if cloud is None:
                continue
            clouds.append(cloud)
            temp = src.temperature_at(when, hold_before_start)
            if temp is not None:
                temps.append(temp)
        if not clouds:
            return None
        temperature = sum(temps) / len(temps) if temps else self._default_temperature
        return WeatherAt(cloud_cover=sum(clouds) / len(clouds), temperature=temperature)


def daily_to_hourly(
    day_start: datetime,
    cloud_cover: float | None,
    temp_high: float | None,
    temp_low: float | None,
) -> list[WeatherSample]:
    """Expand one daily forecast into hourly samples covering the local day.

    ``day_start`` must be local midnight with a real time zone (ZoneInfo).
    Samples are produced every real hour (UTC arithmetic) from local midnight
    to the next local midnight inclusive, so DST days yield 24 or 26 samples.
    Cloud cover is held constant. Temperature follows a simple diurnal cycle
    with the minimum at 06:00 and maximum at 15:00 local time; temperature has
    only a ~0.4 %/K effect on output, so this approximation is adequate.
    """
    tz = day_start.tzinfo
    if tz is None:
        raise ValueError("day_start must be timezone aware")
    next_day = datetime.combine(day_start.date() + timedelta(days=1), time(0), tzinfo=tz)
    start_utc = day_start.astimezone(timezone.utc)
    end_utc = next_day.astimezone(timezone.utc)

    if temp_high is not None and temp_low is None:
        temp_low = temp_high - 8.0
    samples: list[WeatherSample] = []
    when = start_utc
    while when <= end_utc:
        temp: float | None = None
        if temp_high is not None and temp_low is not None:
            local = when.astimezone(tz)
            hour = local.hour + local.minute / 60.0
            if when == end_utc:
                hour = 24.0
            if 6.0 <= hour <= 15.0:
                frac = 0.5 - 0.5 * math.cos(math.pi * (hour - 6.0) / 9.0)
            else:
                h = hour if hour > 15.0 else hour + 24.0
                frac = 0.5 + 0.5 * math.cos(math.pi * (h - 15.0) / 15.0)
            temp = temp_low + (temp_high - temp_low) * frac
        samples.append(WeatherSample(time=when, cloud_cover=cloud_cover, temperature=temp))
        when += timedelta(hours=1)
    return samples
