"""Solar position, extraterrestrial irradiance and atmosphere helpers.

Pure-Python (stdlib only) so the model can run inside Home Assistant without
numpy/pandas and can be unit tested in isolation.

Solar position follows the NOAA solar calculator (Meeus, "Astronomical
Algorithms"), accurate to roughly 0.01 deg for years 1901-2099, which is far
below the error budget of a weather-driven PV forecast.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import math

_DEG = math.pi / 180.0
_RAD = 180.0 / math.pi


@dataclass(frozen=True, slots=True)
class SolarPosition:
    """Sun position for an observer.

    Angles are in degrees. Azimuth is measured clockwise from true north
    (0 = N, 90 = E, 180 = S, 270 = W).
    """

    zenith: float
    """Geometric (true) zenith angle, without refraction."""
    apparent_zenith: float
    """Zenith angle corrected for atmospheric refraction."""
    azimuth: float
    declination: float
    equation_of_time: float
    """Equation of time in minutes."""

    @property
    def apparent_elevation(self) -> float:
        """Apparent solar elevation angle in degrees."""
        return 90.0 - self.apparent_zenith


def julian_day(when: datetime) -> float:
    """Return the Julian day (UT) for a timezone-aware datetime."""
    if when.tzinfo is None:
        raise ValueError("datetime must be timezone aware")
    return when.timestamp() / 86400.0 + 2440587.5


def _refraction_correction(elevation: float) -> float:
    """Atmospheric refraction in degrees for a true elevation (NOAA)."""
    if elevation > 85.0:
        return 0.0
    if elevation > 5.0:
        te = math.tan(elevation * _DEG)
        arcsec = 58.1 / te - 0.07 / te**3 + 0.000086 / te**5
    elif elevation > -0.575:
        arcsec = 1735.0 + elevation * (
            -518.2 + elevation * (103.4 + elevation * (-12.79 + elevation * 0.711))
        )
    else:
        arcsec = -20.772 / math.tan(elevation * _DEG)
    return arcsec / 3600.0


def solar_position(
    when: datetime,
    latitude: float,
    longitude: float,
    pressure: float = 101325.0,
    temperature: float = 10.0,
) -> SolarPosition:
    """Compute the sun position for ``when`` at the given location.

    ``latitude`` is positive north, ``longitude`` positive east, both in
    degrees. ``pressure`` (Pa) and ``temperature`` (degC) scale the
    refraction correction (Bennett/Meeus factor P/1010 hPa * 283/(273+T)),
    which matters near the horizon at high-altitude sites.
    """
    jd = julian_day(when)
    jc = (jd - 2451545.0) / 36525.0

    geom_mean_long = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    geom_mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eccent = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    m_rad = geom_mean_anom * _DEG
    eq_ctr = (
        math.sin(m_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + math.sin(2.0 * m_rad) * (0.019993 - 0.000101 * jc)
        + math.sin(3.0 * m_rad) * 0.000289
    )
    true_long = geom_mean_long + eq_ctr
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * math.sin(omega * _DEG)

    mean_obliq = (
        23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    )
    obliq_corr = mean_obliq + 0.00256 * math.cos(omega * _DEG)

    declination = math.asin(math.sin(obliq_corr * _DEG) * math.sin(app_long * _DEG))

    var_y = math.tan(obliq_corr * _DEG / 2.0) ** 2
    l0 = geom_mean_long * _DEG
    eq_time = (
        4.0
        * _RAD
        * (
            var_y * math.sin(2.0 * l0)
            - 2.0 * eccent * math.sin(m_rad)
            + 4.0 * eccent * var_y * math.sin(m_rad) * math.cos(2.0 * l0)
            - 0.5 * var_y * var_y * math.sin(4.0 * l0)
            - 1.25 * eccent * eccent * math.sin(2.0 * m_rad)
        )
    )

    utc = when.astimezone(timezone.utc)
    minutes_utc = utc.hour * 60.0 + utc.minute + (utc.second + utc.microsecond / 1e6) / 60.0
    true_solar_time = (minutes_utc + eq_time + 4.0 * longitude) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0  # degrees, negative before solar noon

    lat = latitude * _DEG
    ha = hour_angle * _DEG
    cos_zen = math.sin(lat) * math.sin(declination) + math.cos(lat) * math.cos(
        declination
    ) * math.cos(ha)
    cos_zen = max(-1.0, min(1.0, cos_zen))
    zenith = math.acos(cos_zen) * _RAD

    # Azimuth clockwise from north. atan2 form is well defined everywhere
    # except exactly at the poles / zenith, where azimuth is meaningless.
    azimuth = (
        math.atan2(
            math.sin(ha),
            math.cos(ha) * math.sin(lat) - math.tan(declination) * math.cos(lat),
        )
        * _RAD
        + 180.0
    ) % 360.0

    elevation = 90.0 - zenith
    refraction_scale = (pressure / 101000.0) * (283.0 / (273.0 + temperature))
    apparent_zenith = zenith - _refraction_correction(elevation) * refraction_scale

    return SolarPosition(
        zenith=zenith,
        apparent_zenith=apparent_zenith,
        azimuth=azimuth,
        declination=declination * _RAD,
        equation_of_time=eq_time,
    )


def extraterrestrial_irradiance(when: datetime, solar_constant: float = 1366.1) -> float:
    """Extraterrestrial normal irradiance (W/m2), Spencer (1971)."""
    doy = when.astimezone(timezone.utc).timetuple().tm_yday
    b = 2.0 * math.pi * (doy - 1) / 365.0
    rover_r0_sqrd = (
        1.00011
        + 0.034221 * math.cos(b)
        + 0.00128 * math.sin(b)
        + 0.000719 * math.cos(2.0 * b)
        + 0.000077 * math.sin(2.0 * b)
    )
    return solar_constant * rover_r0_sqrd


def relative_airmass(zenith: float) -> float | None:
    """Relative optical air mass, Kasten & Young (1989).

    ``zenith`` should be the apparent zenith in degrees. Returns ``None`` when
    the sun is below the horizon.
    """
    if zenith >= 90.0:
        return None
    return 1.0 / (math.cos(zenith * _DEG) + 0.50572 * (96.07995 - zenith) ** -1.6364)


def relative_airmass_kasten1966(zenith: float) -> float | None:
    """Relative air mass, Kasten (1966); used by the DISC model."""
    if zenith >= 90.0:
        return None
    return 1.0 / (math.cos(zenith * _DEG) + 0.15 * (93.885 - zenith) ** -1.253)


def altitude_to_pressure(altitude: float) -> float:
    """Standard-atmosphere surface pressure (Pa) for an altitude in metres."""
    return 100.0 * ((44331.514 - altitude) / 11880.516) ** (1.0 / 0.1902632)
