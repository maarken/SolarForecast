"""Pure model tests against reference values computed with pvlib 0.16.1.

Reference values were produced with pvlib (SPA solar position, spencer
extraterrestrial, kastenyoung1989 airmass, ineichen, disc, perez
allsitescomposite1990, martin_ruiz IAMs, pvsyst_cell, pvwatts inverter).
"""

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from custom_components.solar_forecast.model.forecast import PVModel, Site, compute_forecast
from custom_components.solar_forecast.model.irradiance import (
    disc_dni,
    iam_martin_ruiz,
    iam_martin_ruiz_diffuse,
    ineichen_clearsky,
    perez_sky_diffuse,
)
from custom_components.solar_forecast.model.linke import (
    interpolate_linke_turbidity,
    load_monthly_linke_turbidity,
)
from custom_components.solar_forecast.model.pvsystem import (
    ArrayConfig,
    InverterConfig,
    cell_temperature_pvsyst,
    inverter_ac_power,
)
from custom_components.solar_forecast.model.solar import (
    altitude_to_pressure,
    extraterrestrial_irradiance,
    relative_airmass,
    solar_position,
)
from custom_components.solar_forecast.model.weather import (
    EnsembleWeather,
    WeatherSample,
    WeatherSeries,
    daily_to_hourly,
)

UTC = timezone.utc


@pytest.mark.parametrize(
    ("ts", "lat", "lon", "alt", "zenith", "app_zenith", "azimuth"),
    [
        ("2026-06-21T20:00:00+00:00", 48.75, -122.48, 30, 25.421279, 25.413529, 173.679240),
        ("2026-12-21T02:00:00+00:00", -33.87, 151.21, 50, 10.546631, 10.543618, 351.217065),
        ("2026-03-20T15:00:00+00:00", -16.5, -68.15, 3640, 29.662421, 29.656479, 58.648815),
    ],
)
def test_solar_position_vs_spa(ts, lat, lon, alt, zenith, app_zenith, azimuth) -> None:
    """NOAA algorithm agrees with NREL SPA to ~0.02 deg."""
    pos = solar_position(datetime.fromisoformat(ts), lat, lon, altitude_to_pressure(alt), 20.0)
    assert pos.zenith == pytest.approx(zenith, abs=0.02)
    assert pos.apparent_zenith == pytest.approx(app_zenith, abs=0.02)
    assert pos.azimuth == pytest.approx(azimuth, abs=0.05)


def test_clear_sky_chain_vs_pvlib() -> None:
    """Extraterrestrial, airmass and Ineichen match pvlib to float precision."""
    dni_extra = extraterrestrial_irradiance(datetime(2026, 6, 21, tzinfo=UTC))
    assert dni_extra == pytest.approx(1321.6235925714136, rel=1e-12)
    am = relative_airmass(40.0)
    assert am == pytest.approx(1.304223540913535, rel=1e-12)
    am_abs = am * altitude_to_pressure(500) / 101325
    cs = ineichen_clearsky(40.0, am_abs, 3.2, 500, dni_extra)
    assert cs.ghi == pytest.approx(760.8979333049784, rel=1e-9)
    assert cs.dni == pytest.approx(867.8389382988306, rel=1e-9)
    assert cs.dhi == pytest.approx(96.09473709888562, rel=1e-9)


def test_disc_perez_iam_temperature_inverter_vs_pvlib() -> None:
    """Component models match pvlib."""
    doy = date(2026, 4, 10).timetuple().tm_yday
    assert disc_dni(450.0, 60.0, doy, 95000) == pytest.approx(680.2653338745577, rel=1e-9)

    dni_extra = 1321.6235925714136
    am = 1.304223540913535
    sky = perez_sky_diffuse(35, 200, 150.0, 600.0, dni_extra, 40.0, 160.0, am)
    assert sky == pytest.approx(181.69798936267057, rel=1e-9)

    assert iam_martin_ruiz(0.6427876096865394) == pytest.approx(0.9839000766691123, rel=1e-9)
    iam_sky, iam_gnd = iam_martin_ruiz_diffuse(35)
    assert iam_sky == pytest.approx(0.9557027503425566, rel=1e-9)
    assert iam_gnd == pytest.approx(0.8325568553615091, rel=1e-9)

    assert cell_temperature_pvsyst(800, 25, u_c=20, module_efficiency=0.2) == pytest.approx(53.8)
    inv = InverterConfig(ac_power_w=4600, efficiency_nominal=0.96)
    assert inverter_ac_power(inv, 3000, 5000) == pytest.approx(2887.572156227583, rel=1e-9)
    assert inverter_ac_power(inv, 10000, 5000) == 4600  # clipped


def test_iam_diffuse_tilt_180_does_not_crash() -> None:
    """Regression: 1 + cos(pi) == 0 used to raise ZeroDivisionError."""
    iam_sky, iam_gnd = iam_martin_ruiz_diffuse(180.0)
    assert 0.0 <= iam_sky <= 1.0
    assert 0.0 <= iam_gnd <= 1.0


def test_linke_table_known_cell() -> None:
    """Table loads and gives plausible seasonal turbidity."""
    tl = load_monthly_linke_turbidity(48.75, -122.48)
    assert len(tl) == 12
    assert all(1.5 < v < 6 for v in tl)
    assert min(tl[5:8]) > min(tl[0], tl[11])  # hazier in summer
    mid_jan = interpolate_linke_turbidity(tl, date(2026, 1, 16), 0.0)
    assert mid_jan == pytest.approx(tl[0], abs=0.02)


def _model(arrays=None, inverter=None) -> PVModel:
    tl = tuple(load_monthly_linke_turbidity(48.75, -122.48))
    return PVModel(
        Site(48.75, -122.48, 30.0, tl),
        arrays or [ArrayConfig("S", 30, 180, 5000)],
        inverter or InverterConfig(),
    )


def _flat_weather(start: datetime, hours: int, cloud: float) -> EnsembleWeather:
    return EnsembleWeather(
        [
            WeatherSeries(
                [WeatherSample(start + timedelta(hours=h), cloud, 20.0) for h in range(hours + 1)]
            )
        ]
    )


def test_power_at_respects_gaps() -> None:
    """Regression: power_at used positional indexing and ignored gaps."""
    t0 = datetime(2025, 6, 21, 12, tzinfo=UTC)
    samples = [WeatherSample(t0 + timedelta(hours=h), 0.0, 20.0) for h in range(4)]
    samples += [WeatherSample(t0 + timedelta(hours=h), 0.0, 20.0) for h in range(11, 16)]
    weather = EnsembleWeather([WeatherSeries(samples)])
    model = _model()
    result = compute_forecast(model, weather, t0, t0 + timedelta(hours=15))
    # Inside the >6 h gap there is no data.
    assert result.power_at(t0 + timedelta(hours=6)) is None
    # After the gap the value matches a direct evaluation.
    when = t0 + timedelta(hours=12)
    direct = model.evaluate(when, 0.0, 20.0).total
    assert result.power_at(when).total == pytest.approx(direct)
    # Hours touching the gap are not reported.
    assert t0 + timedelta(hours=5) not in result.hourly_wh


def test_hourly_energy_matches_fine_integration() -> None:
    """5 minute trapezoid integration is accurate versus 30 s integration."""
    t0 = datetime(2026, 6, 21, 7, tzinfo=UTC)
    model = _model()
    weather = _flat_weather(t0, 24, 0.2)
    result = compute_forecast(model, weather, t0, t0 + timedelta(hours=24))
    hour = datetime(2026, 6, 21, 13, tzinfo=UTC)  # ~06:00 local, steep ramp
    fine = (
        sum(model.evaluate(hour + timedelta(seconds=30 * k + 15), 0.2, 20.0)[0] for k in range(120))
        / 120.0
    )
    assert result.hourly_wh[hour] == pytest.approx(fine, rel=0.01, abs=1.0)


def test_cloud_monotonic_and_bounded() -> None:
    """Output decreases with cloud and never exceeds the inverter rating."""
    t0 = datetime(2026, 6, 21, 7, tzinfo=UTC)
    model = _model(inverter=InverterConfig(ac_power_w=3500))
    totals = []
    for cloud in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = compute_forecast(model, _flat_weather(t0, 24, cloud), t0, t0 + timedelta(hours=24))
        assert max(s.total for s in r.samples) <= 3500 + 1e-9
        assert min(s.total for s in r.samples) >= 0.0
        totals.append(sum(r.hourly_wh.values()))
    assert totals == sorted(totals, reverse=True)
    assert 0.15 < totals[-1] / totals[0] < 0.4


def test_orientation_symmetry() -> None:
    """East and west arrays produce similar daily energy at the solstice."""
    t0 = datetime(2026, 6, 21, 7, tzinfo=UTC)
    weather = _flat_weather(t0, 24, 0.0)
    east = compute_forecast(
        _model([ArrayConfig("E", 30, 90, 5000)]), weather, t0, t0 + timedelta(hours=24)
    )
    west = compute_forecast(
        _model([ArrayConfig("W", 30, 270, 5000)]), weather, t0, t0 + timedelta(hours=24)
    )
    e, w = sum(east.hourly_wh.values()), sum(west.hourly_wh.values())
    assert e == pytest.approx(w, rel=0.03)


def test_interpolation_across_dst_fall_back() -> None:
    """Regression: same-ZoneInfo arithmetic is wall clock and broke on DST."""
    tz = ZoneInfo("America/Los_Angeles")
    t0 = datetime(2025, 11, 2, 1, 0, tzinfo=tz)  # 01:00 PDT = 08:00Z
    t1 = datetime(2025, 11, 2, 2, 0, tzinfo=tz)  # 02:00 PST = 10:00Z
    series = WeatherSeries([WeatherSample(t0, 0.0, 10.0), WeatherSample(t1, 1.0, 10.0)])
    mid = datetime(2025, 11, 2, 9, 0, tzinfo=UTC)
    assert series.cloud_at(mid) == pytest.approx(0.5)
    late = datetime(2025, 11, 2, 9, 59, tzinfo=UTC)
    assert series.cloud_at(late) <= 1.0


def test_daily_to_hourly_dst_days() -> None:
    """Daily expansion spans the real local day on DST transition days."""
    tz = ZoneInfo("America/Los_Angeles")
    fall = daily_to_hourly(datetime(2025, 11, 2, tzinfo=tz), 0.5, 20.0, 10.0)
    spring = daily_to_hourly(datetime(2025, 3, 9, tzinfo=tz), 0.5, 20.0, 10.0)
    assert len(fall) == 26  # 25 hour day, inclusive endpoints
    assert len(spring) == 24  # 23 hour day
    for samples in (fall, spring):
        times = [s.time for s in samples]
        assert len(set(times)) == len(times)
        temps = [s.temperature for s in samples]
        assert max(temps) == pytest.approx(20.0, abs=0.5)
        assert min(temps) >= 10.0 - 1e-9


def test_ensemble_averages_and_extends() -> None:
    """Two sources are averaged; the longer one extends the horizon."""
    t0 = datetime(2026, 6, 21, tzinfo=UTC)
    a = WeatherSeries([WeatherSample(t0 + timedelta(hours=h), 0.2, 10.0) for h in range(10)])
    b = WeatherSeries([WeatherSample(t0 + timedelta(hours=h), 0.6, 20.0) for h in range(20)])
    ens = EnsembleWeather([a, b])
    mid = ens.at(t0 + timedelta(hours=5))
    assert mid.cloud_cover == pytest.approx(0.4)
    assert mid.temperature == pytest.approx(15.0)
    late = ens.at(t0 + timedelta(hours=15))
    assert late.cloud_cover == pytest.approx(0.6)
    assert ens.at(t0 + timedelta(hours=25)) is None
