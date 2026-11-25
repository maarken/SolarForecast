"""Unit tests for the calibration fit (pure model, no Home Assistant)."""

from datetime import datetime, timedelta, timezone
import random

import pytest

from custom_components.solar_forecast.model.calibration import (
    IDENTITY,
    Calibration,
    HourRecord,
    fit_calibration,
    weighted_quantile,
)
from custom_components.solar_forecast.model.forecast import PVModel, Site
from custom_components.solar_forecast.model.linke import load_monthly_linke_turbidity
from custom_components.solar_forecast.model.pvsystem import ArrayConfig, InverterConfig

UTC = timezone.utc


def test_weighted_quantile() -> None:
    """Equal weights reproduce the Hazen quantile; weights shift it."""
    values = [1.0, 2.0, 3.0, 4.0]
    assert weighted_quantile(values, [1, 1, 1, 1], 0.5) == pytest.approx(2.5)
    assert weighted_quantile(values, [1, 1, 1, 1], 0.0) == 1.0
    assert weighted_quantile(values, [1, 1, 1, 1], 1.0) == 4.0
    assert weighted_quantile(values, [1, 1, 1, 10], 0.5) > 3.5
    assert weighted_quantile([5.0], [2.0], 0.7) == 5.0


def test_identity_and_interpolation() -> None:
    """Identity is neutral; beam shading interpolates, wraps and spares diffuse."""
    assert IDENTITY.factor(123.0, 45.0, 0.7, 0.5, 0.0) == 1.0
    cal = Calibration(global_factor=0.9, shading={(0, 2): 0.5, (23, 2): 0.7})
    # Bin centres: az 7.5 (bin 0) and 352.5 (bin 23); el 25 (bin 2).
    assert cal.beam_transmission(7.5, 25.0) == pytest.approx(0.5)
    assert cal.beam_transmission(352.5, 25.0) == pytest.approx(0.7)
    assert cal.beam_transmission(0.0, 25.0) == pytest.approx(0.6)  # across north
    assert cal.beam_transmission(180.0, 60.0) == pytest.approx(1.0)  # no data
    # Shading only affects the beam share.
    assert cal.system_factor(7.5, 25.0, 1.0) == pytest.approx(0.9 * 0.5)
    assert cal.system_factor(7.5, 25.0, 0.0) == pytest.approx(0.9)
    assert cal.system_factor(7.5, 25.0, 0.6) == pytest.approx(0.9 * (0.4 + 0.6 * 0.5))


def test_cloud_curve_lead_time_blend() -> None:
    """Nowcast curve applies at short leads, day-ahead curve at long leads."""
    cal = Calibration(
        cloud_factors=tuple(1.0 + 0.1 * i for i in range(10)),
        cloud_factors_day_ahead=(2.0,) * 10,
    )
    assert cal.cloud_factor(0.0, 0.0) == pytest.approx(1.0)
    assert cal.cloud_factor(0.1, 1.0) == pytest.approx(1.05)
    assert cal.cloud_factor(1.0, 2.0) == pytest.approx(1.9)
    assert cal.cloud_factor(0.0, 24.0) == pytest.approx(2.0)
    assert cal.cloud_factor(0.0, 7.5) == pytest.approx(1.5)  # halfway 3..12 h


def _model() -> PVModel:
    tl = tuple(load_monthly_linke_turbidity(48.75, -122.48))
    return PVModel(
        Site(48.75, -122.48, 30.0, tl), [ArrayConfig("S", 30, 180, 6000)], InverterConfig()
    )


def _hour_wh(fn, start: datetime) -> float:
    return (
        sum(
            (0.5 if k in (0, 12) else 1.0) * fn(start + timedelta(minutes=5 * k)) for k in range(13)
        )
        / 12
    )


def _records(model: PVModel, truth, days: int, seed: int = 1) -> list[HourRecord]:
    rng = random.Random(seed)
    start = datetime(2026, 5, 20, tzinfo=UTC)
    out = []
    for d in range(days):
        day_cloud = rng.choice([0.0, 0.1, 0.5, 0.9])
        for h in range(13, 24):  # daylight in UTC for this site (06-17 local)
            t = start + timedelta(days=d, hours=h)
            cloud = min(max(day_cloud + rng.uniform(-0.05, 0.05), 0.0), 1.0)
            out.append(HourRecord(t, cloud, 18.0, _hour_wh(lambda x, c=cloud: truth(x, c), t)))
    return out


def test_fit_perfect_model_is_near_identity() -> None:
    """When measurements equal the model, the calibration stays neutral."""
    model = _model()
    recs = _records(model, lambda t, c: model.evaluate(t, c, 18.0).total, days=12)
    cal = fit_calibration(model, recs, recs[-1].start)
    assert cal.global_factor == pytest.approx(1.0, abs=0.01)
    assert all(f == pytest.approx(1.0, abs=0.02) for f in cal.cloud_factors)
    assert not cal.shading or all(v > 0.95 for v in cal.shading.values())


def test_fit_recovers_losses_and_beam_shading() -> None:
    """A system with 85 % output and a morning obstruction of the beam is learnt."""
    model = _model()

    def truth(t: datetime, c: float) -> float:
        ev = model.evaluate(t, c, 18.0)
        dc = ev.dc_total * 0.85
        if 60 <= ev.azimuth <= 100 and ev.elevation < 30:
            dc *= 1.0 - ev.beam_fraction  # beam fully blocked, diffuse unaffected
        from custom_components.solar_forecast.model.pvsystem import inverter_ac_power

        return inverter_ac_power(model.inverter, dc, model.nameplate)

    recs = _records(model, truth, days=25)
    cal = fit_calibration(model, recs, recs[-1].start)
    assert cal.global_factor == pytest.approx(0.85, abs=0.03)
    assert cal.beam_transmission(80.0, 15.0) < 0.7  # coarse 15x10 deg bins blur edges
    assert cal.beam_transmission(180.0, 60.0) == pytest.approx(1.0, abs=0.05)
    # Under overcast the shaded direction keeps (almost) the system factor.
    assert cal.system_factor(80.0, 15.0, 0.05) > 0.8


def test_fit_rejects_snow_and_curtailment_outliers() -> None:
    """A week of snow-covered panels does not drag the cloud response down."""
    model = _model()
    clean = _records(model, lambda t, c: model.evaluate(t, c, 18.0).total, days=30)
    snowy_start = clean[-1].start - timedelta(days=10)
    recs = [
        HourRecord(r.start, r.cloud_cover, -2.0, r.actual_wh * 0.15)
        if snowy_start <= r.start < snowy_start + timedelta(days=7)
        else r
        for r in clean
    ]
    cal = fit_calibration(model, recs, recs[-1].start)
    assert cal.rejected_hours > 30
    assert cal.global_factor == pytest.approx(1.0, abs=0.02)
    assert all(f == pytest.approx(1.0, abs=0.05) for f in cal.cloud_factors)


def test_fit_ignores_outages_and_low_sun() -> None:
    """Zero production (outage) hours do not drag the factor down."""
    model = _model()
    recs = _records(model, lambda t, c: model.evaluate(t, c, 18.0).total, days=10)
    outage = [HourRecord(r.start, r.cloud_cover, r.temperature, 0.0) for r in recs[:40]]
    cal = fit_calibration(model, outage + recs[40:], recs[-1].start)
    assert cal.global_factor == pytest.approx(1.0, abs=0.01)


def test_fit_no_data() -> None:
    """No records gives an inactive calibration."""
    cal = fit_calibration(_model(), [], datetime(2026, 6, 1, tzinfo=UTC))
    assert cal.global_factor == 1.0
    assert not cal.active
