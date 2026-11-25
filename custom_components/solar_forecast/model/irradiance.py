"""Irradiance models: clear sky, cloud attenuation, decomposition, transposition.

All functions work on scalars and return W/m2. Implementations mirror the
corresponding pvlib functions so they can be validated against pvlib:

* :func:`ineichen_clearsky`    -> ``pvlib.clearsky.ineichen``
* :func:`disc_dni`             -> ``pvlib.irradiance.disc``
* :func:`perez_sky_diffuse`    -> ``pvlib.irradiance.perez`` (allsitescomposite1990)
* :func:`cloud_cover_to_ghi`   -> Kasten & Czeplak (1980)
"""

from dataclasses import dataclass
import math

from .solar import relative_airmass_kasten1966

_DEG = math.pi / 180.0


@dataclass(frozen=True, slots=True)
class Irradiance:
    """Horizontal irradiance components (W/m2)."""

    ghi: float
    dni: float
    dhi: float


ZERO_IRRADIANCE = Irradiance(0.0, 0.0, 0.0)


def ineichen_clearsky(
    apparent_zenith: float,
    airmass_absolute: float | None,
    linke_turbidity: float,
    altitude: float,
    dni_extra: float,
) -> Irradiance:
    """Ineichen & Perez (2002) clear-sky model, as implemented by pvlib."""
    if apparent_zenith >= 90.0 or airmass_absolute is None:
        return ZERO_IRRADIANCE

    cos_zenith = max(math.cos(apparent_zenith * _DEG), 0.0)
    tl = linke_turbidity

    fh1 = math.exp(-altitude / 8000.0)
    fh2 = math.exp(-altitude / 1250.0)
    cg1 = 5.09e-05 * altitude + 0.868
    cg2 = 3.92e-05 * altitude + 0.0387

    ghi = math.exp(-cg2 * airmass_absolute * (fh1 + fh2 * (tl - 1.0)))
    ghi = cg1 * dni_extra * cos_zenith * max(ghi, 0.0)

    b = 0.664 + 0.163 / fh1
    bnci = dni_extra * max(b * math.exp(-0.09 * airmass_absolute * (tl - 1.0)), 0.0)

    if cos_zenith > 0.0:
        bnci_2 = (1.0 - (0.1 - 0.2 * math.exp(-tl)) / (0.1 + 0.882 / fh1)) / cos_zenith
        bnci_2 = ghi * min(max(bnci_2, 0.0), 1e20)
    else:
        bnci_2 = 0.0

    dni = min(bnci, bnci_2)
    dhi = ghi - dni * cos_zenith
    return Irradiance(ghi=ghi, dni=dni, dhi=dhi)


def cloud_cover_to_ghi(
    ghi_clear: float, cloud_cover: float, a: float = 0.75, b: float = 3.4
) -> float:
    """Global horizontal irradiance under cloud, Kasten & Czeplak (1980).

    ``cloud_cover`` is a fraction in [0, 1]. ``GHI = GHI_clear * (1 - a*C^b)``.
    """
    c = min(max(cloud_cover, 0.0), 1.0)
    return ghi_clear * (1.0 - a * c**b)


def clearness_index(
    ghi: float,
    zenith: float,
    dni_extra: float,
    min_cos_zenith: float = 0.065,
    max_clearness_index: float = 2.0,
) -> float:
    """Clearness index kt = GHI / (I0 * cos(zenith)), as in pvlib."""
    cos_zenith = max(math.cos(zenith * _DEG), min_cos_zenith)
    kt = ghi / (dni_extra * cos_zenith)
    return min(max(kt, 0.0), max_clearness_index)


def disc_dni(
    ghi: float,
    zenith: float,
    day_of_year: int,
    pressure: float | None = 101325.0,
    min_cos_zenith: float = 0.065,
    max_zenith: float = 87.0,
    max_airmass: float = 12.0,
) -> float:
    """Direct normal irradiance from GHI, DISC model (Maxwell 1987).

    Mirrors ``pvlib.irradiance.disc``. Note that DISC uses its own
    extraterrestrial irradiance with a 1370 W/m2 solar constant.
    """
    b = 2.0 * math.pi * (day_of_year - 1) / 365.0
    i0 = 1370.0 * (
        1.00011
        + 0.034221 * math.cos(b)
        + 0.00128 * math.sin(b)
        + 0.000719 * math.cos(2.0 * b)
        + 0.000077 * math.sin(2.0 * b)
    )
    if zenith > max_zenith or ghi < 0.0:
        return 0.0

    kt = clearness_index(ghi, zenith, i0, min_cos_zenith, max_clearness_index=1.0)

    am = relative_airmass_kasten1966(zenith)
    if am is None:
        return 0.0
    if pressure is not None:
        am = am * pressure / 101325.0
    am = min(am, max_airmass)

    kt2 = kt * kt
    kt3 = kt2 * kt
    if kt <= 0.6:
        a = 0.512 - 1.56 * kt + 2.286 * kt2 - 2.222 * kt3
        bb = 0.37 + 0.962 * kt
        c = -0.28 + 0.932 * kt - 2.048 * kt2
    else:
        a = -5.743 + 21.77 * kt - 27.49 * kt2 + 11.56 * kt3
        bb = 41.4 - 118.5 * kt + 66.05 * kt2 + 31.9 * kt3
        c = -47.01 + 184.2 * kt - 222.0 * kt2 + 73.81 * kt3

    delta_kn = a + bb * math.exp(c * am)
    knc = 0.866 - 0.122 * am + 0.0121 * am**2 - 0.000653 * am**3 + 1.4e-05 * am**4
    kn = knc - delta_kn
    dni = kn * i0
    return max(dni, 0.0)


def cloudy_sky_irradiance(
    clear: Irradiance,
    cloud_cover: float,
    zenith: float,
    day_of_year: int,
    pressure: float,
    true_zenith: float | None = None,
) -> Irradiance:
    """All-sky irradiance from clear-sky irradiance and cloud cover fraction.

    GHI comes from Kasten-Czeplak. The beam is the clear-sky (Ineichen) DNI
    scaled by the relative beam attenuation DISC predicts between the
    clear-sky GHI and the cloudy GHI:

        DNI = DNI_clear * DISC(GHI) / DISC(GHI_clear)

    This reproduces the Ineichen beam/diffuse split exactly at zero cloud
    (so there is no discontinuity versus the clear-sky forecast), while
    using the empirical DISC relation to move energy from beam to diffuse as
    cloud increases. DHI follows by closure.

    ``zenith`` is the apparent (refraction-corrected) zenith used for the
    closure; ``true_zenith`` (defaults to ``zenith``) is passed to DISC, which
    is defined on the geometric zenith.
    """
    if clear.ghi <= 0.0:
        return ZERO_IRRADIANCE
    ghi = cloud_cover_to_ghi(clear.ghi, cloud_cover)
    cos_zenith = math.cos(zenith * _DEG)
    dz = zenith if true_zenith is None else true_zenith
    disc_clear = disc_dni(clear.ghi, dz, day_of_year, pressure)
    if disc_clear > 0.0:
        ratio = disc_dni(ghi, dz, day_of_year, pressure) / disc_clear
    else:
        # DISC is undefined near the horizon; scale the beam with GHI.
        ratio = ghi / clear.ghi
    dni = clear.dni * min(max(ratio, 0.0), 1.0)
    if cos_zenith > 0.0:
        dni = min(dni, ghi / cos_zenith)  # closure guard: DHI must stay >= 0
    else:
        dni = 0.0
    dhi = max(ghi - dni * cos_zenith, 0.0)
    return Irradiance(ghi=ghi, dni=dni, dhi=dhi)


def angle_of_incidence_cos(
    surface_tilt: float, surface_azimuth: float, solar_zenith: float, solar_azimuth: float
) -> float:
    """Cosine of the angle of incidence on a tilted plane (unclipped)."""
    return math.cos(solar_zenith * _DEG) * math.cos(surface_tilt * _DEG) + math.sin(
        solar_zenith * _DEG
    ) * math.sin(surface_tilt * _DEG) * math.cos((solar_azimuth - surface_azimuth) * _DEG)


# Perez et al. (1990) 'allsitescomposite1990' coefficients, as published
# (Solar Energy 44(5), Table 6) and as used by pvlib.
# Rows: sky clearness bins 1..8; columns: (f11, f12, f13) and (f21, f22, f23).
_PEREZ_F1 = (
    (-0.008, 0.588, -0.062),
    (0.130, 0.683, -0.151),
    (0.330, 0.487, -0.221),
    (0.568, 0.187, -0.295),
    (0.873, -0.392, -0.362),
    (1.132, -1.237, -0.412),
    (1.060, -1.600, -0.359),
    (0.678, -0.327, -0.250),
)
_PEREZ_F2 = (
    (-0.060, 0.072, -0.022),
    (-0.019, 0.066, -0.029),
    (0.055, -0.064, -0.026),
    (0.109, -0.152, -0.014),
    (0.226, -0.462, 0.001),
    (0.288, -0.823, 0.056),
    (0.264, -1.127, 0.131),
    (0.156, -1.377, 0.251),
)
_PEREZ_EPS_BINS = (1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2)


def perez_sky_diffuse(
    surface_tilt: float,
    surface_azimuth: float,
    dhi: float,
    dni: float,
    dni_extra: float,
    solar_zenith: float,
    solar_azimuth: float,
    airmass: float | None,
) -> float:
    """In-plane sky diffuse irradiance, Perez et al. (1990).

    ``airmass`` is the relative air mass. Mirrors ``pvlib.irradiance.perez``
    with ``model='allsitescomposite1990'``.
    """
    if dhi <= 0.0 or airmass is None:
        return 0.0

    kappa = 1.041  # for solar_zenith in radians
    z = solar_zenith * _DEG

    delta = dhi * airmass / dni_extra
    z3 = kappa * z**3
    eps = ((dhi + dni) / dhi + z3) / (1.0 + z3)

    ebin = 7
    for i, edge in enumerate(_PEREZ_EPS_BINS):
        if eps < edge:
            ebin = i
            break

    f11, f12, f13 = _PEREZ_F1[ebin]
    f21, f22, f23 = _PEREZ_F2[ebin]
    f1 = max(f11 + f12 * delta + z * f13, 0.0)
    f2 = f21 + f22 * delta + z * f23

    a = max(
        angle_of_incidence_cos(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth),
        0.0,
    )
    b = max(math.cos(solar_zenith * _DEG), math.cos(85.0 * _DEG))

    term1 = 0.5 * (1.0 - f1) * (1.0 + math.cos(surface_tilt * _DEG))
    term2 = f1 * a / b
    term3 = f2 * math.sin(surface_tilt * _DEG)
    return max(dhi * (term1 + term2 + term3), 0.0)


def ground_diffuse(surface_tilt: float, ghi: float, albedo: float) -> float:
    """Ground-reflected in-plane irradiance for an isotropic ground."""
    return ghi * albedo * (1.0 - math.cos(surface_tilt * _DEG)) * 0.5


# --- Incidence angle modifiers (Martin & Ruiz 2001) -------------------------


def iam_martin_ruiz(cos_aoi: float, a_r: float = 0.16) -> float:
    """Beam incidence angle modifier, normalised to 1 at normal incidence."""
    if cos_aoi <= 0.0:
        return 0.0
    return (1.0 - math.exp(-cos_aoi / a_r)) / (1.0 - math.exp(-1.0 / a_r))


def iam_martin_ruiz_diffuse(surface_tilt: float, a_r: float = 0.16) -> tuple[float, float]:
    """Sky and ground diffuse IAM for isotropic radiance (Martin & Ruiz 2001).

    Returns ``(iam_sky, iam_ground)``. Mirrors ``pvlib.iam.martin_ruiz_diffuse``
    with ``c1 = 4/(3*pi) ~ 0.4244`` and ``c2 = 0.5*a_r - 0.154``.
    """
    c1 = 0.4244
    c2 = 0.5 * a_r - 0.154
    # Like pvlib, nudge exactly-180 tilt to avoid 1 + cos(pi) == 0.
    surface_tilt = min(surface_tilt, 180.0 - 1e-6)
    beta = surface_tilt * _DEG
    sin_beta = math.sin(beta) if surface_tilt < 90.0 else math.sin(math.pi - beta)

    def _iam(trig: float) -> float:
        return 1.0 - math.exp(-(c1 + c2 * trig) * trig / a_r)

    trig_sky = sin_beta + (math.pi - beta - sin_beta) / (1.0 + math.cos(beta))
    iam_sky = _iam(trig_sky)

    denom = 1.0 - math.cos(beta)
    if denom <= 1e-12:
        iam_gnd = 0.0  # horizontal surface sees no ground
    else:
        trig_gnd = sin_beta + (beta - sin_beta) / denom
        iam_gnd = _iam(trig_gnd)
    return iam_sky, iam_gnd
