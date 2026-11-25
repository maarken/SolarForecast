"""PV array, cell temperature and inverter models."""

from dataclasses import dataclass

from .irradiance import (
    Irradiance,
    angle_of_incidence_cos,
    ground_diffuse,
    iam_martin_ruiz,
    iam_martin_ruiz_diffuse,
    perez_sky_diffuse,
)

# PVsyst thermal loss factors (W/m2K) by mounting, wind term disabled
# (u_v = 0) because forecast wind is at 10 m and unreliable at module level.
MOUNTING_U_C: dict[str, float] = {
    "open_rack": 29.0,  # free-standing, ground or flat-roof racking
    "roof_gap": 20.0,  # semi-integrated / flush roof mount with air gap
    "integrated": 15.0,  # building integrated, insulated back
}


@dataclass(frozen=True, slots=True)
class ArrayConfig:
    """Static description of one PV array (one orientation)."""

    name: str
    tilt: float
    """Surface tilt from horizontal, degrees (0 = flat, 90 = vertical)."""
    azimuth: float
    """Surface azimuth, degrees clockwise from north (180 = south facing)."""
    dc_power_w: float
    """Nameplate DC power at STC, W."""
    temp_coefficient: float = -0.0037
    """Power temperature coefficient, 1/degC (e.g. -0.0037 = -0.37 %/K)."""
    mounting: str = "roof_gap"
    losses: float = 0.10
    """DC system losses as a fraction (soiling, wiring, mismatch, LID...)."""
    albedo: float = 0.20
    iam_a_r: float = 0.16
    module_efficiency: float = 0.20
    """Module efficiency at STC, used by the PVsyst thermal model."""


@dataclass(frozen=True, slots=True)
class PlaneOfArray:
    """In-plane irradiance components (W/m2)."""

    beam: float
    sky_diffuse: float
    ground_diffuse: float
    effective: float
    """Irradiance converted by the cells after incidence angle losses."""
    beam_effective: float = 0.0
    """Beam part of ``effective`` (after its incidence angle modifier)."""

    @property
    def total(self) -> float:
        """Total in-plane irradiance before optical losses."""
        return self.beam + self.sky_diffuse + self.ground_diffuse


ZERO_POA = PlaneOfArray(0.0, 0.0, 0.0, 0.0)


def plane_of_array(
    array: ArrayConfig,
    irr: Irradiance,
    dni_extra: float,
    solar_zenith: float,
    solar_azimuth: float,
    airmass: float | None,
) -> PlaneOfArray:
    """Transpose horizontal irradiance onto the array plane (Perez)."""
    if irr.ghi <= 0.0:
        return ZERO_POA

    cos_aoi = angle_of_incidence_cos(array.tilt, array.azimuth, solar_zenith, solar_azimuth)
    beam = irr.dni * max(cos_aoi, 0.0)
    sky = perez_sky_diffuse(
        array.tilt,
        array.azimuth,
        irr.dhi,
        irr.dni,
        dni_extra,
        solar_zenith,
        solar_azimuth,
        airmass,
    )
    ground = ground_diffuse(array.tilt, irr.ghi, array.albedo)

    iam_beam = iam_martin_ruiz(cos_aoi, array.iam_a_r)
    iam_sky, iam_gnd = iam_martin_ruiz_diffuse(array.tilt, array.iam_a_r)
    beam_effective = beam * iam_beam
    effective = beam_effective + sky * iam_sky + ground * iam_gnd
    return PlaneOfArray(
        beam=beam,
        sky_diffuse=sky,
        ground_diffuse=ground,
        effective=effective,
        beam_effective=beam_effective,
    )


def cell_temperature_pvsyst(
    poa_global: float,
    temp_air: float,
    u_c: float,
    u_v: float = 0.0,
    wind_speed: float = 1.0,
    module_efficiency: float = 0.1,
    alpha_absorption: float = 0.9,
) -> float:
    """Cell temperature, PVsyst model (mirrors ``pvlib.temperature.pvsyst_cell``)."""
    total_loss_factor = u_c + u_v * wind_speed
    heat_input = poa_global * alpha_absorption * (1.0 - module_efficiency)
    return temp_air + heat_input / total_loss_factor


def pvwatts_dc(
    effective_irradiance: float, temp_cell: float, pdc0: float, gamma_pdc: float
) -> float:
    """PVWatts DC power (mirrors ``pvlib.pvsystem.pvwatts_dc``)."""
    return effective_irradiance * 0.001 * pdc0 * (1.0 + gamma_pdc * (temp_cell - 25.0))


def array_dc_power(array: ArrayConfig, poa: PlaneOfArray, temp_air: float) -> float:
    """DC output of an array (W) after temperature and DC system losses."""
    if poa.effective <= 0.0:
        return 0.0
    t_cell = cell_temperature_pvsyst(
        poa.total,
        temp_air,
        u_c=MOUNTING_U_C.get(array.mounting, MOUNTING_U_C["roof_gap"]),
        module_efficiency=array.module_efficiency,
    )
    dc = pvwatts_dc(poa.effective, t_cell, array.dc_power_w, array.temp_coefficient)
    return max(dc * (1.0 - array.losses), 0.0)


@dataclass(frozen=True, slots=True)
class InverterConfig:
    """System inverter. ``ac_power_w`` of ``None`` means no clipping/limit."""

    ac_power_w: float | None = None
    efficiency_nominal: float = 0.96


def inverter_dc_from_ac(
    inv: InverterConfig, pac: float, pdc_total_nameplate: float
) -> float | None:
    """Invert :func:`inverter_ac_power`: DC power that yields ``pac``.

    Returns ``None`` if ``pac`` is at or above the AC limit (clipped, so the
    DC power is not identifiable). Solved by bisection; the PVWatts curve is
    monotonic in DC power over the operating range.
    """
    if pac <= 0.0:
        return 0.0
    if inv.ac_power_w and pac >= 0.98 * inv.ac_power_w:
        return None
    lo, hi = 0.0, max(pac * 2.0, 1.0)
    while inverter_ac_power(inv, hi, pdc_total_nameplate) < pac:
        hi *= 2.0
        if hi > 1e9:
            return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if inverter_ac_power(inv, mid, pdc_total_nameplate) < pac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def inverter_ac_power(inv: InverterConfig, pdc: float, pdc_total_nameplate: float) -> float:
    """AC power from total DC power, PVWatts inverter model.

    Mirrors ``pvlib.inverter.pvwatts`` (eta_inv_ref = 0.9637). When no AC
    rating is configured, the inverter is sized to the DC nameplate for the
    part-load efficiency curve and no clipping is applied.
    """
    if pdc <= 0.0:
        return 0.0
    eta_nom = inv.efficiency_nominal
    eta_ref = 0.9637
    pac0 = inv.ac_power_w if inv.ac_power_w else pdc_total_nameplate * eta_nom
    if pac0 <= 0.0:
        return 0.0
    pdc0 = pac0 / eta_nom
    zeta = pdc / pdc0
    eta = eta_nom / eta_ref * (-0.0162 * zeta - 0.0059 / zeta + 0.9858)
    pac = max(eta * pdc, 0.0)
    if inv.ac_power_w:
        pac = min(pac, inv.ac_power_w)
    return pac
