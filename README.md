# SolarForecast
A solar forecast integration for Home Assistant

A Home Assistant custom integration that forecasts PV power and energy from a
physical model of your system. It uses your site's location and elevation, the
geometry of each panel array, and the cloud cover and temperature forecasts
from your existing weather entities, for example
[Pirate Weather](https://github.com/Pirate-Weather/pirate-weather-ha) or
OpenWeatherMap.

It works entirely locally. It calls no cloud API beyond what your weather
integrations already fetch.

## Features

- Multiple arrays with different tilt, azimuth and size. Each array is a
  config subentry with its own device and sensors.
- Several weather entities can be combined. Their cloud cover and temperature
  are averaged, and a source with a shorter horizon does not cut the forecast
  short.
- Hourly forecasts come first. Daily forecasts extend the horizon up to 7 days.
- Integrates with the **Energy dashboard** solar forecast.
- Hours that have already passed are frozen and persisted, so today's total
  and the Energy dashboard keep the forecast as it stood for each hour.

## Model

The model is pure Python with no NumPy or pvlib dependency. Each function is
validated against [pvlib](https://pvlib-python.readthedocs.io/).

| Step | Model |
|---|---|
| Sun position | NOAA / Meeus algorithm with refraction correction |
| Clear-sky irradiance | Ineichen–Perez using the SoDa monthly Linke turbidity climatology (bundled, 0.5°) and site elevation |
| Cloud attenuation | Kasten–Czeplak: GHI = GHI<sub>clear</sub> · (1 − 0.75·C<sup>3.4</sup>) |
| Beam/diffuse split | Clear-sky DNI scaled by the DISC (Maxwell) attenuation ratio; DHI by closure |
| Transposition | Perez 1990 anisotropic sky, plus isotropic ground reflection |
| Reflection losses | Martin–Ruiz incidence angle modifier (beam, sky and ground) |
| Cell temperature | PVsyst thermal model; the loss factor depends on the mounting type |
| DC power | PVWatts with the module temperature coefficient and DC losses |
| AC power | PVWatts inverter efficiency curve, clipped at the inverter AC rating |

Power is evaluated every 5 minutes, and hourly energy is integrated with the
trapezoidal rule.

## Installation

1. Copy `custom_components/solar_forecast` into your Home Assistant
   `config/custom_components/` folder, or add this repository to HACS as a
   custom repository.
2. Restart Home Assistant.
3. Go to **Settings > Devices & services > Add integration > Solar Forecast**.

Requires Home Assistant 2026.10 or later.

## Configuration

**Site and weather:** latitude, longitude and elevation (defaults come from
your Home Assistant configuration), one or more weather entities, and
optionally the inverter AC limit and efficiency. The weather entities must
provide `cloud_coverage` in their hourly or daily forecast. Pirate Weather and
OpenWeatherMap both do.

**Each PV array:**

| Field | Meaning |
|---|---|
| DC nameplate power | Total STC watts of the modules in the array |
| Tilt | 0° = flat, 90° = vertical |
| Azimuth | Compass bearing the panels face: 0° N, 90° E, **180° S**, 270° W |
| Mounting | Open rack, roof mount with an air gap, or building integrated |
| Temperature coefficient | P<sub>max</sub> coefficient from the datasheet, for example −0.35 %/°C |
| DC system losses | Soiling, wiring, mismatch, degradation and nameplate tolerance (default 10 %) |
| Module efficiency | Used by the thermal model |
| Ground albedo | 0.2 is typical; fresh snow is 0.6–0.8 |

To add more arrays, open the integration entry and choose **Add PV array**.
To change the site settings, use **Reconfigure**.

**Options:**

- **Production sensor** enables calibration (see below).
- **Turbidity scale** multiplies the Linke turbidity climatology. Leave it at
  1.0, especially when calibration is enabled.

## Calibration from measured production

If you select a sensor that measures your actual PV output, the forecast
learns from it. Under **Configure**, choose either of these sensor types:

- An energy sensor (kWh, state class `total` or `total_increasing`), such as
  your inverter's lifetime or daily yield
- A power sensor (W, state class `measurement`)

The integration reads the sensor's hourly long-term statistics from the
recorder.

The calibration learns two corrections:

1. **System factor and horizon shading.** In hours that were forecast as
   clear, it compares measured output with clear-sky model output. A robust
   quantile gives the system factor, which captures real losses, soiling,
   and orientation or rating errors. Hours that turned out cloudier than
   forecast don't drag it down.

   Within each sun-position bin (15° of azimuth × 10° of elevation), any
   further shortfall is treated as an obstruction of the direct beam, such
   as a tree or a building. It is applied only to the beam share of the
   light. Shading therefore matters on sunny hours and disappears under
   overcast.
2. **Cloud response.** In each 10 % cloud-cover band, it compares measured
   output with the forecast. This corrects your weather provider's cloud
   bias at your site, including the average effect of its forecast misses.

   Two cloud curves are learned. One uses the forecast as it stood at the
   time (used for the next few hours). The other uses the forecast as it
   stood the evening before (used from about 12 hours ahead). Bands with
   little data borrow from neighbouring bands.

Details:

- Recent data counts more; its weight halves every 30 days.
- Training recomputes the model for each stored hour with your current
  configuration, so editing an array doesn't invalidate the calibration.
- All comparisons are made on the DC side, with measured AC converted to DC
  through the inverter model. Inverter clipping and efficiency are therefore
  still modelled correctly.
- These hours are excluded: outages; likely snow cover (forecast at 4 °C or
  colder with output far below normal); metering glitches; low sun; inverter
  clipping; and hours right after a gap in the sensor's statistics.

The calibration starts helping within a few days and settles after one to
three weeks. An energy sensor is preferred over a power sensor: hourly power
averages are less exact around outages.

Two diagnostic sensors show how it is doing:

- **Calibration factor:** the effective clear-sky correction. Its attributes
  include the system factor, both cloud-response curves, the number of
  shaded bins, and the hours used and rejected.
- **Day-ahead forecast error (7 days):** the mean absolute error of each
  day's forecast as made the evening before, compared with measured
  production. Its attributes also include the bias, the same-day error and
  per-day values.

Limitations:

- **Export limits, zero-export setups and curtailment when the battery is
  full.** Measured output then reflects demand, not sunshine. Calibration
  learns this as reduced output.
- **Shading resolution.** The shading map works at 15° × 10° resolution from
  hourly data, so sharp horizon edges are smoothed.
- **Per-array values.** Calibration is learned from the site total and
  applied to every array alike. Shading on one array is therefore spread
  across all arrays in the per-array sensors; the site total is still right.

## Accuracy notes

Cloud cover forecasts are the largest source of error. Cloud *fraction* alone
does not describe cloud thickness, so a sky that is 100 % covered by thin
cirrus and one covered by thick stratus look the same. Kasten–Czeplak is a
well-established average relationship between the two. Combining two weather
sources usually reduces the error.

## Data attribution

The Linke turbidity climatology comes from SoDa / Meteotest (Remund et al.,
2003), as distributed with pvlib. It was block-averaged to 0.5°.
