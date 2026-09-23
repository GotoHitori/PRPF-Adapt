# Data preparation

Raw radar data and Pangu-Weather forecasts are not redistributed; each source keeps its own licence.

## SEVIR (VIL)
* VIL storm events, 384×384 → 128×128, 10-min cadence, 5 input + 20 output frames.
* Chronological split at 2019-06-01 00:00 (`build_sevir_pairs_by_date` in `sevir/*/train.py`).
* `sevir/splits/sevir_{train,test}_periods.txt`: one sample per line, e.g. `storm_201906010034_634`.
* `sevir/splits/sevir_{train,test}_pangufile.txt`: matching Pangu file stem, e.g. `storm_201906010034_634_201906010000_upper`.
* Normalisation: `norm_stats.npz`, `pangu_channel_stats.npz` in `sevir/m0` and `sevir/stf`.

## MeteoNet (North-West reflectivity, dBZ)
* 128×128, 10-min cadence; sample IDs in `meteonet/splits/meteonet_{train,test}_5to20.txt` (e.g. `20180901_0030`).
* Radar: one `.npy` per sample ID. Pangu: one directory per sample ID with the `upper` and `surface` NetCDF files.
* Normalisation: `meteonet/pangu_channel_stats.npz`.

## Pangu-Weather conditioning (34 channels)
5 upper-air variables (u, v, T, q, z) on {500, 600, 700, 850, 925, 1000} hPa + 4 surface fields (t2m, u10, v10, msl),
cropped to the radar domain, resized to 32×32 and interpolated to the 20 lead times by the data loaders.
Pangu-Weather weights are covered by their own licence and are not included.
<!-- TODO: add the script that generates the Pangu forecast files -->
