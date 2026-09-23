import os
import re
import glob
import csv
import warnings
import ctypes
import ctypes.util
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def normalize_archive_sample(radar, pangu, radar_mean, radar_std, pangu_means, pangu_stds):
    radar = (radar - radar_mean) / max(radar_std, 1e-6)
    pangu = (pangu - pangu_means.view(1, -1, 1, 1)) / pangu_stds.view(1, -1, 1, 1)
    return radar, pangu

def _xarray():
    import xarray
    return xarray


def _validate_netcdf_attributes(attributes, path, variable):
    for name in ("scale_factor", "add_offset", "missing_value"):
        if name in attributes:
            raise ValueError(f"unsupported NetCDF attribute {name}:{variable}:{path}")
    if "_FillValue" in attributes:
        value = np.asarray(attributes["_FillValue"])
        if value.size != 1 or not np.isnan(value.reshape(-1)[0]):
            raise ValueError(f"unsupported NetCDF attribute _FillValue:{variable}:{path}")


class _NetCDFValueReader:
    def __init__(self):
        library_path = ctypes.util.find_library("netcdf")
        if library_path is None:
            raise RuntimeError("libnetcdf is unavailable")
        self.library = ctypes.CDLL(library_path)
        self.library.nc_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.library.nc_open.restype = ctypes.c_int
        self.library.nc_close.argtypes = [ctypes.c_int]
        self.library.nc_close.restype = ctypes.c_int
        self.library.nc_inq_varid.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        self.library.nc_inq_varid.restype = ctypes.c_int
        self.library.nc_inq_varndims.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.library.nc_inq_varndims.restype = ctypes.c_int
        self.library.nc_inq_vardimid.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.library.nc_inq_vardimid.restype = ctypes.c_int
        self.library.nc_inq_dimlen.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
        self.library.nc_inq_dimlen.restype = ctypes.c_int
        self.library.nc_get_var_double.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self.library.nc_get_var_double.restype = ctypes.c_int
        self.library.nc_inq_attid.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        self.library.nc_inq_attid.restype = ctypes.c_int
        self.library.nc_get_att_double.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
        self.library.nc_get_att_double.restype = ctypes.c_int

    def _attributes(self, ncid, varid, path, variable):
        attributes = {}
        for name in ("scale_factor", "add_offset", "missing_value", "_FillValue"):
            attribute_id = ctypes.c_int()
            if self.library.nc_inq_attid(
                ncid, varid, name.encode(), ctypes.byref(attribute_id)
            ) == 0:
                if name != "_FillValue":
                    attributes[name] = True
                else:
                    value = ctypes.c_double()
                    if self.library.nc_get_att_double(
                        ncid, varid, name.encode(), ctypes.byref(value)
                    ) != 0:
                        raise ValueError(f"unsupported NetCDF attribute {name}:{variable}:{path}")
                    attributes[name] = value.value
        _validate_netcdf_attributes(attributes, path, variable)

    def read(self, path, names):
        ncid = ctypes.c_int()
        if self.library.nc_open(os.fsencode(path), 0, ctypes.byref(ncid)) != 0:
            raise RuntimeError(f"cannot open NetCDF file:{path}")
        try:
            result = {}
            for name in names:
                varid = ctypes.c_int()
                if self.library.nc_inq_varid(ncid.value, name.encode(), ctypes.byref(varid)) != 0:
                    raise RuntimeError(f"missing NetCDF variable {name}:{path}")
                self._attributes(ncid.value, varid.value, path, name)
                rank = ctypes.c_int()
                if self.library.nc_inq_varndims(ncid.value, varid.value, ctypes.byref(rank)) != 0:
                    raise RuntimeError(f"cannot inspect NetCDF variable {name}:{path}")
                dimension_ids = (ctypes.c_int * rank.value)()
                if self.library.nc_inq_vardimid(ncid.value, varid.value, dimension_ids) != 0:
                    raise RuntimeError(f"cannot inspect NetCDF dimensions {name}:{path}")
                shape = []
                for dimension_id in dimension_ids:
                    length = ctypes.c_size_t()
                    if self.library.nc_inq_dimlen(
                        ncid.value, dimension_id, ctypes.byref(length)
                    ) != 0:
                        raise RuntimeError(f"cannot inspect NetCDF dimension {name}:{path}")
                    shape.append(length.value)
                values = np.empty(int(np.prod(shape, dtype=np.int64)), dtype=np.float64)
                if self.library.nc_get_var_double(
                    ncid.value,
                    varid.value,
                    values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ) != 0:
                    raise RuntimeError(f"cannot read NetCDF variable {name}:{path}")
                result[name] = values.reshape(shape).astype(np.float32)
            return result
        finally:
            self.library.nc_close(ncid.value)


_NETCDF_VALUE_READER = None


def _read_netcdf_arrays(path, names):
    global _NETCDF_VALUE_READER
    if _NETCDF_VALUE_READER is None:
        _NETCDF_VALUE_READER = _NetCDFValueReader()
    return _NETCDF_VALUE_READER.read(path, names)

FIXED_WFM_C = 34
RADAR_IN_T = 5
RADAR_OUT_T = 20
PANGU_T = 5
PANGU_FIXED_H = 32
PANGU_FIXED_W = 32
RADAR_FIXED_H = 128
RADAR_FIXED_W = 128
PANGU_UPPER_VARIABLES = (
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "geopotential",
)
PANGU_SOURCE_PRESSURE_LEVELS_HPA = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
PANGU_PRESSURE_LEVELS_HPA = (500, 600, 700, 850, 925, 1000)
PANGU_PRESSURE_LEVEL_INDICES = tuple(
    PANGU_SOURCE_PRESSURE_LEVELS_HPA.index(level) for level in PANGU_PRESSURE_LEVELS_HPA
)

PANGU_CHANNEL_ORDER = tuple(
    [
        *[
            f"{variable}[{level}hPa]"
            for variable in PANGU_UPPER_VARIABLES
            for level in PANGU_PRESSURE_LEVELS_HPA
        ],
        "temperature_2m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "mean_sea_level_pressure",
    ]
)

_STATS_PROVENANCE_FIELDS = {
    "source_event_count",
    "source_split_hash",
    "dtype",
    "channel_order",
    "source_audit_sha256",
    "source_audit_decision",
    "paper_exact_ready",
    "pangu_causality_status",
    "training_permission",
    "generation_id",
}
_RADAR_STATS_FIELDS = {"radar_mean", "radar_std", "value_count", *_STATS_PROVENANCE_FIELDS}
_PANGU_STATS_FIELDS = {"means", "stds", "pixel_counts", *_STATS_PROVENANCE_FIELDS}


def _stats_scalar(archive, name):
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"statistics field must be scalar:{name}")
    return value.item()


def _require_dtype(array, dtype, name):
    value = np.asarray(array)
    if value.dtype != np.dtype(dtype):
        raise ValueError(f"statistics field has wrong dtype:{name}")
    return value


def _require_unicode(array, name):
    value = np.asarray(array)
    if value.dtype.kind != "U":
        raise ValueError(f"statistics field has wrong dtype:{name}")
    return value


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def load_required_stats(radar_path, pangu_path, expected_split_hash=None):
    radar_path = Path(radar_path)
    pangu_path = Path(pangu_path)
    if not radar_path.is_file():
        raise FileNotFoundError(radar_path)
    if not pangu_path.is_file():
        raise FileNotFoundError(pangu_path)
    with np.load(radar_path, allow_pickle=False) as radar, np.load(
        pangu_path, allow_pickle=False
    ) as pangu:
        if set(radar.files) != _RADAR_STATS_FIELDS:
            raise ValueError("wrong radar fields")
        if set(pangu.files) != _PANGU_STATS_FIELDS:
            raise ValueError("wrong Pangu fields")
        radar_mean_array = _require_dtype(radar["radar_mean"], np.float64, "radar_mean")
        radar_std_array = _require_dtype(radar["radar_std"], np.float64, "radar_std")
        pangu_means = _require_dtype(pangu["means"], np.float64, "means")
        pangu_stds = _require_dtype(pangu["stds"], np.float64, "stds")
        pangu_counts = _require_dtype(pangu["pixel_counts"], np.int64, "pixel_counts")
        radar_mean = float(_stats_scalar({"radar_mean": radar_mean_array}, "radar_mean"))
        radar_std = float(_stats_scalar({"radar_std": radar_std_array}, "radar_std"))
        if pangu_means.shape != (FIXED_WFM_C,) or pangu_stds.shape != (FIXED_WFM_C,):
            raise ValueError("Pangu normalization must contain exactly 34 channels")
        if pangu_counts.shape != (FIXED_WFM_C,):
            raise ValueError("Pangu pixel counts must contain exactly 34 channels")
        numeric = np.concatenate(
            [np.array([radar_mean, radar_std]), pangu_means, pangu_stds]
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError("normalization values must be finite")
        if radar_std < 1e-6 or np.any(pangu_stds < 1e-6):
            raise ValueError("normalization standard deviations must be at least 1e-6")
        radar_count = _stats_scalar(
            {"value_count": _require_dtype(radar["value_count"], np.int64, "value_count")},
            "value_count",
        )
        expected_radar_count = 10776 * 25 * RADAR_FIXED_H * RADAR_FIXED_W
        expected_pangu_count = 10776 * PANGU_T * PANGU_FIXED_H * PANGU_FIXED_W
        if radar_count != expected_radar_count:
            raise ValueError("radar value count does not match the exact protocol")
        if not np.all(pangu_counts == expected_pangu_count):
            raise ValueError("Pangu pixel count does not match the exact protocol")
        _require_dtype(radar["source_event_count"], np.int64, "source_event_count")
        _require_dtype(pangu["source_event_count"], np.int64, "source_event_count")
        radar_events = _stats_scalar(radar, "source_event_count")
        pangu_events = _stats_scalar(pangu, "source_event_count")
        if radar_events != 10776 or pangu_events != 10776:
            raise ValueError("source event count must be exactly 10776")
        radar_split_hash = _stats_scalar(radar, "source_split_hash")
        pangu_split_hash = _stats_scalar(pangu, "source_split_hash")
        if not _valid_sha256(radar_split_hash) or radar_split_hash != pangu_split_hash:
            raise ValueError("source split hash mismatch")
        if expected_split_hash is not None and radar_split_hash != expected_split_hash:
            raise ValueError("source split hash does not match expected split hash")
        radar_audit_hash = _stats_scalar(radar, "source_audit_sha256")
        pangu_audit_hash = _stats_scalar(pangu, "source_audit_sha256")
        if not _valid_sha256(radar_audit_hash) or radar_audit_hash != pangu_audit_hash:
            raise ValueError("source audit provenance mismatch")
        for archive, order in (
            (radar, ("radar",)),
            (pangu, PANGU_CHANNEL_ORDER),
        ):
            _require_unicode(archive["dtype"], "dtype")
            _require_unicode(archive["channel_order"], "channel_order")
            _require_unicode(archive["source_split_hash"], "source_split_hash")
            _require_unicode(archive["source_audit_sha256"], "source_audit_sha256")
            _require_unicode(archive["source_audit_decision"], "source_audit_decision")
            _require_unicode(archive["pangu_causality_status"], "pangu_causality_status")
            _require_unicode(archive["training_permission"], "training_permission")
            _require_unicode(archive["generation_id"], "generation_id")
            _require_dtype(archive["paper_exact_ready"], np.bool_, "paper_exact_ready")
            if _stats_scalar(archive, "dtype") != "float64":
                raise ValueError("statistics dtype provenance must be float64")
            if tuple(np.asarray(archive["channel_order"]).tolist()) != order:
                raise ValueError("statistics channel order mismatch")
            if _stats_scalar(archive, "source_audit_decision") != "blocked":
                raise ValueError("source audit decision must remain blocked")
            if bool(_stats_scalar(archive, "paper_exact_ready")):
                raise ValueError("paper-exact readiness must remain false")
            if _stats_scalar(archive, "pangu_causality_status") != "blocked":
                raise ValueError("Pangu causality status must remain blocked")
            if _stats_scalar(archive, "training_permission") != "blocked":
                raise ValueError("statistics cannot grant training permission")
        radar_generation = _stats_scalar(radar, "generation_id")
        pangu_generation = _stats_scalar(pangu, "generation_id")
        if not _valid_sha256(radar_generation) or radar_generation != pangu_generation:
            raise ValueError("statistics generation identity mismatch")
    return radar_mean, radar_std, pangu_means.copy(), pangu_stds.copy()

def _pad_hw_4d(x, Ht, Wt):
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = Ht - h, Wt - w
    if ph < 0 or pw < 0:
        x = x[..., :min(h, Ht), :min(w, Wt)]
        h, w = x.shape[-2], x.shape[-1]
        ph, pw = Ht - h, Wt - w
    if ph == 0 and pw == 0:
        return x
    return F.pad(x, (0, pw, 0, ph), mode="replicate")

def load_catalog(path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    tbl: Dict[str, Dict[str, str]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("radar_id") or row.get("vil_id") or row.get("id") or row.get("rid") or row.get("station")
            if not rid:
                continue
            rid = str(rid).strip().upper()
            m = re.match(r"^R?(\d+)$", rid)
            if m:
                rid_norm = f"R{m.group(1)}"
            else:
                rid_norm = rid if rid.startswith("R") else f"R{rid}"
            tbl[rid_norm] = row
    return tbl

def parse_fname(fname: str) -> Tuple[str, str]:
    m = re.match(r"(\d{8}_\d{4})_R(\d+)", fname)
    if m is None:
        raise ValueError(f"invalid filename: {fname}")
    return m.group(1), f"R{m.group(2)}"

_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})")

def _list_pangu_pairs_in_dir(folder: str) -> List[Tuple[datetime, str, str]]:
    up_files = sorted(glob.glob(os.path.join(folder, "*upper*.nc")))
    sf_files = sorted(glob.glob(os.path.join(folder, "*surface*.nc")))
    if not up_files or not sf_files:
        raise RuntimeError(f"[Pangu] missing upper/surface files under: {folder}")

    def ts_from_name(p: str) -> Optional[datetime]:
        m = _TS_RE.search(os.path.basename(p))
        if not m:
            return None
        return datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M")

    up_map: Dict[datetime, str] = {}
    for p in up_files:
        ts = ts_from_name(p)
        if ts:
            up_map[ts] = p

    pairs: List[Tuple[datetime, str, str]] = []
    for p in sf_files:
        ts = ts_from_name(p)
        if ts and ts in up_map:
            pairs.append((ts, up_map[ts], p))

    if not pairs:
        raise RuntimeError(f"[Pangu] no matched upper/surface pairs in {folder}")
    pairs.sort(key=lambda x: x[0])
    return pairs

def _choose_5_around_ts(pairs: List[Tuple[datetime, str, str]], ts_dt: datetime) -> List[Tuple[datetime, str, str]]:
    if len(pairs) <= PANGU_T:
        return pairs[:PANGU_T]
    idx = -1
    for i, (t, _, _) in enumerate(pairs):
        if t <= ts_dt:
            idx = i
        else:
            break
    if idx < 0:
        return pairs[:PANGU_T]
    start = max(0, idx - (PANGU_T - 1))
    end = start + PANGU_T
    if end > len(pairs):
        end = len(pairs)
        start = end - PANGU_T
    return pairs[start:end]

def _resize_hw_to_fixed(arr: np.ndarray, target_h: int = PANGU_FIXED_H, target_w: int = PANGU_FIXED_W) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D array [C,H,W], got {arr.shape}")
    C, h, w = arr.shape
    if h == target_h and w == target_w:
        return arr
    t = torch.from_numpy(arr).unsqueeze(0)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()

def _read_upper_30ch(upper_path: str) -> np.ndarray:
    cubes = []
    try:
        with _xarray().open_dataset(
            upper_path, decode_cf=False, mask_and_scale=False
        ) as ds:
            values = {}
            for v in PANGU_UPPER_VARIABLES:
                if v in ds:
                    _validate_netcdf_attributes(getattr(ds[v], "attrs", {}), upper_path, v)
                    values[v] = ds[v].values.astype(np.float32)
    except ModuleNotFoundError:
        values = _read_netcdf_arrays(upper_path, PANGU_UPPER_VARIABLES)
    for v in PANGU_UPPER_VARIABLES:
        if v not in values:
            raise RuntimeError(f"Missing required variable {v} in {upper_path}")
        val = values[v]
        if val.ndim == 4:
            val = val[0]
        if val.ndim != 3:
            raise RuntimeError(f"Invalid dimensions for variable {v} in {upper_path}")
        if val.shape[0] < len(PANGU_SOURCE_PRESSURE_LEVELS_HPA):
            raise RuntimeError(f"Missing official Pangu pressure levels for variable {v} in {upper_path}")
        cubes.append(val[np.asarray(PANGU_PRESSURE_LEVEL_INDICES)])
    up = np.concatenate(cubes, axis=0)
    up = _resize_hw_to_fixed(up)
    return up

def _read_surface_4ch(surface_path: str) -> np.ndarray:
    VARS = ['temperature_2m','u_component_of_wind_10m','v_component_of_wind_10m','mean_sea_level_pressure']
    cubes = []
    try:
        with _xarray().open_dataset(
            surface_path, decode_cf=False, mask_and_scale=False
        ) as ds:
            values = {}
            for v in VARS:
                if v in ds:
                    _validate_netcdf_attributes(getattr(ds[v], "attrs", {}), surface_path, v)
                    values[v] = ds[v].values.astype(np.float32)
    except ModuleNotFoundError:
        values = _read_netcdf_arrays(surface_path, VARS)
    for v in VARS:
        if v not in values:
            raise RuntimeError(f"Missing required variable {v} in {surface_path}")
        val = values[v]
        if val.ndim == 3:
            val = val[0]
        if val.ndim != 2:
            raise RuntimeError(f"Invalid dimensions for variable {v} in {surface_path}")
        cubes.append(val[None, :, :])
    sf = np.concatenate(cubes, axis=0)
    sf = _resize_hw_to_fixed(sf)
    return sf

def _load_pangu_5(folder: str, anchor_ts: datetime) -> np.ndarray:
    pairs = _list_pangu_pairs_in_dir(folder)
    chosen = _choose_5_around_ts(pairs, anchor_ts)
    frames = []
    for _, up_p, sf_p in chosen:
        up30 = _read_upper_30ch(up_p)
        sf4 = _read_surface_4ch(sf_p)
        frame = np.concatenate([up30, sf4], axis=0)
        frames.append(frame)
    arr = np.stack(frames, axis=0)
    return arr.astype(np.float32)

class RadarPanguDataset(Dataset):
    def __init__(self, radar_dir: str, pangu_root: str, catalog_csv: str, split: str = "train"):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train|val|test")
        self.radar_dir = os.path.join(radar_dir, split)
        self.pangu_root = pangu_root
        self.catalog = load_catalog(catalog_csv)
        self.split = split

        all_files = glob.glob(os.path.join(self.radar_dir, "*.npy"))
        self.files: List[str] = []
        for p in all_files:
            try:
                ts, rid = parse_fname(os.path.basename(p))
                if rid in self.catalog:
                    self.files.append(p)
            except Exception:
                continue
        if not self.files:
            raise RuntimeError(f"no usable radar files in {self.radar_dir}")

        try:
            ns = np.load("norm_stats.npz")
            self.radar_mean = float(ns["radar_mean"])
            self.radar_std = float(ns["radar_std"])
        except Exception:
            self.radar_mean = 0.0
            self.radar_std = 1.0

        try:
            ps = np.load("pangu_channel_stats.npz")
            means = ps["means"].astype(np.float32)
            stds = ps["stds"].astype(np.float32)
            if means.shape[0] != FIXED_WFM_C:
                warnings.warn(f"Pangu stats shape mismatch: {means.shape} != {FIXED_WFM_C}")
            stds = np.where(stds < 1e-6, 1.0, stds)
            self.pangu_channel_means = torch.from_numpy(means).view(1, means.shape[0], 1, 1)
            self.pangu_channel_stds = torch.from_numpy(stds).view(1, stds.shape[0], 1, 1)
        except Exception:
            self.pangu_channel_means = torch.zeros((1, FIXED_WFM_C, 1, 1))
            self.pangu_channel_stds = torch.ones((1, FIXED_WFM_C, 1, 1))

        self._rid_dir_cache: Dict[Tuple[str, str], str] = {}

    def _pangu_dir(self, ts: str, rid: str) -> str:
        cache_key = (ts, rid)
        if cache_key in self._rid_dir_cache:
            return self._rid_dir_cache[cache_key]
        root = self.pangu_root
        pattern = os.path.join(root, "**", f"*{rid}*")
        candidates = [p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p)]
        if not candidates:
            rid_digits = re.sub(r"\D", "", rid or "")
            pattern2 = os.path.join(root, "**", f"*R{rid_digits}*")
            candidates = [p for p in glob.glob(pattern2, recursive=True) if os.path.isdir(p)]
        def has_pair(d: str) -> bool:
            ups = glob.glob(os.path.join(d, "*upper*.nc"))
            sfs = glob.glob(os.path.join(d, "*surface*.nc"))
            return len(ups) > 0 and len(sfs) > 0
        candidates = [d for d in candidates if has_pair(d)]
        if not candidates:
            raise RuntimeError(f"[Pangu] cannot find folder for rid={rid} under {root}")
        candidates.sort(key=lambda x: len(x))
        chosen = candidates[0]
        self._rid_dir_cache[cache_key] = chosen
        return chosen

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        base = os.path.basename(p)
        ts, rid = parse_fname(base)
        arr = np.load(p)
        if arr.ndim == 3:
            x = arr[:RADAR_IN_T]
            y = arr[RADAR_IN_T:RADAR_IN_T+RADAR_OUT_T]
            x = x[:, None, :, :]
            y = y[:, None, :, :]
        elif arr.ndim == 4:
            x = arr[:RADAR_IN_T]
            y = arr[RADAR_IN_T:RADAR_IN_T+RADAR_OUT_T]
        else:
            raise ValueError(f"Unexpected radar shape {arr.shape} for {p}")

        x = (x.astype(np.float32) - self.radar_mean) / max(self.radar_std, 1e-6)
        y = (y.astype(np.float32) - self.radar_mean) / max(self.radar_std, 1e-6)

        ts_dt = datetime.strptime(ts, "%Y%m%d_%H%M")
        folder = self._pangu_dir(ts, rid)
        pangu5 = _load_pangu_5(folder, ts_dt)
        pangu5 = torch.from_numpy(pangu5).float()
        pangu5 = (pangu5 - self.pangu_channel_means) / self.pangu_channel_stds
        return torch.from_numpy(x).float(), pangu5, torch.from_numpy(y).float()

_SEVIR_TS12_RE = re.compile(r"(20\d{10})")
def _extract_first_ts12(s: str):
    m = _SEVIR_TS12_RE.search(s)
    return m.group(1) if m else None
def _extract_last_ts12(s: str):
    ms = _SEVIR_TS12_RE.findall(s)
    return ms[-1] if ms else None
def _to_dt_ts12(ts12: str) -> datetime:
    return datetime.strptime(ts12, "%Y%m%d%H%M")
def _is_data_line(s: str) -> bool:
    return bool(_SEVIR_TS12_RE.search(s))

def _find_sevir_radar_path(sevir_root: str, radar_entry: str) -> str:
    train_dir = os.path.join(sevir_root, "train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"SEVIR train dir not found: {train_dir}")
    date_time = _extract_first_ts12(radar_entry)
    if not date_time:
        raise ValueError(f"Cannot parse datetime from radar_entry: {radar_entry}")
    parts = radar_entry.split("_")
    seq_id = None
    for p in reversed(parts):
        if p.isdigit():
            seq_id = p
            break
    if seq_id is None:
        raise ValueError(f"Cannot parse seq_id from radar_entry: {radar_entry}")
    date = date_time[:8]
    time = date_time[8:]
    patterns = [
        f"{date}_{time}_*_{seq_id}_vil.npy",
        f"{date}_{time}_*_{seq_id}*.npy",
        f"*{date}*{time}*{seq_id}*.npy",
        f"*{seq_id}*.npy",
    ]
    for pat in patterns:
        matches = glob.glob(os.path.join(train_dir, pat))
        if matches:
            return matches[0]
    for fn in os.listdir(train_dir):
        if fn.endswith(".npy") and (seq_id in fn) and (date in fn):
            return os.path.join(train_dir, fn)
    raise FileNotFoundError(f"Radar file not found for entry={radar_entry} under {train_dir}")

def _extract_rid_from_radar_filename(fname: str):
    base = os.path.basename(fname)
    parts = base.split("_")
    for p in parts:
        if (p.startswith("R") or p.startswith("S")) and len(p) > 5:
            return p
    return None

def _resize_radar_thw_to_fixed(radar_thw: np.ndarray, target_h: int = RADAR_FIXED_H, target_w: int = RADAR_FIXED_W) -> np.ndarray:
    t = torch.from_numpy(radar_thw.astype(np.float32))
    if t.ndim != 3:
        raise ValueError(f"expected [T,H,W], got {t.shape}")
    t = t.unsqueeze(1)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze(1).numpy()

class SevirTxtPanguDataset(Dataset):
    def __init__(self, pairs, sevir_root: str, pangu_root: str, Tin=5, Tout=20):
        self.pairs = pairs
        self.sevir_root = sevir_root
        self.pangu_root = pangu_root
        self.Tin = Tin
        self.Tout = Tout

        try:
            ns = np.load("norm_stats.npz")
            self.radar_mean = float(ns["radar_mean"])
            self.radar_std = float(ns["radar_std"])
        except Exception:
            self.radar_mean = 0.0
            self.radar_std = 1.0

        try:
            ps = np.load("pangu_channel_stats.npz")
            means = ps["means"].astype(np.float32)
            stds = ps["stds"].astype(np.float32)
            stds = np.where(stds < 1e-6, 1.0, stds)
            self.pangu_channel_means = torch.from_numpy(means).view(1, means.shape[0], 1, 1)
            self.pangu_channel_stds = torch.from_numpy(stds).view(1, stds.shape[0], 1, 1)
        except Exception:
            self.pangu_channel_means = torch.zeros((1, FIXED_WFM_C, 1, 1))
            self.pangu_channel_stds = torch.ones((1, FIXED_WFM_C, 1, 1))

        self._rid_dir_cache = {}

    def __len__(self):
        return len(self.pairs)

    def _pangu_dir(self, rid: str) -> str:
        if rid in self._rid_dir_cache:
            return self._rid_dir_cache[rid]
        root = self.pangu_root
        pattern = os.path.join(root, "**", f"*{rid}*")
        candidates = [p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p)]
        def has_pair(d: str) -> bool:
            ups = glob.glob(os.path.join(d, "*upper*.nc"))
            sfs = glob.glob(os.path.join(d, "*surface*.nc"))
            return len(ups) > 0 and len(sfs) > 0
        candidates = [d for d in candidates if has_pair(d)]
        if not candidates:
            raise RuntimeError(f"[Pangu] cannot find folder containing both upper/surface for rid={rid} under {root}")
        candidates.sort(key=lambda x: len(x))
        chosen = candidates[0]
        self._rid_dir_cache[rid] = chosen
        return chosen

    def __getitem__(self, idx):
        radar_entry, pangu_entry = self.pairs[idx]
        radar_path = _find_sevir_radar_path(self.sevir_root, radar_entry)
        radar = np.load(radar_path)

        radar_thw = None
        if radar.ndim == 3:
            if radar.shape[-1] >= self.Tin + self.Tout and radar.shape[0] > 64 and radar.shape[1] > 64:
                radar_thw = np.transpose(radar, (2, 0, 1))
            elif radar.shape[0] >= self.Tin + self.Tout:
                radar_thw = radar
            else:
                raise ValueError(f"Unexpected radar shape: {radar.shape} for {radar_path}")
        elif radar.ndim == 4:
            if radar.shape[0] >= self.Tin + self.Tout:
                radar_thw = radar[:, 0, :, :]
            elif radar.shape[-1] >= self.Tin + self.Tout:
                radar_thw = np.transpose(radar[0], (2, 0, 1))
            else:
                raise ValueError(f"Unexpected radar shape: {radar.shape} for {radar_path}")
        else:
            raise ValueError(f"Unexpected radar shape: {radar.shape} for {radar_path}")

        x = radar_thw[:self.Tin]
        y = radar_thw[self.Tin:self.Tin + self.Tout]
        x = _resize_radar_thw_to_fixed(x, RADAR_FIXED_H, RADAR_FIXED_W)[:, None, :, :]
        y = _resize_radar_thw_to_fixed(y, RADAR_FIXED_H, RADAR_FIXED_W)[:, None, :, :]

        rid = _extract_rid_from_radar_filename(radar_path)
        if rid is None:
            raise RuntimeError(f"Cannot extract RID from radar filename: {radar_path}")

        anchor_ts12 = _extract_last_ts12(pangu_entry)
        if not anchor_ts12:
            raise ValueError(f"Cannot parse anchor time from pangu_entry: {pangu_entry}")
        anchor_dt = _to_dt_ts12(anchor_ts12)

        folder = self._pangu_dir(rid)
        pangu_5 = _load_pangu_5(folder, anchor_dt)

        x = (x.astype(np.float32) - self.radar_mean) / max(self.radar_std, 1e-6)
        y = (y.astype(np.float32) - self.radar_mean) / max(self.radar_std, 1e-6)

        p = torch.from_numpy(pangu_5).float()
        p = (p - self.pangu_channel_means) / self.pangu_channel_stds

        return torch.from_numpy(x).float(), p, torch.from_numpy(y).float()


class PaperExactSevirTxtPanguDataset(SevirTxtPanguDataset):
    def __init__(
        self,
        pairs,
        sevir_root,
        pangu_root,
        radar_stats_path,
        pangu_stats_path,
        expected_split_hash,
        Tin=5,
        Tout=20,
    ):
        radar_mean, radar_std, pangu_means, pangu_stds = load_required_stats(
            radar_stats_path,
            pangu_stats_path,
            expected_split_hash=expected_split_hash,
        )
        super().__init__(pairs, sevir_root, pangu_root, Tin=Tin, Tout=Tout)
        self.radar_mean = radar_mean
        self.radar_std = radar_std
        self.pangu_channel_means = torch.from_numpy(
            pangu_means.astype(np.float32)
        ).view(1, FIXED_WFM_C, 1, 1)
        self.pangu_channel_stds = torch.from_numpy(
            pangu_stds.astype(np.float32)
        ).view(1, FIXED_WFM_C, 1, 1)

_STRICT_RADAR_RE = re.compile(r"^(\d{8}_\d{4})_([RS][A-Za-z0-9]+)_\d+_vil\.npy$")
_STRICT_PANGU_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})to"
    r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})_([RS][A-Za-z0-9]+)$"
)

def parse_strict_pangu_start(folder):
    match = _STRICT_PANGU_RE.match(os.path.basename(folder))
    if match is None:
        raise ValueError(f"invalid strict Pangu directory: {folder}")
    return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M")

def build_strict_archive_pairs(radar_root, pangu_root, max_start_delta_minutes=60):
    radar_dir = os.path.join(radar_root, "train") if os.path.isdir(os.path.join(radar_root, "train")) else radar_root
    by_event = {}
    for folder in glob.glob(os.path.join(pangu_root, "**"), recursive=True):
        if not os.path.isdir(folder):
            continue
        match = _STRICT_PANGU_RE.match(os.path.basename(folder))
        if match is None:
            continue
        start = parse_strict_pangu_start(folder)
        by_event.setdefault(match.group(3), []).append((start, folder))
    pairs = []
    audit = {"radar_total": 0, "matched": 0, "missing_event": 0, "time_mismatch": 0, "invalid_radar_name": 0}
    for radar_path in sorted(glob.glob(os.path.join(radar_dir, "*_vil.npy"))):
        audit["radar_total"] += 1
        match = _STRICT_RADAR_RE.match(os.path.basename(radar_path))
        if match is None:
            audit["invalid_radar_name"] += 1
            continue
        radar_time = datetime.strptime(match.group(1), "%Y%m%d_%H%M")
        candidates = by_event.get(match.group(2), [])
        if not candidates:
            audit["missing_event"] += 1
            continue
        start, folder = min(candidates, key=lambda item: abs((item[0] - radar_time).total_seconds()))
        if abs((start - radar_time).total_seconds()) > max_start_delta_minutes * 60:
            audit["time_mismatch"] += 1
            continue
        pairs.append((radar_path, folder))
        audit["matched"] += 1
    return pairs, audit

class ArchiveMatchedDataset(Dataset):
    def __init__(self, radar_root, pangu_root, split="train", val_fraction=0.1, max_delta_minutes=60):
        pairs, self.audit = build_strict_archive_pairs(radar_root, pangu_root, max_delta_minutes)
        cut = max(1, int(len(pairs) * (1.0 - val_fraction)))
        self.pairs = pairs[:cut] if split == "train" else pairs[cut:]
        if not self.pairs:
            raise RuntimeError(f"no strictly matched archive samples: {self.audit}")
        try:
            stats = np.load("norm_stats.npz")
            self.radar_mean = float(stats["radar_mean"])
            self.radar_std = float(stats["radar_std"])
        except Exception:
            self.radar_mean = 0.0
            self.radar_std = 1.0
        try:
            stats = np.load("pangu_channel_stats.npz")
            self.pangu_channel_means = torch.from_numpy(stats["means"].astype(np.float32))
            self.pangu_channel_stds = torch.from_numpy(stats["stds"].astype(np.float32))
        except Exception:
            self.pangu_channel_means = torch.zeros(34)
            self.pangu_channel_stds = torch.ones(34)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        radar_path, folder = self.pairs[idx]
        radar = np.load(radar_path)
        if radar.ndim == 3 and radar.shape[-1] >= 25:
            radar = np.transpose(radar, (2, 0, 1))
        elif radar.ndim == 4 and radar.shape[-1] >= 25:
            radar = np.transpose(radar[0], (2, 0, 1))
        x = _resize_radar_thw_to_fixed(radar[:5], 128, 128)[:, None]
        y = _resize_radar_thw_to_fixed(radar[5:25], 128, 128)[:, None]
        anchor = parse_strict_pangu_start(folder)
        pangu = _load_pangu_5(folder, anchor)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        pangu = torch.from_numpy(pangu).float()
        x, pangu = normalize_archive_sample(x, pangu, self.radar_mean, self.radar_std, self.pangu_channel_means, self.pangu_channel_stds)
        y = (y - self.radar_mean) / max(self.radar_std, 1e-6)
        return x, pangu, y

def build_sevir_pairs_by_date(train_periods_path: str, train_pangu_path: str, test_periods_path: str, test_pangu_path: str, boundary: datetime = datetime(2019, 6, 1, 0, 0)):
    def read_lines(p):
        with open(p, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    train_r = read_lines(train_periods_path)
    train_p = read_lines(train_pangu_path)
    test_r = read_lines(test_periods_path)
    test_p = read_lines(test_pangu_path)

    tr_pairs = [(r, p) for r, p in zip(train_r, train_p) if _is_data_line(r) and _is_data_line(p)]
    te_pairs = [(r, p) for r, p in zip(test_r, test_p) if _is_data_line(r) and _is_data_line(p)]
    all_pairs = tr_pairs + te_pairs

    out_train, out_test = [], []
    for r, p in all_pairs:
        ts12 = _extract_first_ts12(r)
        if not ts12:
            continue
        dt = _to_dt_ts12(ts12)
        if dt < boundary:
            out_train.append((r, p))
        else:
            out_test.append((r, p))
    return out_train, out_test
