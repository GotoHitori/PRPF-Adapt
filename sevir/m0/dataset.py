import os
import re
import glob
import csv
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import torch.nn.functional as F

FIXED_WFM_C = 34
RADAR_IN_T = 5
RADAR_OUT_T = 20
PANGU_T = 5
PANGU_FIXED_H = 32
PANGU_FIXED_W = 32
RADAR_FIXED_H = 128
RADAR_FIXED_W = 128

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
    VARS = ['u_component_of_wind','v_component_of_wind','temperature','specific_humidity','geopotential']
    n_layers_per_var = 6
    cubes = []
    with xr.open_dataset(upper_path) as ds:
        for v in VARS:
            if v in ds:
                val = ds[v].values.astype(np.float32)
                if val.ndim == 4:
                    val = val[0]
                if val.shape[0] < n_layers_per_var:
                    pad = np.zeros((n_layers_per_var - val.shape[0], val.shape[1], val.shape[2]), dtype=np.float32)
                    val = np.concatenate([val, pad], axis=0)
                cubes.append(val[:n_layers_per_var])
            else:
                raise RuntimeError(f"Missing required variable {v} in {upper_path}")
    up = np.concatenate(cubes, axis=0)
    up = _resize_hw_to_fixed(up)
    return up

def _read_surface_4ch(surface_path: str) -> np.ndarray:
    VARS = ['temperature_2m','u_component_of_wind_10m','v_component_of_wind_10m','mean_sea_level_pressure']
    cubes = []
    with xr.open_dataset(surface_path) as ds:
        for v in VARS:
            if v in ds:
                val = ds[v].values.astype(np.float32)
                if val.ndim == 3:
                    val = val[0]
                cubes.append(val[None, :, :])
            else:
                raise RuntimeError(f"Missing required variable {v} in {surface_path}")
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
