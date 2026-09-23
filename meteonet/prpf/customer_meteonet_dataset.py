import os
import re
import glob
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import xarray as xr

RADAR_FIXED_H = 128
RADAR_FIXED_W = 128
PANGU_FIXED_H = 32
PANGU_FIXED_W = 32
PANGU_T = 5
PANGU_C = 34
RADAR_MAX = 70.0

_ID_RE = re.compile(r"^\d{8}_\d{4}$")
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})")

def _read_lines(p: str) -> List[str]:
    with open(p, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _id_to_dt(sid: str) -> datetime:
    return datetime.strptime(sid, "%Y%m%d_%H%M")

def _as_thw(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2:
        return a[None, :, :]
    if a.ndim == 3:
        if a.shape[-1] >= 25:
            return np.transpose(a, (2, 0, 1))
        return a
    if a.ndim == 4:
        if a.shape[0] >= 25:
            return a[:, 0]
        if a.shape[-1] >= 25:
            return np.transpose(a[0], (2, 0, 1))
    raise ValueError(f"Unsupported radar shape {a.shape}")

def _resize_radar_thw(thw: np.ndarray, H: int, W: int) -> np.ndarray:
    t = torch.from_numpy(thw.astype(np.float32)).unsqueeze(1)
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t.squeeze(1).numpy()

def _resize_hw_to_fixed(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(0)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()

def _load_pangu_stats(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.path.exists(path):
        try:
            ps = np.load(path)
            means = ps["means"].astype(np.float32)
            stds = ps["stds"].astype(np.float32)
            
            if means.shape[0] != PANGU_C:
                print(f"Warning: pangu_channel_stats.npz has {means.shape[0]} channels, expected {PANGU_C}. Falling back.")
                raise ValueError("Shape mismatch")
            
            stds = np.where(stds < 1e-6, 1.0, stds)
            m = torch.from_numpy(means).view(1, PANGU_C, 1, 1)
            s = torch.from_numpy(stds).view(1, PANGU_C, 1, 1)
            return m, s
        except Exception:
            pass
    return torch.zeros((1, PANGU_C, 1, 1)), torch.ones((1, PANGU_C, 1, 1))

def _find_radar_path(radar_root: str, sid: str) -> str:
    for sub in ["train", "test", "val"]:
        p = os.path.join(radar_root, sub, f"{sid}.npy")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Radar not found for {sid}")

def _pangu_dir(pangu_root: str, sid: str) -> str:
    d = os.path.join(pangu_root, sid)
    if os.path.isdir(d):
        return d
    raise FileNotFoundError(f"Pangu folder not found: {d}")

def _list_pangu_pairs_in_dir(folder: str) -> List[Tuple[datetime, str, str]]:
    up_files = sorted(glob.glob(os.path.join(folder, "*upper*.nc")))
    sf_files = sorted(glob.glob(os.path.join(folder, "*surface*.nc")))
    if not up_files or not sf_files:
        raise RuntimeError(folder)

    def ts_from_name(p: str) -> Optional[datetime]:
        m = _TS_RE.search(os.path.basename(p))
        if not m:
            return None
        return datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M")

    up_map = {}
    for p in up_files:
        ts = ts_from_name(p)
        if ts:
            up_map[ts] = p

    pairs = []
    for p in sf_files:
        ts = ts_from_name(p)
        if ts and ts in up_map:
            pairs.append((ts, up_map[ts], p))

    if not pairs:
        raise RuntimeError(folder)
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
        start = 0
    else:
        start = max(0, idx - (PANGU_T - 1))
    end = start + PANGU_T
    if end > len(pairs):
        end = len(pairs)
        start = end - PANGU_T
    return pairs[start:end]

def _read_upper_30ch(upper_path: str) -> np.ndarray:
    VARS = ['u_component_of_wind','v_component_of_wind','temperature','specific_humidity','geopotential']
    L = 6
    cubes = []
    with xr.open_dataset(upper_path) as ds:
        for v in VARS:
            val = ds[v].values.astype(np.float32)
            if val.ndim == 4:
                val = val[0]
            if val.shape[0] < L:
                pad = np.zeros((L - val.shape[0], val.shape[1], val.shape[2]), dtype=np.float32)
                val = np.concatenate([val, pad], axis=0)
            cubes.append(val[:L])
    up = np.concatenate(cubes, axis=0)
    return _resize_hw_to_fixed(up, PANGU_FIXED_H, PANGU_FIXED_W)

def _read_surface_4ch(surface_path: str) -> np.ndarray:
    VARS = ['temperature_2m','u_component_of_wind_10m','v_component_of_wind_10m','mean_sea_level_pressure']
    cubes = []
    with xr.open_dataset(surface_path) as ds:
        for v in VARS:
            val = ds[v].values.astype(np.float32)
            if val.ndim == 3:
                val = val[0]
            cubes.append(val[None, :, :])
    sf = np.concatenate(cubes, axis=0)
    return _resize_hw_to_fixed(sf, PANGU_FIXED_H, PANGU_FIXED_W)

def _load_pangu_5(folder: str, anchor_dt: datetime) -> np.ndarray:
    pairs = _list_pangu_pairs_in_dir(folder)
    chosen = _choose_5_around_ts(pairs, anchor_dt)
    frames = []
    for _, up_p, sf_p in chosen:
        up30 = _read_upper_30ch(up_p)
        sf4 = _read_surface_4ch(sf_p)
        frames.append(np.concatenate([up30, sf4], axis=0))
    arr = np.stack(frames, axis=0).astype(np.float32)
    return arr

class MeteoNetTxtPanguDataset(Dataset):
    def __init__(self, id_list: List[str], radar_root: str, pangu_root: str,
                 Tin: int = 5, Tout: int = 20,
                 pangu_stats_path: str = "pangu_channel_stats.npz"):
        self.ids = [s for s in id_list if s and (_ID_RE.match(s) is not None)]
        self.radar_root = radar_root
        self.pangu_root = pangu_root
        self.Tin = int(Tin)
        self.Tout = int(Tout)
        self.radar_mean = 0.0
        self.radar_std = RADAR_MAX
        self.pangu_means, self.pangu_stds = _load_pangu_stats(pangu_stats_path)
        self.pangu_channel_means = self.pangu_means
        self.pangu_channel_stds = self.pangu_stds
        self._pangu_cache = {}

    def __len__(self):
        return len(self.ids)

    def future_max_dbz(self, idx: int) -> float:
        sid = self.ids[idx]
        radar = np.load(_find_radar_path(self.radar_root, sid), mmap_mode="r")
        future = _as_thw(radar)[self.Tin:self.Tin + self.Tout]
        return float(np.nanmax(future))

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        radar_path = _find_radar_path(self.radar_root, sid)
        radar = np.load(radar_path)
        thw = _as_thw(radar).astype(np.float32)
        
        x = thw[:self.Tin]
        y = thw[self.Tin:self.Tin + self.Tout]
        x = _resize_radar_thw(x, RADAR_FIXED_H, RADAR_FIXED_W)[:, None, :, :]
        y = _resize_radar_thw(y, RADAR_FIXED_H, RADAR_FIXED_W)[:, None, :, :]
        
        x = np.clip(x, 0.0, RADAR_MAX) / RADAR_MAX
        y = np.clip(y, 0.0, RADAR_MAX) / RADAR_MAX

        if sid in self._pangu_cache:
            folder = self._pangu_cache[sid]
        else:
            folder = _pangu_dir(self.pangu_root, sid)
            self._pangu_cache[sid] = folder

        anchor_dt = _id_to_dt(sid)
        pangu5 = _load_pangu_5(folder, anchor_dt)
        p = torch.from_numpy(pangu5).float()
        p = (p - self.pangu_means) / self.pangu_stds

        return torch.from_numpy(x).float(), p, torch.from_numpy(y).float()

def load_id_list(path: str) -> List[str]:
    return _read_lines(path)
