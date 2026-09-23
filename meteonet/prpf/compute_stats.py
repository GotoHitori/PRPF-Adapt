import argparse
import os

import numpy as np

from .dataset import ArchiveMatchedDataset, _load_pangu_5, parse_strict_pangu_start


def finalize_stats(total, total_sq, count):
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def compute(radar_root, pangu_root, output_dir, limit=0):
    dataset = ArchiveMatchedDataset(radar_root, pangu_root, "train")
    pairs = dataset.pairs[:limit] if limit else dataset.pairs
    radar_sum = 0.0
    radar_sq = 0.0
    radar_count = 0
    pangu_sum = np.zeros(34, dtype=np.float64)
    pangu_sq = np.zeros(34, dtype=np.float64)
    pangu_count = 0
    for index, (radar_path, folder) in enumerate(pairs, 1):
        radar = np.load(radar_path).astype(np.float64)
        radar_sum += radar.sum()
        radar_sq += np.square(radar).sum()
        radar_count += radar.size
        pangu = _load_pangu_5(folder, parse_strict_pangu_start(folder)).astype(np.float64)
        pangu_sum += pangu.sum(axis=(0, 2, 3))
        pangu_sq += np.square(pangu).sum(axis=(0, 2, 3))
        pangu_count += pangu.shape[0] * pangu.shape[2] * pangu.shape[3]
        if index % 100 == 0 or index == len(pairs):
            print(f"{index}/{len(pairs)}", flush=True)
    radar_mean, radar_std = finalize_stats(np.array([radar_sum]), np.array([radar_sq]), radar_count)
    pangu_mean, pangu_std = finalize_stats(pangu_sum, pangu_sq, pangu_count)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "norm_stats.npz"), radar_mean=radar_mean[0], radar_std=radar_std[0], sample_count=len(pairs))
    np.savez(os.path.join(output_dir, "pangu_channel_stats.npz"), means=pangu_mean.astype(np.float32), stds=pangu_std.astype(np.float32), sample_count=len(pairs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar_root", required=True)
    parser.add_argument("--pangu_root", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    compute(args.radar_root, args.pangu_root, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
