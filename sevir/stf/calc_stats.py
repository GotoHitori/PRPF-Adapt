import numpy as np
import torch
from dataset import RadarPanguDataset
from torch.utils.data import DataLoader
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radar_dir', required=True)
    parser.add_argument('--pangu_root', required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    ds = RadarPanguDataset(args.radar_dir, args.pangu_root, args.catalog, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    radar_sum = radar_sqsum = radar_count = 0
    pangu_sum = pangu_sqsum = pangu_count = 0
    for xr, xp, _ in dl:
        radar_sum += xr.sum().item()
        radar_sqsum += (xr ** 2).sum().item()
        radar_count += xr.numel()

        pangu_sum += xp.sum().item()
        pangu_sqsum += (xp ** 2).sum().item()
        pangu_count += xp.numel()
    radar_mean = radar_sum / radar_count
    radar_std = np.sqrt(radar_sqsum / radar_count - radar_mean ** 2)
    pangu_mean = pangu_sum / pangu_count
    pangu_std = np.sqrt(pangu_sqsum / pangu_count - pangu_mean ** 2)

    np.savez('norm_stats.npz',
             radar_mean=radar_mean, radar_std=radar_std,
             pangu_mean=pangu_mean, pangu_std=pangu_std)
    print('radar mean/std:', radar_mean, radar_std)
    print('pangu mean/std:', pangu_mean, pangu_std)

if __name__ == '__main__':
    main()
