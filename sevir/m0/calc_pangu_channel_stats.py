import os
import glob
import numpy as np
from tqdm import tqdm
from dataset import RadarPanguDataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--radar_dir', required=True)
    parser.add_argument('--pangu_root', required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()


    ds = RadarPanguDataset(
        radar_dir=args.radar_dir,
        pangu_root=args.pangu_root,
        catalog_csv=args.catalog,
        split=args.split
    )


    n_total = len(ds)
    first_pangu = ds[0][1].numpy()                
    T, C, H, W = first_pangu.shape
    print(f"Sample pangu shape: T={T}, C={C}, H={H}, W={W}")


    sum_pangu = np.zeros((C,), dtype=np.float64)
    sum2_pangu = np.zeros((C,), dtype=np.float64)
    count = 0

    for i in tqdm(range(n_total), desc="Pangu stats"):
        pangu = ds[i][1].numpy()                

        arr = pangu.reshape(-1, C)
        sum_pangu += arr.sum(axis=0)
        sum2_pangu += (arr ** 2).sum(axis=0)
        count += arr.shape[0]

    means = sum_pangu / count
    stds = np.sqrt(sum2_pangu / count - means ** 2)

    print("Per-channel means:", means)
    print("Per-channel stds:", stds)
    np.savez('pangu_channel_stats.npz', means=means, stds=stds)

if __name__ == '__main__':
    main()
