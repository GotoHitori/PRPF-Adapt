import os
import math
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from dataset import SevirTxtPanguDataset, build_sevir_pairs_by_date
from model_v2 import PRPF_SetGoGAN_Generator

                                                                               
                      
                                                                               
THRESHOLDS = [16, 74, 133, 160, 181, 219]

def _extract_state(ckpt):
    state = ckpt.get("gen", ckpt.get("model", ckpt.get("state_dict", ckpt))) if isinstance(ckpt, dict) else ckpt
    new_state = {}
    for k, v in state.items():
        new_key = k
        for prefix in ['module.', 'generator.']:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        new_state[new_key] = v
    return new_state

def _denorm(x, mean, std):
    return x * std + mean

def _safe_log10(x: float) -> float:
    return math.log10(max(float(x), 1e-12))

def _psnr_from_mse(mse: float, data_range: float = 255.0) -> float:
    return 20.0 * _safe_log10(data_range) - 10.0 * _safe_log10(mse)

def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def _compute_csi_hss_counts(pred, gt, thr: float):
    pred_b, gt_b = pred >= thr, gt >= thr
    hits = np.logical_and(pred_b, gt_b).sum(dtype=np.int64)
    misses = np.logical_and(~pred_b, gt_b).sum(dtype=np.int64)
    false_alarms = np.logical_and(pred_b, ~gt_b).sum(dtype=np.int64)
    correct_neg = np.logical_and(~pred_b, ~gt_b).sum(dtype=np.int64)
    return hits, misses, false_alarms, correct_neg

def _csi_from_counts(h, m, f) -> float:
    return 0.0 if (h + m + f) == 0 else float(h) / float(h + m + f)

def _hss_from_counts(h, m, f, cn) -> float:
    denom = (m + cn) * (h + m) + (h + f) * (f + cn)
    return 0.0 if denom == 0 else float(2 * (h * cn - f * m)) / float(denom)

def _maybe_resize_to(pred, target_shape):
    Ht, Wt = target_shape[-2], target_shape[-1]
    if pred.shape[-2:] == (Ht, Wt):
        return pred
    return F.interpolate(pred.view(-1, 1, pred.shape[-2], pred.shape[-1]), size=(Ht, Wt), mode="bilinear", align_corners=False).view(pred.shape[0], pred.shape[1], 1, Ht, Wt)

def _gaussian_1d(window_size: int, sigma: float, device):
    coords = torch.arange(window_size, device=device).float() - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    return g / g.sum()

def _gaussian_window_2d(window_size: int, sigma: float, device):
    g1 = _gaussian_1d(window_size, sigma, device)
    g2 = (g1[:, None] * g1[None, :]).contiguous()
    return g2.view(1, 1, window_size, window_size)

def ssim_torch(img1, img2, data_range=255.0, window_size=11, sigma=1.5, eps=1e-6):
    device = img1.device
    w = _gaussian_window_2d(window_size, sigma, device)
    pad = window_size // 2
    mu1 = F.conv2d(img1, w, padding=pad, groups=1)
    mu2 = F.conv2d(img2, w, padding=pad, groups=1)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, w, padding=pad, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, w, padding=pad, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, w, padding=pad, groups=1) - mu12
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / (den + eps)).mean(dim=[1, 2, 3])

                                                                               
                       
                                                                               
def plot_sequence(x_obs, y_true, y_pred, save_path, title_id):
    bounds = [0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 256]
    labels = ["0-16", "16-31", "31-59", "59-74", "74-100", "100-133", "133-160", "160-181", "181-219", "219-255"]
    mids = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(bounds) - 1)]
    colors = ["#3b3b3b", "#00ffff", "#00b7ff", "#0070ff", "#00ff00", "#a8ff00", "#ffff00", "#ffb000", "#ff0000", "#ff00ff"]
    cmap, norm = ListedColormap(colors), BoundaryNorm(bounds, ListedColormap(colors).N)

    fig, axes = plt.subplots(3, 25, figsize=(55, 6.5))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for ax in axes.flatten(): ax.axis('off')

                            
    for t in range(5):
        axes[0, t].imshow(x_obs[t], cmap=cmap, norm=norm, interpolation='nearest')
        axes[0, t].set_title(f"T-{(5-t)*5} Min", fontsize=14)
    axes[0, 0].text(-0.2, 0.5, 'Context: $y$', va='center', ha='right', transform=axes[0, 0].transAxes, fontsize=20)

                            
    for t in range(20):
        axes[1, t+5].imshow(y_true[t], cmap=cmap, norm=norm, interpolation='nearest')
        axes[1, t+5].set_title(f"T+{ (t+1)*5 } Min", fontsize=14)
    axes[1, 5].text(-0.2, 0.5, 'Target: $x$', va='center', ha='right', transform=axes[1, 5].transAxes, fontsize=20)

                                
    mappable = None
    for t in range(20):
        mappable = axes[2, t+5].imshow(y_pred[t], cmap=cmap, norm=norm, interpolation='nearest')
    axes[2, 5].text(-0.2, 0.5, 'Ours', va='center', ha='right', transform=axes[2, 5].transAxes, fontsize=20)

    cbar_ax = fig.add_axes([0.91, 0.15, 0.008, 0.7])
    cbar = fig.colorbar(mappable, cax=cbar_ax, ticks=mids, spacing='proportional')
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=14)

    fig.suptitle(f"Fixed Sequence Visualization [Ours]: {title_id}", fontsize=24, y=0.98)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def plot_layers(L, save_path, title_id, leads=(0, 9, 19)):
    """Latent-layer decomposition figure (STF): rows = selected lead times,
    cols = K layers (500 -> 1000 hPa) + echo-top proxy (highest layer whose
    water exceeds 10% of that frame's max). L: [Tout, K, H, W] numpy."""
    Tsel = [t for t in leads if t < L.shape[0]]
    K = L.shape[1]
    fig, axes = plt.subplots(len(Tsel), K + 1, figsize=(2.2 * (K + 1), 2.4 * len(Tsel)))
    if len(Tsel) == 1:
        axes = axes[None, :]
    level_names = ["500", "600", "700", "850", "925", "1000"][:K]
    vmax = max(float(L.max()), 1e-6)
    for r, t in enumerate(Tsel):
        for k in range(K):
            axes[r, k].imshow(L[t, k], cmap="viridis", vmin=0, vmax=vmax)
            axes[r, k].axis('off')
            if r == 0:
                axes[r, k].set_title(f"{level_names[k]} hPa", fontsize=10)
        thr = 0.1 * max(float(L[t].max()), 1e-6)
        mask = L[t] > thr                                                                    
        has = mask.any(axis=0)
        top_idx = np.argmax(mask, axis=0)                                                     
        echo_top = np.where(has, K - top_idx, 0)                                    
        axes[r, K].imshow(echo_top, cmap="magma", vmin=0, vmax=K)
        axes[r, K].axis('off')
        if r == 0:
            axes[r, K].set_title("echo-top proxy", fontsize=10)
        axes[r, 0].text(-0.25, 0.5, f"T+{(t + 1) * 10}min", va='center', ha='right',
                        rotation=90, transform=axes[r, 0].transAxes, fontsize=10)
    fig.suptitle(f"STF latent layer decomposition: {title_id}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.close(fig)

                                                                               
                      
                                                                               
@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _, test_pairs = build_sevir_pairs_by_date(
        args.train_periods, args.train_pangu, args.test_periods, args.test_pangu, boundary=datetime(2019, 6, 1, 0, 0))

    test_ds = SevirTxtPanguDataset(test_pairs, sevir_root=args.sevir_root, pangu_root=args.pangu_root, Tin=args.Tin, Tout=args.Tout)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    gen = PRPF_SetGoGAN_Generator(
        att_dim=args.att_dim,
        swin_depth=args.swin_depth,
        use_ltam=not args.no_ltam,
        use_csmmsa=not args.no_csmmsa,
        use_crossattn=not args.no_crossattn,
        use_channel_fusion=not args.no_channel_fusion,
        use_swin=not args.no_swin,
        use_temporal=not args.no_temporal,
        use_stf=not args.no_stf,
        stf_layers=args.stf_layers,
        stf_layer_width=args.stf_layer_width,
        stf_permute_layers=args.stf_permute_layers,
        stf_prior=not args.stf_no_prior,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    gen.load_state_dict(_extract_state(ckpt), strict=False)
    gen.eval()

    radar_mean, radar_std = float(getattr(test_ds, "radar_mean", 0.0)), float(getattr(test_ds, "radar_std", 1.0))
    total_mse, total_mae, total_count = 0.0, 0.0, 0
    csi_counts = {thr: np.array([0, 0, 0], dtype=np.int64) for thr in THRESHOLDS}
    hss_counts = {thr: np.array([0, 0, 0, 0], dtype=np.int64) for thr in THRESHOLDS}
                                                                                         
    csi_counts_lt = {thr: np.zeros((args.Tout, 3), dtype=np.int64) for thr in THRESHOLDS}
    psnr_list, ssim_list = [], []

    os.makedirs(args.vis_dir, exist_ok=True)
    vis_count = 0

    print("Running evaluation for 'Ours' (PRPF_SetGoGAN_Generator) on the full test set...")

    for step, (x_obs, x_wfm, y_seq) in enumerate(test_loader):
        x_obs, x_wfm, y_seq = x_obs.to(device).float(), x_wfm.to(device).float(), y_seq.to(device).float()

        pred, _ = gen(x_obs, x_wfm)
            
        if isinstance(pred, (tuple, list)): 
            pred = pred[0]
            
        pred = _maybe_resize_to(pred, y_seq.shape)

        pred_raw = torch.clamp(_denorm(pred, radar_mean, radar_std), 0.0, 255.0)
        gt_raw = torch.clamp(_denorm(y_seq, radar_mean, radar_std), 0.0, 255.0)
        x_obs_raw = torch.clamp(_denorm(x_obs, radar_mean, radar_std), 0.0, 255.0)

        L_lat = None
        if args.vis_layers and (not args.no_stf) and vis_count < 3:
            _, _, L_lat, _ = gen(x_obs, x_wfm, return_stf=True)

        for b_idx in range(pred_raw.shape[0]):
            if vis_count < 10:
                c_obs = x_obs_raw[b_idx].squeeze(1).cpu().numpy() if x_obs_raw.dim() == 5 else x_obs_raw[b_idx].cpu().numpy()
                c_gt = gt_raw[b_idx].squeeze(1).cpu().numpy() if gt_raw.dim() == 5 else gt_raw[b_idx].cpu().numpy()
                c_pred = pred_raw[b_idx].squeeze(1).cpu().numpy() if pred_raw.dim() == 5 else pred_raw[b_idx].cpu().numpy()
                
                seq_id = f"Seq_{vis_count+1:02d}"
                plot_sequence(c_obs, c_gt, c_pred, os.path.join(args.vis_dir, f"ours_{seq_id}.png"), seq_id)
                                                                                        
                if L_lat is not None and vis_count < 3:
                    L_np = L_lat[b_idx].detach().cpu().numpy()                    
                    plot_layers(L_np, os.path.join(args.vis_dir, f"layers_{seq_id}.png"), seq_id)
                vis_count += 1

        total_mse += torch.mean((pred_raw - gt_raw) ** 2).item()
        total_mae += torch.mean(torch.abs(pred_raw - gt_raw)).item()
        total_count += 1

        pred_np, gt_np = _to_numpy(pred_raw), _to_numpy(gt_raw)
        for thr in THRESHOLDS:
            h, m, f, cn = _compute_csi_hss_counts(pred_np, gt_np, thr)
            csi_counts[thr][:3] += [h, m, f]
            hss_counts[thr] += [h, m, f, cn]
                                                  
        T_lt = pred_np.shape[1]
        for t in range(min(T_lt, args.Tout)):
            for thr in THRESHOLDS:
                h, m, f, _ = _compute_csi_hss_counts(pred_np[:, t], gt_np[:, t], thr)
                csi_counts_lt[thr][t] += [h, m, f]

        bt = pred_raw.shape[0] * pred_raw.shape[1]
        p2, g2 = pred_raw.view(bt, 1, pred_raw.shape[-2], pred_raw.shape[-1]), gt_raw.view(bt, 1, gt_raw.shape[-2], gt_raw.shape[-1])
        mse_per = torch.mean((p2 - g2) ** 2, dim=[1, 2, 3]).detach().cpu().numpy()
        psnr_list.append(float(np.mean([_psnr_from_mse(m) for m in mse_per])) if len(mse_per) else 0.0)
        ssim_list.append(float(ssim_torch(p2, g2, data_range=255.0, window_size=11, sigma=1.5).mean().item()))

        if (step + 1) % 10 == 0:
            print(f"\r[{step+1}/{len(test_loader)}] Batches Evaluated...", end="")
    
    print()

    mse_avg = total_mse / max(total_count, 1)
    mae_avg = total_mae / max(total_count, 1)
    psnr_avg = float(np.mean(psnr_list)) if psnr_list else 0.0
    ssim_avg = float(np.mean(ssim_list)) if ssim_list else 0.0

    csi_vals, csi_by_thr = [], {}
    for thr in THRESHOLDS:
        h, m, f = csi_counts[thr][:3]
        csi = _csi_from_counts(h, m, f)
        csi_by_thr[thr] = csi
        csi_vals.append(csi)
    csi_avg = float(np.mean(csi_vals)) if csi_vals else 0.0

    hss_vals = [_hss_from_counts(*hss_counts[thr]) for thr in THRESHOLDS]
    hss_avg = float(np.mean(hss_vals)) if hss_vals else 0.0

    print("\n" + "="*115)
    print(f"Saved 10 fixed sequences plots to: {args.vis_dir}/")
    print("-" * 115)
    print(f"{'Model':<12} | {'MSE':<10} | {'MAE ':<8} | {'PSNR' :<9} | {'SSIM':<9} | {'HSS AVG':<8} | {'CSI-16':<8} | {'CSI-74':<8} | {'CSI-133':<8} | {'CSI-AVG':<8}")
    print("-" * 115)
    print(f"{'Ours':<12} | {mse_avg:<10.4f} | {mae_avg:<8.4f} | {psnr_avg:<9.4f} | {ssim_avg:<9.4f} | {hss_avg:<8.4f} | {csi_by_thr[16]:<8.4f} | {csi_by_thr[74]:<8.4f} | {csi_by_thr[133]:<8.4f} | {csi_avg:<8.4f}")
    print("=" * 115)

                                                                                                  
    print("\nPer-threshold CSI / HSS over ALL six SEVIR VIL thresholds "
          "(CSI-AVG and HSS-AVG are the mean of these six):")
    print(f"  {'thr':>5} | {'CSI':>8} | {'HSS':>8}")
    print("  " + "-"*27)
    for thr in THRESHOLDS:
        h, m, f = csi_counts[thr][:3]
        print(f"  {thr:>5} | {_csi_from_counts(h, m, f):>8.4f} | {_hss_from_counts(*hss_counts[thr]):>8.4f}")
    print(f"  {'AVG':>5} | {csi_avg:>8.4f} | {hss_avg:>8.4f}")

                                                                                               
    print("\nPer-lead-time CSI (minutes ahead; cadence assumed 10 min/frame):")
    header = "  lead |" + "".join([f" CSI-{thr:<4}" for thr in THRESHOLDS]) + f" | {'CSI-AVG':>8}"
    print(header)
    print("  " + "-"*(len(header)-2))
    for t in range(args.Tout):
        row_vals = []
        for thr in THRESHOLDS:
            h, m, f = csi_counts_lt[thr][t]
            row_vals.append(_csi_from_counts(h, m, f))
        lead_min = (t + 1) * 10
        avg_t = float(np.mean(row_vals))
        cells = "".join([f" {v:<7.4f}" for v in row_vals])
        print(f"  {lead_min:>4} |{cells} | {avg_t:>8.4f}")
    print("=" * 115 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sevir_root", type=str, required=True)
    parser.add_argument("--pangu_root", type=str, required=True)
    parser.add_argument("--train_periods", type=str, default="./sevir_train_periods.txt")
    parser.add_argument("--train_pangu", type=str, default="./sevir_train_pangufile.txt")
    parser.add_argument("--test_periods", type=str, default="./sevir_test_periods.txt")
    parser.add_argument("--test_pangu", type=str, default="./sevir_test_pangufile.txt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vis_dir", type=str, default="./vis_output")

    parser.add_argument("--Tin", type=int, default=5)
    parser.add_argument("--Tout", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

                                                                        
    parser.add_argument('--att_dim', type=int, default=128)
    parser.add_argument('--swin_depth', type=int, default=4)
    parser.add_argument('--no_ltam', action='store_true')
    parser.add_argument('--no_csmmsa', action='store_true')
    parser.add_argument('--no_crossattn', action='store_true')
    parser.add_argument('--no_channel_fusion', action='store_true')
    parser.add_argument('--no_swin', action='store_true')
    parser.add_argument('--no_temporal', action='store_true')

                                                          
    parser.add_argument('--no_stf', action='store_true')
    parser.add_argument('--stf_layers', type=int, default=6)
    parser.add_argument('--stf_layer_width', type=int, default=16)
    parser.add_argument('--stf_permute_layers', action='store_true')
    parser.add_argument('--stf_no_prior', action='store_true')
    parser.add_argument('--vis_layers', action='store_true',
                        help="Additionally save latent-layer decomposition figures (K layers + "
                             "echo-top proxy) for the first 3 visualised sequences (STF only).")

    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()