import os
import argparse
import re
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

from dataset import SevirTxtPanguDataset, build_sevir_pairs_by_date
from model_v2 import PRPF_SetGoGAN_Generator, FrameDiscriminator, SeqDiscriminator3D

THRESHOLDS = [16, 74, 133, 160, 181, 219]

                                              
PANGU_U_IDX = 0                                    
PANGU_V_IDX = 6                                                                   
PANGU_WIND_LEVELS = 6
PANGU_STEER_LEVELS = (0, 1, 2)                    

                                             
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

                                                  
class PhysicsInformedAdvectionDiffusionLoss(nn.Module):
    def __init__(self, phase=1, dt=1.0, kappa=0.01, w_phys=0.1, w_smooth=0.05,
                 pangu_means=None, pangu_stds=None, steer_phi=0.2,
                 u_idx=PANGU_U_IDX, v_idx=PANGU_V_IDX, n_wind_levels=PANGU_WIND_LEVELS,
                 steer_levels=PANGU_STEER_LEVELS,
                 device='cuda'):
        super().__init__()
        self.phase = phase
        self.dt = dt
        self.kappa = kappa
        self.w_phys = w_phys
        self.w_smooth = w_smooth

        self.steer_phi = float(steer_phi)
        self.u_idx = int(u_idx)
        self.v_idx = int(v_idx)
        self.n_wind_levels = int(n_wind_levels)
        self.steer_levels = tuple(int(l) for l in steer_levels)

        if pangu_means is None or pangu_stds is None:
            self.register_buffer('pangu_means', None)
            self.register_buffer('pangu_stds', None)
            self._has_pangu_stats = False
        else:
            self.register_buffer('pangu_means', pangu_means.clone().float().view(1, -1, 1, 1))
            self.register_buffer('pangu_stds', pangu_stds.clone().float().view(1, -1, 1, 1))
            self._has_pangu_stats = True

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).float().view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).float().view(1, 1, 3, 3) / 8.0
        laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device).float().view(1, 1, 3, 3)
        self.register_buffer('Kx', sobel_x)
        self.register_buffer('Ky', sobel_y)
        self.register_buffer('Klap', laplace)

    def compute_grads(self, img):
        gx = F.conv2d(img, self.Kx, padding=1)
        gy = F.conv2d(img, self.Ky, padding=1)
        lap = F.conv2d(img, self.Klap, padding=1)
        return gx, gy, lap

    @torch.no_grad()
    def compute_u_steer(self, x_wfm, H, W):
        B, Tw, C, Hp, Wp = x_wfm.shape
        xw = x_wfm.reshape(B * Tw, C, Hp, Wp)
        if self._has_pangu_stats:
            xw = xw * self.pangu_stds.to(xw.device) + self.pangu_means.to(xw.device)

        u_ch = [self.u_idx + l for l in self.steer_levels]
        v_ch = [self.v_idx + l for l in self.steer_levels]
        u = xw[:, u_ch].mean(dim=1, keepdim=True)
        v = xw[:, v_ch].mean(dim=1, keepdim=True)

        u = u.reshape(B, Tw, 1, Hp, Wp).mean(dim=1)
        v = v.reshape(B, Tw, 1, Hp, Wp).mean(dim=1)

        u = F.interpolate(u, size=(H, W), mode='bilinear', align_corners=False)
        v = F.interpolate(v, size=(H, W), mode='bilinear', align_corners=False)
        return self.steer_phi * u, self.steer_phi * v

    def forward(self, pred_seq, target_seq, res_flow, x_wfm, mean, std):
        y_raw = target_seq * std + mean
        y_raw = torch.clamp(y_raw, 0.0, 255.0)

        w = torch.ones_like(target_seq)

        if self.phase == 1:
            w[y_raw >= 16] = 2.0
            w[y_raw >= 74] = 4.0
            w[y_raw >= 133] = 8.0
            w[y_raw >= 160] = 12.0
        else:
            w[y_raw >= 16] = 1.0
            w[y_raw >= 74] = 1.5
            w[y_raw >= 133] = 2.0
            w[y_raw >= 160] = 3.0

        loss_recon = torch.mean(w * (pred_seq - target_seq)**2) + torch.mean(w * torch.abs(pred_seq - target_seq))

        B, T, C, H, W = pred_seq.shape
        curr_phi = pred_seq[:, :-1].reshape(-1, 1, H, W)
        next_phi = pred_seq[:, 1:].reshape(-1, 1, H, W)
        dphi_dt = (next_phi - curr_phi) / self.dt

        gx, gy, lap = self.compute_grads(curr_phi)

        u_res = res_flow[:, :-1, 0:1].reshape(-1, 1, H, W)
        v_res = res_flow[:, :-1, 1:2].reshape(-1, 1, H, W)

        u_steer, v_steer = self.compute_u_steer(x_wfm, H, W)
        u_steer = u_steer.unsqueeze(1).expand(B, T - 1, 1, H, W).reshape(-1, 1, H, W)
        v_steer = v_steer.unsqueeze(1).expand(B, T - 1, 1, H, W).reshape(-1, 1, H, W)

        u_tot = u_steer + u_res
        v_tot = v_steer + v_res

        advection = u_tot * gx + v_tot * gy
        diffusion = self.kappa * lap
        residual = dphi_dt + advection - diffusion
        loss_phys = torch.mean(torch.sqrt(residual**2 + 1e-6))

        rf_flat = res_flow.reshape(-1, 2, H, W)
        gfx, gfy, _ = self.compute_grads(rf_flat[:, 0:1])
        gvx, gvy, _ = self.compute_grads(rf_flat[:, 1:2])
        loss_smooth = torch.mean(torch.abs(gfx) + torch.abs(gfy) + torch.abs(gvx) + torch.abs(gvy))

        total_loss = loss_recon + self.w_phys * loss_phys + self.w_smooth * loss_smooth
        return total_loss, loss_phys.item()

def hinge_g_loss(fake_logits):
    return -fake_logits.mean()

def get_adv_weight(epoch, total_epochs, base_weight, phase):
    if phase == 2:
        return base_weight

    warmup_epochs = int(total_epochs * 0.2)
    if epoch <= warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_weight * progress

def train_one_epoch(gen, ema, d_frame, d_seq, g_opt, d_opt, g_scheduler, d_scheduler, phys_crit, loader, device, current_lambda_adv, mean, std):
    gen.train()
    d_frame.train()
    d_seq.train()
    stats = {'g': 0.0, 'd': 0.0, 'adv_f': 0.0, 'adv_s': 0.0, 'phys': 0.0}
    n = 0

    pbar = tqdm(loader, dynamic_ncols=True, desc="Train", leave=False)

    for x_obs, x_wfm, y_seq in pbar:
        x_obs = x_obs.to(device).float()
        x_wfm = x_wfm.to(device).float()
        y_seq = y_seq.to(device).float()

        B, T_out, _, H, W = y_seq.shape

        with torch.no_grad():
            fake_radar, _ = gen(x_obs, x_wfm)

        if current_lambda_adv[0] > 0 or current_lambda_adv[1] > 0:
            d_opt.zero_grad()
            d_real_f = F.relu(1.0 - d_frame(y_seq.view(-1, 1, H, W))).mean()
            d_fake_f = F.relu(1.0 + d_frame(fake_radar.detach().view(-1, 1, H, W))).mean()
            loss_d_frame = d_real_f + d_fake_f

            d_real_s = F.relu(1.0 - d_seq(y_seq)).mean()
            d_fake_s = F.relu(1.0 + d_seq(fake_radar.detach())).mean()
            loss_d_seq = d_real_s + d_fake_s

            loss_d = loss_d_frame + loss_d_seq
            loss_d.backward()
            d_opt.step()
            stats['d'] += loss_d.item()

        d_scheduler.step()

        g_opt.zero_grad()
        fake_radar, res_flow = gen(x_obs, x_wfm)

        loss_recon_phys, l_phys_val = phys_crit(fake_radar, y_seq, res_flow, x_wfm, mean, std)

        loss_g = loss_recon_phys
        loss_adv_f_val = 0.0
        loss_adv_s_val = 0.0

        if current_lambda_adv[0] > 0:
            logits_f = d_frame(fake_radar.view(-1, 1, H, W))
            loss_adv_f = hinge_g_loss(logits_f)
            loss_g += current_lambda_adv[0] * loss_adv_f
            loss_adv_f_val = loss_adv_f.item()

        if current_lambda_adv[1] > 0:
            logits_s = d_seq(fake_radar)
            loss_adv_s = hinge_g_loss(logits_s)
            loss_g += current_lambda_adv[1] * loss_adv_s
            loss_adv_s_val = loss_adv_s.item()

        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        g_opt.step()
        g_scheduler.step()

        ema.update()

        stats['g'] += loss_g.item()
        stats['adv_f'] += loss_adv_f_val
        stats['adv_s'] += loss_adv_s_val
        stats['phys'] += l_phys_val
        n += 1

        pbar.set_postfix({'g_loss': f"{loss_g.item():.4f}", 'lr': f"{g_opt.param_groups[0]['lr']:.6f}"})

    for k in stats:
        stats[k] /= max(n, 1)
    return stats

@torch.no_grad()
def evaluate(gen, loader, device, mean, std):
    gen.eval()
    mse_sum = 0.0
    n_items = 0

    csi_counts = {thr: [0, 0, 0] for thr in THRESHOLDS}

    pbar = tqdm(loader, dynamic_ncols=True, desc="Val", leave=False)

    for x_obs, x_wfm, y_seq in pbar:
        x_obs = x_obs.to(device).float()
        x_wfm = x_wfm.to(device).float()
        y_seq = y_seq.to(device).float()

        pred, _ = gen(x_obs, x_wfm)

        pred_raw = torch.clamp(pred * std + mean, 0.0, 255.0)
        y_raw = torch.clamp(y_seq * std + mean, 0.0, 255.0)

        diff = pred_raw - y_raw
        mse_sum += (diff ** 2).sum().item()
        n_items += diff.numel()

        for thr in THRESHOLDS:
            p_b = pred_raw >= thr
            g_b = y_raw >= thr

            h = (p_b & g_b).sum().item()
            m = (~p_b & g_b).sum().item()
            f = (p_b & ~g_b).sum().item()

            csi_counts[thr][0] += h
            csi_counts[thr][1] += m
            csi_counts[thr][2] += f

    mse_avg = mse_sum / max(1, n_items)

    csi_vals = []
    csi_133 = 0.0
    for thr in THRESHOLDS:
        h, m, f = csi_counts[thr]
        denom = h + m + f
        csi = 0.0 if denom == 0 else float(h) / float(denom)
        csi_vals.append(csi)
        if thr == 133:
            csi_133 = csi

    csi_avg = sum(csi_vals) / len(csi_vals)

    return mse_avg, csi_avg, csi_133

def _extract_dp_state(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()

def _extract_state_for_load(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '').replace('generator.', '')
        new_state[new_key] = v
    return new_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="1: Phase 1 Training, 2: Phase 2 Fine-tuning")
    parser.add_argument("--sevir_root", type=str, required=True)
    parser.add_argument("--pangu_root", type=str, required=True)
    parser.add_argument("--train_periods", type=str, required=True)
    parser.add_argument("--train_pangu", type=str, required=True)
    parser.add_argument("--test_periods", type=str, required=True)
    parser.add_argument("--test_pangu", type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints_prpf')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to a checkpoint. By default only the WEIGHTS are warm-started "
                             "(epoch restarts from 1, fresh optimizer/scheduler, best reset). "
                             "Add --resume to continue the previous run instead.")
    parser.add_argument('--resume', action='store_true',
                        help="Resume the SAME run: also restore optimizer/scheduler/epoch/best from "
                             "--checkpoint. Omit this to start a fresh phase that only warm-starts weights.")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_adv_f', type=float, default=0.01)
    parser.add_argument('--lambda_adv_s', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--w_phys', type=float, default=0.1)
    parser.add_argument('--w_smooth', type=float, default=0.05)
    parser.add_argument('--steer_phi', type=float, default=0.2)
    parser.add_argument('--att_dim', type=int, default=128)
    parser.add_argument('--swin_depth', type=int, default=4)
    parser.add_argument('--no_ltam', action='store_true')
    parser.add_argument('--no_csmmsa', action='store_true')
    parser.add_argument('--no_crossattn', action='store_true')
    parser.add_argument('--no_channel_fusion', action='store_true')
    parser.add_argument('--no_swin', action='store_true')
    parser.add_argument('--no_temporal', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_pairs, test_pairs = build_sevir_pairs_by_date(
        args.train_periods, args.train_pangu, args.test_periods, args.test_pangu,
        boundary=datetime(2019, 6, 1, 0, 0)
    )

    _TS12_RE = re.compile(r"(20\d{10})")

    def extract_time_for_sort(pair):
        line = pair[0]
        match = _TS12_RE.search(line)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d%H%M")
        return datetime.min

    train_pairs.sort(key=extract_time_for_sort)
    real_train_pairs = train_pairs
    val_pairs = test_pairs

    print(f"--- Running Phase {args.phase} ---")
    print(f"Total training pairs (100%): {len(real_train_pairs)}")
    print(f"Total validation pairs (Test Set): {len(val_pairs)}")

    train_ds = SevirTxtPanguDataset(real_train_pairs, args.sevir_root, args.pangu_root, Tin=5, Tout=20)
    val_ds = SevirTxtPanguDataset(val_pairs, args.sevir_root, args.pangu_root, Tin=5, Tout=20)

    r_mean = getattr(train_ds, "radar_mean", 0.0)
    r_std = getattr(train_ds, "radar_std", 1.0)
    p_means = getattr(train_ds, "pangu_channel_means", None)
    p_stds = getattr(train_ds, "pangu_channel_stds", None)
    if p_means is None or p_stds is None:
        warnings.warn("Pangu channel stats unavailable; U_steer will be computed from normalized winds.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    gen = PRPF_SetGoGAN_Generator(
        att_dim=args.att_dim,
        swin_depth=args.swin_depth,
        use_ltam=not args.no_ltam,
        use_csmmsa=not args.no_csmmsa,
        use_crossattn=not args.no_crossattn,
        use_channel_fusion=not args.no_channel_fusion,
        use_swin=not args.no_swin,
        use_temporal=not args.no_temporal,
    ).to(device)

    n_gen_params = sum(p.numel() for p in gen.parameters())
    print(f"[model] generator parameters: {n_gen_params/1e6:.2f}M")

    d_frame = FrameDiscriminator().to(device)
    d_seq = SeqDiscriminator3D().to(device)

    ema = EMA(gen, decay=0.999)

    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen)
        d_frame = nn.DataParallel(d_frame)
        d_seq = nn.DataParallel(d_seq)

    phys_crit = PhysicsInformedAdvectionDiffusionLoss(
        phase=args.phase, device=device, w_phys=args.w_phys, w_smooth=args.w_smooth,
        pangu_means=p_means, pangu_stds=p_stds, steer_phi=args.steer_phi
    )

    betas = (args.beta1, args.beta2)
    g_opt = optim.AdamW(gen.parameters(), lr=args.lr_g, betas=betas, weight_decay=1e-4)
    d_opt = optim.AdamW(list(d_frame.parameters()) + list(d_seq.parameters()), lr=args.lr_d, betas=betas, weight_decay=1e-4)

    total_steps = len(train_loader) * args.epochs

    if args.phase == 1:
        g_scheduler = optim.lr_scheduler.OneCycleLR(g_opt, max_lr=args.lr_g, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
        d_scheduler = optim.lr_scheduler.OneCycleLR(d_opt, max_lr=args.lr_d, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
    else:
        g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=total_steps, eta_min=1e-6)
        d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=total_steps, eta_min=1e-6)

                                                                                
                        
                                                                                 
                                                                               
                                                                                   
                                                                         
                                                                                 
                                                                          
                                                                                
    start_epoch = 1
    best_val_csi = -1.0

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location='cpu')

                                                    
        gen_state = _extract_state_for_load(ckpt.get('gen', ckpt))
        (gen.module if isinstance(gen, nn.DataParallel) else gen).load_state_dict(gen_state, strict=False)

        if 'd_frame' in ckpt:
            df_state = _extract_state_for_load(ckpt['d_frame'])
            (d_frame.module if isinstance(d_frame, nn.DataParallel) else d_frame).load_state_dict(df_state, strict=False)
        if 'd_seq' in ckpt:
            ds_state = _extract_state_for_load(ckpt['d_seq'])
            (d_seq.module if isinstance(d_seq, nn.DataParallel) else d_seq).load_state_dict(ds_state, strict=False)

                                                                 
        if args.resume:
            if 'g_opt' in ckpt:
                g_opt.load_state_dict(ckpt['g_opt'])
            if 'd_opt' in ckpt:
                d_opt.load_state_dict(ckpt['d_opt'])
            if 'g_scheduler' in ckpt:
                g_scheduler.load_state_dict(ckpt['g_scheduler'])
            if 'd_scheduler' in ckpt:
                d_scheduler.load_state_dict(ckpt['d_scheduler'])
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
            if 'best_val_csi' in ckpt:
                best_val_csi = ckpt['best_val_csi']
            elif 'val_csi_avg' in ckpt:
                best_val_csi = ckpt['val_csi_avg']
            print(f"[resume] Continuing the same run from Epoch {start_epoch}, "
                  f"Best Val CSI: {best_val_csi:.4f} (optimizer/scheduler restored).")
        else:
            print("[warm-start] Loaded weights only. Fresh run: epoch from 1, "
                  "new optimizer/scheduler, best reset to -1.")

                                                           
        ema = EMA(gen, decay=0.999)
                                                                                

    for epoch in range(start_epoch, args.epochs + 1):

        cur_lambda_f = get_adv_weight(epoch, args.epochs, args.lambda_adv_f, args.phase)
        cur_lambda_s = get_adv_weight(epoch, args.epochs, args.lambda_adv_s, args.phase)

        stats = train_one_epoch(
            gen, ema, d_frame, d_seq, g_opt, d_opt, g_scheduler, d_scheduler,
            phys_crit, train_loader, device, (cur_lambda_f, cur_lambda_s), r_mean, r_std
        )

        ema.apply_shadow()
        val_mse, val_csi_avg, val_csi_133 = evaluate(gen, val_loader, device, r_mean, r_std)
        ema.restore()

        print(f"Epoch {epoch:03d}/{args.epochs} | G={stats['g']:.4f} D={stats['d']:.4f} (AdvW: {cur_lambda_f:.4f}) | Val MSE={val_mse:.4f} CSI-AVG={val_csi_avg:.4f} CSI-133={val_csi_133:.4f}")

                                  
        save_dict = {
            'epoch': epoch,
            'gen': _extract_dp_state(gen),
            'd_frame': _extract_dp_state(d_frame),
            'd_seq': _extract_dp_state(d_seq),
            'g_opt': g_opt.state_dict(),
            'd_opt': d_opt.state_dict(),
            'g_scheduler': g_scheduler.state_dict(),
            'd_scheduler': d_scheduler.state_dict(),
            'val_mse': float(val_mse),
            'val_csi_avg': float(val_csi_avg),
            'val_csi_133': float(val_csi_133),
            'best_val_csi': float(max(best_val_csi, val_csi_avg))
        }

                                                            
        ema.apply_shadow()
        torch.save(save_dict, os.path.join(args.save_dir, "latest_model.pth"))

                         
        if val_csi_avg > best_val_csi:
            best_val_csi = val_csi_avg
            torch.save(save_dict, os.path.join(args.save_dir, "best_model.pth"))
            print(f"--> Best EMA Model Saved (Val CSI-AVG: {best_val_csi:.4f}, CSI-133: {val_csi_133:.4f})")

        ema.restore()

if __name__ == "__main__":
    main()
