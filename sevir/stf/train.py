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
        """Large-scale steering flow U_steer derived from frozen Pangu winds.
        x_wfm: [B, Tw, C, Hp, Wp] (channel-normalized, as produced by the dataset).
        Uses only the MID-TROPOSPHERIC levels (500-700 hPa) -- the layer that
        governs storm motion -- rather than the full atmospheric column.
        Returns (u_steer, v_steer), each [B, 1, H, W], in pixels/frame (scaled by steer_phi).
        U_steer is input-derived and carries no gradient (a constant advection field).
        """
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

                                                                  
                                                                             
                                                                            
                                                                             
                                                                           
                                                                            
                                                                           
                                                     
class PerLayerAdvectionLoss(nn.Module):
    def __init__(self, dt=1.0, kappa=0.01, pangu_means=None, pangu_stds=None,
                 steer_phi=0.2, layer_perm=None, shared_wind=False,
                 u_idx=PANGU_U_IDX, v_idx=PANGU_V_IDX,
                 steer_levels=PANGU_STEER_LEVELS, device='cuda'):
        super().__init__()
        self.dt = dt
        self.kappa = kappa
        self.steer_phi = float(steer_phi)
        self.u_idx, self.v_idx = int(u_idx), int(v_idx)
        self.steer_levels = tuple(int(l) for l in steer_levels)
        self.layer_perm = list(layer_perm) if layer_perm is not None else None
                                                                       
                                                                       
        self.shared_wind = bool(shared_wind)

        if pangu_means is None or pangu_stds is None:
            self._has_stats = False
        else:
            self.register_buffer('pangu_means', pangu_means.clone().float().view(1, -1, 1, 1))
            self.register_buffer('pangu_stds', pangu_stds.clone().float().view(1, -1, 1, 1))
            self._has_stats = True

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).float().view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).float().view(1, 1, 3, 3)
        sobel_y = sobel_y / 8.0
        laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device).float().view(1, 1, 3, 3)
        self.register_buffer('Kx', sobel_x)
        self.register_buffer('Ky', sobel_y)
        self.register_buffer('Klap', laplace)

    @torch.no_grad()
    def compute_layer_winds(self, x_wfm, H, W, K):
        """Per-layer steering winds, [B, K, H, W] each, in pixels/frame.
        Layer k uses level layer_perm[k]'s wind so the scrambled-pairing
        ablation stays consistent between FiLM conditioning and physics."""
        B, Tw, C, Hp, Wp = x_wfm.shape
        xw = x_wfm.reshape(B * Tw, C, Hp, Wp)
        if self._has_stats:
            xw = xw * self.pangu_stds.to(xw.device) + self.pangu_means.to(xw.device)
        xw = xw.reshape(B, Tw, C, Hp, Wp).mean(dim=1)                                       

        if self.shared_wind:
            u_ch = [self.u_idx + l for l in self.steer_levels]
            v_ch = [self.v_idx + l for l in self.steer_levels]
            u = xw[:, u_ch].mean(dim=1, keepdim=True).expand(B, K, Hp, Wp)
            v = xw[:, v_ch].mean(dim=1, keepdim=True).expand(B, K, Hp, Wp)
        else:
            perm = self.layer_perm if self.layer_perm is not None else list(range(K))
            u = torch.stack([xw[:, self.u_idx + perm[k]] for k in range(K)], dim=1)
            v = torch.stack([xw[:, self.v_idx + perm[k]] for k in range(K)], dim=1)

        u = F.interpolate(u, size=(H, W), mode='bilinear', align_corners=False)
        v = F.interpolate(v, size=(H, W), mode='bilinear', align_corners=False)
        return self.steer_phi * u, self.steer_phi * v

    def forward(self, L, res_flow, x_wfm):
        """L: [B, T, K, H, W] latent water layers (physical, >= 0);
        res_flow: [B, T, 2, H, W] shared learnable residual flow."""
        B, T, K, H, W = L.shape
        u_l, v_l = self.compute_layer_winds(x_wfm, H, W, K)                        

        curr = L[:, :-1].reshape(-1, 1, H, W)                                              
        nxt = L[:, 1:].reshape(-1, 1, H, W)
        dphi_dt = (nxt - curr) / self.dt

        gx = F.conv2d(curr, self.Kx, padding=1)
        gy = F.conv2d(curr, self.Ky, padding=1)
        lap = F.conv2d(curr, self.Klap, padding=1)

                                                                                
        u_tot = (u_l.unsqueeze(1) + res_flow[:, :-1, 0:1]).reshape(-1, 1, H, W)
        v_tot = (v_l.unsqueeze(1) + res_flow[:, :-1, 1:2]).reshape(-1, 1, H, W)

        residual = dphi_dt + u_tot * gx + v_tot * gy - self.kappa * lap
        return torch.mean(torch.sqrt(residual ** 2 + 1e-6))

def stf_profile_losses(L, prior, mass_c=0.1):
    """Profile-shape prior + vertical smoothness for the latent layer stack.
    L: [B, T, K, H, W]; prior: [B, T, K, Hp, Wp] (softmax over K, coarse grid).
    The shape loss is soft-masked by column mass (profile shape is meaningless
    in dry columns); prior buys identifiability + interpretability, VIL skill
    does not depend on it (see --stf_no_prior ablation)."""
    B, T, K, H, W = L.shape
    Hp, Wp = prior.shape[-2:]
    Ld = F.adaptive_avg_pool2d(L.reshape(B * T, K, H, W), (Hp, Wp))
    col = Ld.sum(dim=1, keepdim=True)
    shape = Ld / (col + 1e-6)
    wmask = col / (col + mass_c)
    l_prof = (wmask * (shape - prior.reshape(B * T, K, Hp, Wp)) ** 2).mean()
    l_zsmooth = ((L[:, :, 1:] - L[:, :, :-1]) ** 2).mean()
    return l_prof, l_zsmooth

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

def train_one_epoch(gen, ema, d_frame, d_seq, g_opt, d_opt, g_scheduler, d_scheduler, phys_crit, loader, device, current_lambda_adv, mean, std, stf=None):
    gen.train()
    d_frame.train()
    d_seq.train()
    stats = {'g': 0.0, 'd': 0.0, 'adv_f': 0.0, 'adv_s': 0.0, 'phys': 0.0, 'lphys': 0.0, 'prof': 0.0}
    n = 0
    stf_on = stf is not None and stf.get('on', False)

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
        if stf_on:
            fake_radar, res_flow, L_layers, prof_prior = gen(x_obs, x_wfm, return_stf=True)
        else:
            fake_radar, res_flow = gen(x_obs, x_wfm)

        loss_recon_phys, l_phys_val = phys_crit(fake_radar, y_seq, res_flow, x_wfm, mean, std)

        loss_g = loss_recon_phys
        loss_adv_f_val = 0.0
        loss_adv_s_val = 0.0

                                                                                 
        if stf_on:
            l_layer = stf['crit'](L_layers, res_flow, x_wfm)
            loss_g = loss_g + stf['w_layer'] * l_layer
            stats['lphys'] += l_layer.item()
            if prof_prior.numel() > 0:
                l_prof, l_zs = stf_profile_losses(L_layers, prof_prior)
                loss_g = loss_g + stf['w_prof'] * l_prof + stf['w_zsmooth'] * l_zs
                stats['prof'] += l_prof.item()

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
        new_key = k.replace('module.', '')
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
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to pre-trained model for Phase 2")
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
    parser.add_argument('--steer_phi', type=float, default=0.2,
                        help="Maps Pangu winds (m/s) to pixels/frame for U_steer. Theoretical SEVIR "
                             "value = dt/pixel = 600s/3000m = 0.2 (10-min cadence, 384km->128px). "
                             "Recalibrate per dataset; see calibrate_steer_phi.py.")

                                     
    parser.add_argument('--att_dim', type=int, default=128,
                        help="Internal feature width C (fusion width = 2*att_dim). 128 -> ~7.7M generator params.")
    parser.add_argument('--swin_depth', type=int, default=4, help="Number of Swin blocks in spatial refinement.")
                                                                                       
    parser.add_argument('--no_ltam', action='store_true', help="Ablate LTAM radar-history modulation.")
    parser.add_argument('--no_csmmsa', action='store_true', help="Ablate the CS-MMSA local cross-scale modulation branch.")
    parser.add_argument('--no_crossattn', action='store_true', help="Ablate the global cross-attention alignment branch.")
    parser.add_argument('--no_channel_fusion', action='store_true', help="Ablate bidirectional channel fusion + CSA.")
    parser.add_argument('--no_swin', action='store_true', help="Ablate Swin spatial refinement.")
    parser.add_argument('--no_temporal', action='store_true', help="Ablate temporal refinement (3D conv + temporal attention).")

                                                                         
    parser.add_argument('--no_stf', action='store_true',
                        help="Ablate STF entirely (K=1 collapse: original DualHeadDecoder).")
    parser.add_argument('--stf_layers', type=int, default=6,
                        help="Number of latent water layers K (mapped 1:1 to the 6 Pangu pressure levels).")
    parser.add_argument('--stf_layer_width', type=int, default=16,
                        help="Per-layer channel width g in the grouped decoder branch.")
    parser.add_argument('--stf_permute_layers', action='store_true',
                        help="ABLATION: scramble the level<->layer pairing (reversed derangement) in BOTH "
                             "the FiLM conditioning and the per-layer physics winds. Isolates 'vertical "
                             "semantic alignment' from mere multi-channel capacity.")
    parser.add_argument('--stf_no_diffadv', action='store_true',
                        help="ABLATION: all layers share the mid-tropospheric steering wind "
                             "(differential advection OFF -> tomography loses its shear parallax).")
    parser.add_argument('--stf_no_prior', action='store_true',
                        help="ABLATION: drop the climatological profile-shape prior head "
                             "(VIL skill should hold; layer visualisations degrade).")
    parser.add_argument('--w_layer_phys', type=float, default=0.05,
                        help="Weight of the per-layer differential-advection physics loss.")
    parser.add_argument('--w_prof', type=float, default=0.01,
                        help="Weight of the profile-shape prior loss.")
    parser.add_argument('--w_zsmooth', type=float, default=0.001,
                        help="Weight of the vertical smoothness regulariser on the layer stack.")

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
        warnings.warn("Pangu channel stats unavailable; U_steer will be computed from normalized winds "
                      "(physically meaningful steering requires pangu_channel_stats.npz).")

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
        use_stf=not args.no_stf,
        stf_layers=args.stf_layers,
        stf_layer_width=args.stf_layer_width,
        stf_permute_layers=args.stf_permute_layers,
        stf_prior=not args.stf_no_prior,
    ).to(device)
    n_gen_params = sum(p.numel() for p in gen.parameters())
    print(f"[model] generator parameters: {n_gen_params/1e6:.2f}M "
          f"(att_dim={args.att_dim}, swin_depth={args.swin_depth}, "
          f"ltam={not args.no_ltam}, csmmsa={not args.no_csmmsa}, crossattn={not args.no_crossattn}, "
          f"channel_fusion={not args.no_channel_fusion}, swin={not args.no_swin}, temporal={not args.no_temporal}, "
          f"stf={not args.no_stf}, K={args.stf_layers}, permute={args.stf_permute_layers}, "
          f"diffadv={not args.stf_no_diffadv}, prior={not args.stf_no_prior})")
    d_frame = FrameDiscriminator().to(device)
    d_seq = SeqDiscriminator3D().to(device)

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading pretrained weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        gen.load_state_dict(_extract_state_for_load(ckpt['gen']), strict=False)
        if 'd_frame' in ckpt:
            d_frame.load_state_dict(_extract_state_for_load(ckpt['d_frame']), strict=False)
        if 'd_seq' in ckpt:
            d_seq.load_state_dict(_extract_state_for_load(ckpt['d_seq']), strict=False)
        print("Pretrained weights successfully loaded!")

    ema = EMA(gen, decay=0.999)

    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen)
        d_frame = nn.DataParallel(d_frame)
        d_seq = nn.DataParallel(d_seq)


    phys_crit = PhysicsInformedAdvectionDiffusionLoss(
        phase=args.phase, device=device, w_phys=args.w_phys, w_smooth=args.w_smooth,
        pangu_means=p_means, pangu_stds=p_stds, steer_phi=args.steer_phi
    )

                                                                            
                                                               
    stf = {'on': not args.no_stf}
    if stf['on']:
        from model_v2 import stf_build_layer_perm
        layer_perm = stf_build_layer_perm(args.stf_layers, args.stf_permute_layers)
        stf['crit'] = PerLayerAdvectionLoss(
            pangu_means=p_means, pangu_stds=p_stds, steer_phi=args.steer_phi,
            layer_perm=layer_perm, shared_wind=args.stf_no_diffadv, device=device
        ).to(device)
        stf['w_layer'] = args.w_layer_phys
        stf['w_prof'] = args.w_prof
        stf['w_zsmooth'] = args.w_zsmooth
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


    best_val_csi = -1.0

    for epoch in range(1, args.epochs + 1):

        cur_lambda_f = get_adv_weight(epoch, args.epochs, args.lambda_adv_f, args.phase)
        cur_lambda_s = get_adv_weight(epoch, args.epochs, args.lambda_adv_s, args.phase)

        stats = train_one_epoch(
            gen, ema, d_frame, d_seq, g_opt, d_opt, g_scheduler, d_scheduler,
            phys_crit, train_loader, device, (cur_lambda_f, cur_lambda_s), r_mean, r_std, stf=stf
        )

        ema.apply_shadow()
        val_mse, val_csi_avg, val_csi_133 = evaluate(gen, val_loader, device, r_mean, r_std)
        ema.restore()

        stf_log = f" LPhys={stats['lphys']:.4f} Prof={stats['prof']:.4f}" if stf['on'] else ""
        print(f"Epoch {epoch:03d}/{args.epochs} | G={stats['g']:.4f} D={stats['d']:.4f} (AdvW: {cur_lambda_f:.4f}){stf_log} | Val MSE={val_mse:.4f} CSI-AVG={val_csi_avg:.4f} CSI-133={val_csi_133:.4f}")

        if val_csi_avg > best_val_csi:
            best_val_csi = val_csi_avg
            ema.apply_shadow()
            torch.save({
                'epoch': epoch,
                'gen': _extract_dp_state(gen),
                'd_frame': _extract_dp_state(d_frame),
                'd_seq': _extract_dp_state(d_seq),
                'g_opt': g_opt.state_dict(),
                'val_mse': float(val_mse),
                'val_csi_avg': float(val_csi_avg),
                'val_csi_133': float(val_csi_133)
            }, os.path.join(args.save_dir, "best_model.pth"))
            ema.restore()
            print(f"--> Best EMA Model Saved (Val CSI-AVG: {best_val_csi:.4f}, CSI-133: {val_csi_133:.4f})")

if __name__ == "__main__":
    main()
