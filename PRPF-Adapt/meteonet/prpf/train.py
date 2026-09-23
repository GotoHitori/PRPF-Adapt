import os
import argparse
import re
import warnings
import csv
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

from .dataset import SevirTxtPanguDataset, ArchiveMatchedDataset, build_sevir_pairs_by_date
from .meteonet_dataset import MeteoNetTxtPanguDataset
from .customer_meteonet_dataset import MeteoNetTxtPanguDataset as CustomerMeteoNetDataset
from .customer_meteonet_objectives import (
    CustomerMetricAccumulator,
    LeadTimeWeightedSoftCSILoss,
    MultiThresholdTverskyLoss,
    WeightedSoftCSILoss,
    clear_sky_distillation_loss,
    customer_event_sample_weights,
    outside_threshold_band_distillation_loss,
    passes_rounded_targets,
    strictly_dominates,
)
from .model import PRPF_SetGoGAN_Generator, FrameDiscriminator, SeqDiscriminator3D, copy_shared_weights
from .model_customer_m3 import (
    PRPF_SetGoGAN_Generator as CustomerMeteonetGenerator,
    FrameDiscriminator as CustomerFrameDiscriminator,
    SeqDiscriminator3D as CustomerSeqDiscriminator3D,
    copy_shared_weights as copy_customer_shared_weights,
)
from .meteonet_targets import MeteonetTargets, passes_all_targets, weakest_target_score
from .protocol import balance_weights as protocol_balance_weights, protocol_defaults, validate_protocol_args

THRESHOLDS = [16, 74, 133, 160, 181, 219]
METEONET_THRESHOLDS = [12, 24, 32]
CUSTOMER_METEONET_BASELINE = {
    "mse": 7.686956592947364,
    "csi_12": 0.46927088011883844,
    "csi_24": 0.30835861682013976,
    "csi_32": 0.037241802439858106,
    "csi_avg": 0.2716237664596121,
}

def _base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def _evaluation_model(model):
    return _base_model(model)

def set_adapter_stage(model, adapter_only):
    base = _base_model(model)
    for name, parameter in base.named_parameters():
        parameter.requires_grad = not adapter_only or name.startswith("reliability_router.")

def build_optimizer_groups(model, base_lr, router_multiplier):
    base = _base_model(model)
    shared = []
    router = []
    for name, parameter in base.named_parameters():
        if not parameter.requires_grad:
            continue
        (router if name.startswith("reliability_router.") else shared).append(parameter)
    groups = []
    if shared:
        groups.append({"params": shared, "lr": base_lr})
    if router:
        groups.append({"params": router, "lr": base_lr * router_multiplier})
    return groups

PANGU_U_IDX = 0
PANGU_V_IDX = 6
PANGU_WIND_LEVELS = 6
PANGU_STEER_LEVELS = (0, 1, 2)

class BalancedMSEMAE(nn.Module):
    def __init__(self, thresholds, weights, mean, std, mse_weight=1.0, mae_weight=1.0):
        super().__init__()
        if len(weights) != len(thresholds) + 1:
            raise ValueError("weights must have one more value than thresholds")
        self.thresholds = tuple(float(value) for value in thresholds)
        self.weights = tuple(float(value) for value in weights)
        self.mean = float(mean)
        self.std = float(std)
        self.mse_weight = float(mse_weight)
        self.mae_weight = float(mae_weight)

    def forward(self, prediction, target):
        raw_target = target * self.std + self.mean
        weight = torch.full_like(target, self.weights[0])
        for threshold, value in zip(self.thresholds, self.weights[1:]):
            weight = torch.where(raw_target >= threshold, weight.new_tensor(value), weight)
        error = prediction - target
        return self.mse_weight * (weight * error.square()).mean() + self.mae_weight * (weight * error.abs()).mean()

class SoftCSILoss(nn.Module):
    def __init__(self, threshold, mean, std, temperature=8.0):
        super().__init__()
        self.threshold = float(threshold)
        self.mean = float(mean)
        self.std = float(std)
        self.temperature = float(temperature)

    def forward(self, prediction, target):
        prediction = prediction * self.std + self.mean
        target = target * self.std + self.mean
        pred_prob = torch.sigmoid((prediction - self.threshold) / self.temperature)
        target_prob = torch.sigmoid((target - self.threshold) / self.temperature)
        intersection = (pred_prob * target_prob).sum()
        denominator = pred_prob.sum() + target_prob.sum() - intersection
        return 1.0 - (intersection + 1.0) / (denominator + 1.0)

class MeanSoftCSILoss(nn.Module):
    def __init__(self, thresholds, mean, std, temperature=8.0):
        super().__init__()
        self.losses = nn.ModuleList([
            SoftCSILoss(threshold, mean, std, temperature) for threshold in thresholds
        ])

    def forward(self, prediction, target):
        return torch.stack([loss(prediction, target) for loss in self.losses]).mean()

class MultiScaleMSELoss(nn.Module):
    def __init__(self, scales=(2, 4), weight=1.0):
        super().__init__()
        self.scales = tuple(int(scale) for scale in scales)
        self.weight = float(weight)

    def forward(self, prediction, target):
        value = prediction.new_zeros(())
        for scale in self.scales:
            if scale <= 1:
                pooled_prediction = prediction
                pooled_target = target
            else:
                pooled_prediction = F.avg_pool3d(prediction, (1, scale, scale))
                pooled_target = F.avg_pool3d(target, (1, scale, scale))
            value = value + F.mse_loss(pooled_prediction, pooled_target)
        return self.weight * value / max(len(self.scales), 1)

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

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {name: value.clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state):
        self.decay = float(state["decay"])
        parameters = dict(self.model.named_parameters())
        self.shadow = {
            name: value.to(device=parameters[name].device, dtype=parameters[name].dtype).clone()
            for name, value in state["shadow"].items()
        }
        self.backup = {}

class PhysicsInformedAdvectionDiffusionLoss(nn.Module):
    def __init__(self, phase=1, dt=1.0, kappa=0.01, w_phys=0.1, w_smooth=0.05,
                 pangu_means=None, pangu_stds=None, steer_phi=0.2, v_sign=1.0,
                 u_idx=PANGU_U_IDX, v_idx=PANGU_V_IDX, n_wind_levels=PANGU_WIND_LEVELS,
                 steer_levels=PANGU_STEER_LEVELS,
                 balance_thresholds=THRESHOLDS,
                 balance_weights=(1, 1, 2, 4, 6, 10, 15),
                 device='cuda'):
        super().__init__()
        self.phase = phase
        self.dt = dt
        self.kappa = kappa
        self.w_phys = w_phys
        self.w_smooth = w_smooth

        self.steer_phi = float(steer_phi)
        self.v_sign = float(v_sign)
        self.u_idx = int(u_idx)
        self.v_idx = int(v_idx)
        self.n_wind_levels = int(n_wind_levels)
        self.steer_levels = tuple(int(l) for l in steer_levels)
        self.reconstruction = BalancedMSEMAE(balance_thresholds, balance_weights, 0.0, 1.0)

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
        return self.steer_phi * u, self.steer_phi * self.v_sign * v

    def forward(self, pred_seq, target_seq, res_flow, x_wfm, mean, std):
        y_raw = target_seq * std + mean
        y_raw = torch.clamp(y_raw, 0.0, 255.0)

        self.reconstruction.mean = float(mean)
        self.reconstruction.std = float(std)
        loss_recon = self.reconstruction(pred_seq, target_seq)

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

def train_one_epoch(gen, ema, d_frame, d_seq, g_opt, d_opt, g_scheduler, d_scheduler,
                    phys_crit, loader, device, current_lambda_adv, mean, std,
                    amp=False, max_steps=0, adapter_steps=0, gate_reg_weight=0.0,
                    soft_csi_crit=None, soft_csi_weight=0.0, multiscale_crit=None,
                    multiscale_mse_weight=0.0, raw_mse_weight=0.0,
                    tversky_crit=None, tversky_weight=0.0, teacher_model=None,
                    clear_sky_distill_weight=0.0, clear_sky_threshold=12.0,
                    band_distill_weight=0.0, band_distill_thresholds=(12.0, 24.0, 32.0),
                    band_distill_width=3.0):
    gen.train()
    d_frame.train()
    d_seq.train()
    stats = {'g': 0.0, 'd': 0.0, 'adv_f': 0.0, 'adv_s': 0.0, 'phys': 0.0}
    n = 0

    pbar = tqdm(loader, dynamic_ncols=True, desc="Train", leave=False)

    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    for step, (x_obs, x_wfm, y_seq) in enumerate(pbar):
        if max_steps and step >= max_steps:
            break
        if adapter_steps and step == adapter_steps:
            set_adapter_stage(gen, adapter_only=False)
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
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            fake_radar, res_flow = gen(x_obs, x_wfm)
        with torch.amp.autocast("cuda", enabled=False):
            loss_recon_phys, l_phys_val = phys_crit(fake_radar.float(), y_seq.float(), res_flow.float(), x_wfm.float(), mean, std)

        loss_g = loss_recon_phys
        if raw_mse_weight:
            loss_g = loss_g + raw_mse_weight * F.mse_loss(fake_radar.float(), y_seq.float())
        if soft_csi_crit is not None and soft_csi_weight:
            loss_g = loss_g + soft_csi_weight * soft_csi_crit(fake_radar.float(), y_seq.float())
        if tversky_crit is not None and tversky_weight:
            loss_g = loss_g + tversky_weight * tversky_crit(fake_radar.float(), y_seq.float())
        teacher_radar = None
        if teacher_model is not None and (clear_sky_distill_weight or band_distill_weight):
            with torch.no_grad():
                teacher_radar, _ = teacher_model(x_obs, x_wfm)
        if teacher_radar is not None and clear_sky_distill_weight:
            loss_g = loss_g + clear_sky_distill_weight * clear_sky_distillation_loss(
                fake_radar.float(), teacher_radar.float(), y_seq.float(),
                clear_sky_threshold, mean, std,
            )
        if teacher_radar is not None and band_distill_weight:
            loss_g = loss_g + band_distill_weight * outside_threshold_band_distillation_loss(
                fake_radar.float(), teacher_radar.float(), y_seq.float(),
                band_distill_thresholds, band_distill_width, mean, std,
            )
        if multiscale_crit is not None and multiscale_mse_weight:
            loss_g = loss_g + multiscale_crit(fake_radar.float(), y_seq.float())
        if gate_reg_weight:
            loss_g = loss_g + gate_reg_weight * getattr(_base_model(gen), "router_regularizer", loss_g.new_zeros(()))
        loss_adv_f_val = 0.0
        loss_adv_s_val = 0.0

        if current_lambda_adv[0] > 0:
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits_f = d_frame(fake_radar.view(-1, 1, H, W))
                loss_adv_f = hinge_g_loss(logits_f)
            loss_g += current_lambda_adv[0] * loss_adv_f
            loss_adv_f_val = loss_adv_f.item()

        if current_lambda_adv[1] > 0:
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits_s = d_seq(fake_radar)
                loss_adv_s = hinge_g_loss(logits_s)
            loss_g += current_lambda_adv[1] * loss_adv_s
            loss_adv_s_val = loss_adv_s.item()

        scaler.scale(loss_g).backward()
        scaler.unscale_(g_opt)
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        scaler.step(g_opt)
        scaler.update()
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
def evaluate(gen, loader, device, mean, std, max_steps=0, focus_threshold=133,
             thresholds=None, metric_protocol="global"):
    gen = _evaluation_model(gen)
    gen.eval()
    thresholds = tuple(THRESHOLDS if thresholds is None else thresholds)
    mse_sum = 0.0
    n_items = 0

    csi_counts = {thr: [0, 0, 0] for thr in thresholds}
    customer_metrics = (
        CustomerMetricAccumulator(thresholds, lead_times=20)
        if metric_protocol == "customer" else None
    )

    pbar = tqdm(loader, dynamic_ncols=True, desc="Val", leave=False)

    for step, (x_obs, x_wfm, y_seq) in enumerate(pbar):
        if max_steps and step >= max_steps:
            break
        x_obs = x_obs.to(device).float()
        x_wfm = x_wfm.to(device).float()
        y_seq = y_seq.to(device).float()

        pred, _ = gen(x_obs, x_wfm)

        pred_raw = torch.clamp(pred * std + mean, 0.0, 255.0)
        y_raw = torch.clamp(y_seq * std + mean, 0.0, 255.0)

        diff = pred_raw - y_raw
        mse_sum += (diff ** 2).sum().item()
        n_items += diff.numel()
        if customer_metrics is not None:
            customer_metrics.update(pred_raw, y_raw)

        for thr in thresholds:
            p_b = pred_raw >= thr
            g_b = y_raw >= thr

            h = (p_b & g_b).sum().item()
            m = (~p_b & g_b).sum().item()
            f = (p_b & ~g_b).sum().item()

            csi_counts[thr][0] += h
            csi_counts[thr][1] += m
            csi_counts[thr][2] += f

    if customer_metrics is not None:
        result = customer_metrics.compute()
        csi_by_threshold = result["csi_by_threshold"]
        csi_focus = csi_by_threshold.get(float(focus_threshold), 0.0)
        return result["mse"], result["csi_avg"], csi_focus, csi_by_threshold

    mse_avg = mse_sum / max(1, n_items)

    csi_vals = []
    csi_focus = 0.0
    csi_by_threshold = {}
    for thr in thresholds:
        h, m, f = csi_counts[thr]
        denom = h + m + f
        csi = 0.0 if denom == 0 else float(h) / float(denom)
        csi_vals.append(csi)
        csi_by_threshold[thr] = csi
        if thr == focus_threshold:
            csi_focus = csi

    csi_avg = sum(csi_vals) / len(csi_vals)

    return mse_avg, csi_avg, csi_focus, csi_by_threshold

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

def load_m0_initialization_strict(model, state_dict, model_variant):
    incompatible = model.load_state_dict(_extract_state_for_load(state_dict), strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    expected_missing = (
        {name for name in model.state_dict() if name.startswith("reliability_router.")}
        if model_variant == "m3" else set()
    )
                                                                             
                                                                              
    legacy_band_bias = {"reliability_router.threshold_bias_dbz"}
    allowed_missing = (
        expected_missing, set(), legacy_band_bias
    ) if model_variant == "m3" else (set(),)
    if missing not in allowed_missing or unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"expected_missing={sorted(expected_missing)}"
        )

def build_checkpoint_state(epoch, gen, d_frame, d_seq, g_opt, d_opt, val_mse, val_csi_avg,
                           val_csi_133, args, ema=None, g_scheduler=None, d_scheduler=None,
                           loader_generator=None, val_csi_by_threshold=None,
                           selection_score=None, resolved_targets=None,
                           passes_targets=None):
    state = {
        "epoch": epoch,
        "gen": _extract_dp_state(gen),
        "val_mse": float(val_mse),
        "val_csi_avg": float(val_csi_avg),
        "val_csi_133": float(val_csi_133),
        "args": dict(args),
    }
    if val_csi_by_threshold is not None:
        state["val_csi_by_threshold"] = {
            str(threshold): float(value)
            for threshold, value in val_csi_by_threshold.items()
        }
    if selection_score is not None:
        state["selection_score"] = float(selection_score)
    if resolved_targets is not None:
        state["resolved_targets"] = {
            name: float(value) for name, value in resolved_targets.items()
        }
    if passes_targets is not None:
        state["passes_all_targets"] = bool(passes_targets)
    if d_frame is not None:
        state["d_frame"] = _extract_dp_state(d_frame)
    if d_seq is not None:
        state["d_seq"] = _extract_dp_state(d_seq)
    if g_opt is not None:
        state["g_opt"] = g_opt.state_dict()
    if d_opt is not None:
        state["d_opt"] = d_opt.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    if g_scheduler is not None:
        state["g_scheduler"] = g_scheduler.state_dict()
    if d_scheduler is not None:
        state["d_scheduler"] = d_scheduler.state_dict()
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state

def restore_training_state(state, gen, d_frame, d_seq, g_opt, d_opt, g_scheduler, d_scheduler, ema, loader_generator=None):
    _evaluation_model(gen).load_state_dict(_extract_state_for_load(state["gen"]), strict=False)
    if d_frame is not None and "d_frame" in state:
        _evaluation_model(d_frame).load_state_dict(_extract_state_for_load(state["d_frame"]), strict=False)
    if d_seq is not None and "d_seq" in state:
        _evaluation_model(d_seq).load_state_dict(_extract_state_for_load(state["d_seq"]), strict=False)
    if g_opt is not None and "g_opt" in state:
        g_opt.load_state_dict(state["g_opt"])
    if d_opt is not None and "d_opt" in state:
        d_opt.load_state_dict(state["d_opt"])
    if g_scheduler is not None and "g_scheduler" in state:
        g_scheduler.load_state_dict(state["g_scheduler"])
    if d_scheduler is not None and "d_scheduler" in state:
        d_scheduler.load_state_dict(state["d_scheduler"])
    if ema is not None:
        if "ema" in state:
            ema.load_state_dict(state["ema"])
        else:
            ema.shadow = {
                name: param.data.clone()
                for name, param in ema.model.named_parameters()
                if param.requires_grad
            }
    if loader_generator is not None and "loader_generator" in state:
        loader_generator.set_state(state["loader_generator"])
    return int(state["epoch"]) + 1, float(
        state.get("selection_score", state.get("val_csi_avg", -1.0))
    )

def normalize_physical_threshold(value, mean, std):
    if std <= 0:
        raise ValueError("radar std must be positive")
    return (float(value) - float(mean)) / float(std)


def resolve_soft_csi_thresholds(dataset, model_variant, values):
    if dataset != "meteonet" or model_variant != "m3":
        return ()
    thresholds = tuple(
        float(value.strip()) for value in values.split(',') if value.strip()
    )
    if not thresholds:
        raise ValueError("meteonet_soft_csi_thresholds must contain at least one threshold")
    return thresholds


def meteonet_target_score(metrics, targets):
    if isinstance(targets, dict):
        targets = MeteonetTargets(**targets)
    return weakest_target_score(metrics, targets)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="1: Phase 1 Training, 2: Phase 2 Fine-tuning")
    parser.add_argument("--dataset", choices=["sevir", "meteonet"], default="sevir")
    parser.add_argument("--sevir_root", type=str, required=True)
    parser.add_argument("--pangu_root", type=str, required=True)
    parser.add_argument("--train_periods", type=str, required=True)
    parser.add_argument("--train_pangu", type=str, required=True)
    parser.add_argument("--test_periods", type=str, required=True)
    parser.add_argument("--test_pangu", type=str, required=True)
    parser.add_argument("--meteonet_train_ids", type=str, default=None)
    parser.add_argument("--meteonet_test_ids", type=str, default=None)
    parser.add_argument("--meteonet_radar_root", type=str, default=None)
    parser.add_argument("--meteonet_pangu_stats", type=str, default=None)
    parser.add_argument("--meteonet_val_fraction", type=float, default=0.1,
                        help="Chronological validation fraction carved from MeteoNet training IDs")
    parser.add_argument('--save_dir', type=str, default='checkpoints_prpf')
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to pre-trained model for Phase 2")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=0,
                        help="Validation batch size; 0 reuses --batch_size")
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_adv_f', type=float, default=0.01)
    parser.add_argument('--lambda_adv_s', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--model_variant', choices=['m0', 'm3'], default='m0')
    parser.add_argument('--model_architecture', choices=['current', 'customer_meteonet'], default='current')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--train_fraction', type=float, default=1.0)
    parser.add_argument('--max_train_steps', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint_every', type=int, default=1)
    parser.add_argument('--archive_layout', action='store_true')
    parser.add_argument('--max_val_steps', type=int, default=0)
    parser.add_argument('--adapter_steps', type=int, default=0,
                        help="Legacy within-epoch adapter warmup; do not combine with --adapter_epochs")
    parser.add_argument('--adapter_epochs', type=int, default=0,
                        help="Freeze all shared generator parameters for this many complete epochs")
    parser.add_argument('--router_lr_multiplier', type=float, default=5.0)
    parser.add_argument('--router_echo_threshold_dbz', type=float, default=12.0,
                        help="Physical MeteoNet reflectivity threshold used to activate the M3 router")
    parser.add_argument('--router_max_residual_dbz', type=float, default=0.5,
                        help="Maximum absolute customer M3 output correction in physical dBZ")
    parser.add_argument('--gate_reg_weight', type=float, default=1e-4)
    parser.add_argument('--soft_csi_weight', type=float, default=0.0)
    parser.add_argument('--soft_csi_temperature', type=float, default=8.0)
    parser.add_argument('--meteonet_soft_csi_thresholds', type=str, default='24,32',
                        help="Comma-separated MeteoNet thresholds used by Soft-CSI")
    parser.add_argument('--soft_csi_threshold_weights', type=str, default='',
                        help="Comma-separated per-threshold Soft-CSI weights")
    parser.add_argument('--lead_time_soft_csi', action='store_true')
    parser.add_argument('--tversky_weight', type=float, default=0.0)
    parser.add_argument('--tversky_false_negative_weight', type=float, default=0.7)
    parser.add_argument('--clear_sky_distill_weight', type=float, default=0.0)
    parser.add_argument('--band_distill_weight', type=float, default=0.0)
    parser.add_argument('--band_distill_width', type=float, default=3.0)
    parser.add_argument('--event_balanced_sampling', action='store_true')
    parser.add_argument('--meteonet_metric_protocol', choices=['global', 'customer'], default='global')
    parser.add_argument('--target_tolerance', type=float, default=0.0)
    parser.add_argument('--strict_improvement', action='store_true')
    parser.add_argument('--multiscale_mse_weight', type=float, default=0.0)
    parser.add_argument('--raw_mse_weight', type=float, default=0.0,
                        help="Additional unweighted normalized-pixel MSE term for MSE-focused fine-tuning")
    parser.add_argument('--selection_metric', choices=['csi_avg', 'target_score'], default='csi_avg',
                        help="Checkpoint selection metric; target_score maximizes the weakest target ratio")
    parser.add_argument('--target_mse', type=float, default=7.6870)
    parser.add_argument('--target_csi_12', type=float, default=0.4693)
    parser.add_argument('--target_csi_24', type=float, default=0.3084)
    parser.add_argument('--target_csi_32', type=float, default=0.0372)
    parser.add_argument('--target_csi_avg', type=float, default=0.2716)
    parser.add_argument('--target_csi_focus', type=float, default=0.05460)
    parser.add_argument('--paper_protocol', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stage_name', choices=['stage1', 'stage2'], default='stage1')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--w_phys', type=float, default=0.1)
    parser.add_argument('--w_smooth', type=float, default=0.05)
    parser.add_argument('--steer_phi', type=float, default=0.2,
                        help="Maps Pangu winds (m/s) to pixels/frame for U_steer. Theoretical SEVIR "
                             "value = dt/pixel = 600s/3000m = 0.2 (10-min cadence, 384km->128px). "
                             "Recalibrate per dataset; see calibrate_steer_phi.py.")
    parser.add_argument('--pangu_v_sign', type=float, default=1.0)

    parser.add_argument('--att_dim', type=int, default=128,
                        help="Internal feature width C (fusion width = 2*att_dim). 128 -> ~7.7M generator params.")
    parser.add_argument('--swin_depth', type=int, default=4, help="Number of Swin blocks in spatial refinement.")
    parser.add_argument('--no_ltam', action='store_true', help="Ablate LTAM radar-history modulation.")
    parser.add_argument('--no_csmmsa', action='store_true', help="Ablate the CS-MMSA local cross-scale modulation branch.")
    parser.add_argument('--no_crossattn', action='store_true', help="Ablate the global cross-attention alignment branch.")
    parser.add_argument('--no_channel_fusion', action='store_true', help="Ablate bidirectional channel fusion + CSA.")
    parser.add_argument('--no_swin', action='store_true', help="Ablate Swin spatial refinement.")
    parser.add_argument('--no_temporal', action='store_true', help="Ablate temporal refinement (3D conv + temporal attention).")

    return parser

def main():
    args = build_parser().parse_args()

    if args.adapter_steps and args.adapter_epochs:
        raise ValueError("--adapter_steps and --adapter_epochs are mutually exclusive")
    if args.adapter_epochs and args.model_variant != "m3":
        raise ValueError("--adapter_epochs requires --model_variant m3")

    if args.paper_protocol:
        validate_protocol_args(args)
        protocol = protocol_defaults()
        if args.smoke:
            args.batch_size = min(args.batch_size, protocol.batch_size)
        args.lr_g = protocol.lr_g if args.lr_g == 2e-4 else args.lr_g
        args.lr_d = protocol.lr_d if args.lr_d == 2e-4 else args.lr_d
        args.lambda_adv_f = protocol.lambda_adv_f if args.lambda_adv_f == 0.01 else args.lambda_adv_f
        args.lambda_adv_s = protocol.lambda_adv_s if args.lambda_adv_s == 0.01 else args.lambda_adv_s
        args.w_phys = protocol.w_phys if args.w_phys == 0.1 else args.w_phys
        args.w_smooth = protocol.w_smooth if args.w_smooth == 0.05 else args.w_smooth

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dataset == "meteonet":
        required = {
            "meteonet_train_ids": args.meteonet_train_ids,
            "meteonet_test_ids": args.meteonet_test_ids,
            "meteonet_radar_root": args.meteonet_radar_root,
            "pangu_root": args.pangu_root,
            "meteonet_pangu_stats": args.meteonet_pangu_stats,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"MeteoNet mode requires: {', '.join(missing)}")
        with open(args.meteonet_train_ids) as stream:
            train_ids = [line.strip() for line in stream if line.strip()]
        with open(args.meteonet_test_ids) as stream:
            test_ids = [line.strip() for line in stream if line.strip()]
        if not 0.0 < args.meteonet_val_fraction < 0.5:
            raise ValueError("meteonet_val_fraction must be in (0, 0.5)")
        if args.model_architecture == "customer_meteonet":
            dataset_class = CustomerMeteoNetDataset
            val_ids = test_ids
        else:
            dataset_class = MeteoNetTxtPanguDataset
            val_count = max(1, int(len(train_ids) * args.meteonet_val_fraction))
            val_ids = train_ids[-val_count:]
            train_ids = train_ids[:-val_count]
            overlap = set(train_ids + val_ids).intersection(test_ids)
            if overlap:
                raise ValueError(f"MeteoNet train/validation IDs overlap test IDs: {len(overlap)}")
        train_ds = dataset_class(
            train_ids, args.meteonet_radar_root, args.pangu_root,
            Tin=5, Tout=20, pangu_stats_path=args.meteonet_pangu_stats,
        )
        val_ds = dataset_class(
            val_ids, args.meteonet_radar_root, args.pangu_root,
            Tin=5, Tout=20, pangu_stats_path=args.meteonet_pangu_stats,
        )
        train_pairs, test_pairs = [], []
    elif args.archive_layout:
        train_pairs, test_pairs = [], []
    else:
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

    if not 0 < args.train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    if train_pairs:
        subset_size = max(1, int(len(train_pairs) * args.train_fraction))
        train_pairs = train_pairs[:subset_size]

    real_train_pairs = train_pairs
    val_pairs = test_pairs

    if args.dataset == "meteonet":
        pass
    elif args.archive_layout:
        train_ds = ArchiveMatchedDataset(args.sevir_root, args.pangu_root, "train")
        val_ds = ArchiveMatchedDataset(args.sevir_root, args.pangu_root, "val")
        subset_size = max(1, int(len(train_ds) * args.train_fraction))
        train_ds = torch.utils.data.Subset(train_ds, range(subset_size))
    else:
        train_ds = SevirTxtPanguDataset(real_train_pairs, args.sevir_root, args.pangu_root, Tin=5, Tout=20)
        val_ds = SevirTxtPanguDataset(val_pairs, args.sevir_root, args.pangu_root, Tin=5, Tout=20)

    print(f"--- Running Phase {args.phase} ---")
    print(f"Total training samples: {len(train_ds)}")
    print(f"Total validation samples: {len(val_ds)}")

    stats_ds = train_ds.dataset if isinstance(train_ds, torch.utils.data.Subset) else train_ds
    eval_thresholds = METEONET_THRESHOLDS if args.dataset == "meteonet" else THRESHOLDS
    r_mean = getattr(stats_ds, "radar_mean", 0.0)
    r_std = getattr(stats_ds, "radar_std", 1.0)
    p_means = getattr(stats_ds, "pangu_channel_means", None)
    p_stds = getattr(stats_ds, "pangu_channel_stds", None)
    if p_means is None or p_stds is None:
        warnings.warn("Pangu channel stats unavailable; U_steer will be computed from normalized winds "
                      "(physically meaningful steering requires pangu_channel_stats.npz).")

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_sampler = None
    if args.event_balanced_sampling:
        if args.dataset != "meteonet" or args.model_architecture != "customer_meteonet":
            raise ValueError("--event_balanced_sampling requires customer_meteonet")
        sample_weights = customer_event_sample_weights(train_ds)
        train_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True,
            generator=loader_generator,
        )
        print("[sampling] customer event-balanced weights: "
              f"min={sample_weights.min().item():.1f} max={sample_weights.max().item():.1f}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=train_sampler is None, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              generator=loader_generator)
    val_batch_size = args.val_batch_size or args.batch_size
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    router_threshold = (
        normalize_physical_threshold(args.router_echo_threshold_dbz, r_mean, r_std)
        if args.dataset == "meteonet" else 1.41
    )
    if args.model_architecture == "customer_meteonet":
        if args.dataset != "meteonet":
            raise ValueError("customer_meteonet architecture requires --dataset meteonet")
        generator_class = CustomerMeteonetGenerator
        frame_discriminator_class = CustomerFrameDiscriminator
        seq_discriminator_class = CustomerSeqDiscriminator3D
        shared_weight_copier = copy_customer_shared_weights
        model_kwargs = dict(
            att_dim=args.att_dim,
            router_echo_threshold=router_threshold,
            router_max_residual_dbz=args.router_max_residual_dbz,
        )
    else:
        generator_class = PRPF_SetGoGAN_Generator
        frame_discriminator_class = FrameDiscriminator
        seq_discriminator_class = SeqDiscriminator3D
        shared_weight_copier = copy_shared_weights
        model_kwargs = dict(
            att_dim=args.att_dim,
            swin_depth=args.swin_depth,
            use_ltam=not args.no_ltam,
            use_csmmsa=not args.no_csmmsa,
            use_crossattn=not args.no_crossattn,
            use_channel_fusion=not args.no_channel_fusion,
            use_swin=not args.no_swin,
            use_temporal=not args.no_temporal,
            router_echo_threshold=router_threshold,
        )
    if args.model_variant == "m3":
        shared_source = generator_class(model_variant="m0", **model_kwargs)
        gen = generator_class(model_variant="m3", **model_kwargs)
        shared_weight_copier(shared_source, gen)
        del shared_source
    else:
        gen = generator_class(model_variant="m0", **model_kwargs)
    gen = gen.to(device)
    n_gen_params = sum(p.numel() for p in gen.parameters())
    print(f"[model] generator parameters: {n_gen_params/1e6:.2f}M "
          f"(att_dim={args.att_dim}, swin_depth={args.swin_depth}, "
          f"ltam={not args.no_ltam}, csmmsa={not args.no_csmmsa}, crossattn={not args.no_crossattn}, "
          f"channel_fusion={not args.no_channel_fusion}, swin={not args.no_swin}, temporal={not args.no_temporal})")
    torch.manual_seed(args.seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 1)
    d_frame = frame_discriminator_class().to(device)
    d_seq = seq_discriminator_class().to(device)

    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        print(f"Loading pretrained weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        load_m0_initialization_strict(gen, ckpt['gen'], args.model_variant)
        if 'd_frame' in ckpt:
            d_frame.load_state_dict(_extract_state_for_load(ckpt['d_frame']), strict=True)
        if 'd_seq' in ckpt:
            d_seq.load_state_dict(_extract_state_for_load(ckpt['d_seq']), strict=True)
        print("Pretrained weights successfully loaded!")

    teacher_model = None
    if args.clear_sky_distill_weight or args.band_distill_weight:
        if args.model_architecture != "customer_meteonet" or args.model_variant != "m3" or not args.checkpoint:
            raise ValueError("distillation requires customer_meteonet M3 with --checkpoint")
        teacher_model = generator_class(model_variant="m0", **model_kwargs).to(device)
        teacher_model.load_state_dict(_extract_state_for_load(ckpt['gen']), strict=True)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False

    ema = EMA(gen, decay=args.ema_decay)

    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen)
        d_frame = nn.DataParallel(d_frame)
        d_seq = nn.DataParallel(d_seq)


    if args.dataset == "meteonet":
        balance_thresholds = METEONET_THRESHOLDS
        balance_weights = (1, 2, 4, 8)
    else:
        balance_thresholds = THRESHOLDS
        balance_weights = protocol_balance_weights(args.phase) if args.paper_protocol else (1, 1, 2, 4, 6, 10, 15)
    phys_crit = PhysicsInformedAdvectionDiffusionLoss(
        phase=args.phase, device=device, w_phys=args.w_phys, w_smooth=args.w_smooth,
        pangu_means=p_means, pangu_stds=p_stds, steer_phi=args.steer_phi,
        v_sign=(-1.0 if args.dataset == "meteonet" else args.pangu_v_sign),
        balance_thresholds=balance_thresholds, balance_weights=balance_weights
    )
    soft_csi_thresholds = resolve_soft_csi_thresholds(
        args.dataset, args.model_variant, args.meteonet_soft_csi_thresholds,
    )
    if args.dataset == "meteonet" and args.model_variant == "m3":
        if args.soft_csi_threshold_weights:
            soft_csi_weights = tuple(
                float(value.strip())
                for value in args.soft_csi_threshold_weights.split(',')
                if value.strip()
            )
            soft_csi_class = (
                LeadTimeWeightedSoftCSILoss if args.lead_time_soft_csi
                else WeightedSoftCSILoss
            )
            soft_csi_crit = soft_csi_class(
                soft_csi_thresholds, soft_csi_weights,
                r_mean, r_std, args.soft_csi_temperature,
            )
        else:
            soft_csi_crit = MeanSoftCSILoss(
                soft_csi_thresholds, r_mean, r_std, args.soft_csi_temperature
            )
        tversky_crit = MultiThresholdTverskyLoss(
            soft_csi_thresholds, r_mean, r_std, args.soft_csi_temperature,
            args.tversky_false_negative_weight,
        ) if args.tversky_weight else None
    else:
        soft_csi_crit = SoftCSILoss(133.0, r_mean, r_std, args.soft_csi_temperature) if args.model_variant == "m3" else None
        tversky_crit = None
    multiscale_crit = MultiScaleMSELoss((2, 4), args.multiscale_mse_weight) if args.model_variant == "m3" else None
    betas = (args.beta1, args.beta2)
    set_adapter_stage(gen, adapter_only=False)
    g_groups = build_optimizer_groups(gen, args.lr_g, args.router_lr_multiplier)
    g_opt = optim.AdamW(g_groups, lr=args.lr_g, betas=betas, weight_decay=1e-4)
    if args.model_variant == "m3" and (args.adapter_steps > 0 or args.adapter_epochs > 0):
        set_adapter_stage(gen, adapter_only=True)
    d_opt = optim.AdamW(list(d_frame.parameters()) + list(d_seq.parameters()), lr=args.lr_d, betas=betas, weight_decay=1e-4)

    steps_per_epoch = min(len(train_loader), args.max_train_steps) if args.max_train_steps else len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    if args.phase == 1:
        max_lrs = [group["lr"] for group in g_opt.param_groups]
        g_scheduler = optim.lr_scheduler.OneCycleLR(g_opt, max_lr=max_lrs, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
        d_scheduler = optim.lr_scheduler.OneCycleLR(d_opt, max_lr=args.lr_d, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
    else:
        g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=total_steps, eta_min=1e-6)
        d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=total_steps, eta_min=1e-6)


    best_selection_score = -1.0
    start_epoch = 1
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu")
        start_epoch, best_selection_score = restore_training_state(
            resume, gen, d_frame, d_seq, g_opt, d_opt,
            g_scheduler, d_scheduler, ema, loader_generator,
        )

    csv_path = os.path.join(args.save_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as file:
            csi_names = [f"csi_{threshold}" for threshold in eval_thresholds]
            csv.writer(file).writerow(
                ["epoch", "g", "d", "phys", "val_mse", "csi_avg", *csi_names,
                 "selection_score", "passes_all_targets"]
            )

    for epoch in range(start_epoch, args.epochs + 1):

        if args.model_variant == "m3" and args.adapter_epochs:
            adapter_only = epoch <= args.adapter_epochs
            set_adapter_stage(gen, adapter_only=adapter_only)
            print(f"[stage] epoch={epoch} adapter_only={adapter_only}")

        cur_lambda_f = get_adv_weight(epoch, args.epochs, args.lambda_adv_f, args.phase)
        cur_lambda_s = get_adv_weight(epoch, args.epochs, args.lambda_adv_s, args.phase)

        stats = train_one_epoch(
            gen, ema, d_frame, d_seq, g_opt, d_opt, g_scheduler, d_scheduler,
            phys_crit, train_loader, device, (cur_lambda_f, cur_lambda_s),
            r_mean, r_std, args.amp, args.max_train_steps, args.adapter_steps,
            args.gate_reg_weight, soft_csi_crit, args.soft_csi_weight,
            multiscale_crit, args.multiscale_mse_weight, args.raw_mse_weight,
            tversky_crit, args.tversky_weight, teacher_model,
            args.clear_sky_distill_weight, 12.0,
            args.band_distill_weight, soft_csi_thresholds, args.band_distill_width,
        )

        ema.apply_shadow()
        focus_threshold = 32 if args.dataset == "meteonet" else 133
        val_mse, val_csi_avg, val_csi_133, val_csi_by_threshold = evaluate(
            gen, val_loader, device, r_mean, r_std, args.max_val_steps,
            focus_threshold, eval_thresholds, args.meteonet_metric_protocol
        )
        ema.restore()

        resolved_targets = None
        passes_targets = False
        if args.dataset == "meteonet":
            resolved_targets = (
                dict(CUSTOMER_METEONET_BASELINE) if args.strict_improvement else {
                    "mse": args.target_mse,
                    "csi_12": args.target_csi_12,
                    "csi_24": args.target_csi_24,
                    "csi_32": args.target_csi_32,
                    "csi_avg": args.target_csi_avg,
                }
            )
            meteonet_metrics = {
                "mse": val_mse,
                "csi_12": val_csi_by_threshold[12],
                "csi_24": val_csi_by_threshold[24],
                "csi_32": val_csi_by_threshold[32],
                "csi_avg": val_csi_avg,
            }
            passes_targets = (
                strictly_dominates(meteonet_metrics, resolved_targets)
                if args.strict_improvement else passes_rounded_targets(
                    meteonet_metrics, resolved_targets, args.target_tolerance
                )
            )

        if args.selection_metric == "target_score":
            if args.dataset == "meteonet":
                selection_score = meteonet_target_score(
                    meteonet_metrics, resolved_targets,
                )
            else:
                selection_score = min(
                    args.target_mse / max(val_mse, 1e-12),
                    val_csi_avg / max(args.target_csi_avg, 1e-12),
                    val_csi_133 / max(args.target_csi_focus, 1e-12),
                )
        else:
            selection_score = val_csi_avg

        focus_name = "CSI-32" if args.dataset == "meteonet" else "CSI-133"
        print(f"Epoch {epoch:03d}/{args.epochs} | G={stats['g']:.4f} D={stats['d']:.4f} (AdvW: {cur_lambda_f:.4f}) | Val MSE={val_mse:.4f} CSI-AVG={val_csi_avg:.4f} {focus_name}={val_csi_133:.4f} Select={selection_score:.4f}")
        with open(csv_path, "a", newline="") as file:
            csv.writer(file).writerow([
                epoch, stats["g"], stats["d"], stats["phys"], val_mse, val_csi_avg,
                *(val_csi_by_threshold[threshold] for threshold in eval_thresholds),
                selection_score, passes_targets,
            ])

        state = build_checkpoint_state(
            epoch, gen, d_frame, d_seq, g_opt, d_opt,
            val_mse, val_csi_avg, val_csi_133, vars(args),
            ema, g_scheduler, d_scheduler, loader_generator,
            val_csi_by_threshold, selection_score, resolved_targets, passes_targets,
        )
        if epoch % args.checkpoint_every == 0:
            torch.save(state, os.path.join(args.save_dir, f"checkpoint_{epoch:03d}.pth"))

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            ema.apply_shadow()
            best_state = build_checkpoint_state(
                epoch, gen, d_frame, d_seq, g_opt, d_opt,
                val_mse, val_csi_avg, val_csi_133, vars(args),
                ema, g_scheduler, d_scheduler, loader_generator,
                val_csi_by_threshold, selection_score, resolved_targets, passes_targets,
            )
            torch.save(best_state, os.path.join(args.save_dir, "best_model.pth"))
            ema.restore()
            print(f"--> Best EMA Model Saved ({args.selection_metric}: {best_selection_score:.4f}, {focus_name}: {val_csi_133:.4f})")

if __name__ == "__main__":
    main()
