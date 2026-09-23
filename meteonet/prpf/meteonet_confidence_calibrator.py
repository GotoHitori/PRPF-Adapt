import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_threshold_csi_loss(
    prediction,
    target,
    thresholds=(12.0, 24.0, 32.0),
    radar_scale=70.0,
    temperature=1.5,
    threshold_weights=None,
):
    """Lead-averaged differentiable CSI surrogate for the three dBZ bands."""
    prediction_raw = prediction * float(radar_scale)
    target_raw = target * float(radar_scale)
    losses = []
    for threshold in thresholds:
        pred_prob = torch.sigmoid((prediction_raw - float(threshold)) / float(temperature))
        observed = (target_raw >= float(threshold)).to(prediction.dtype)
        lead_losses = []
        for lead in range(prediction.shape[1]):
            p = pred_prob[:, lead]
            y = observed[:, lead]
            intersection = (p * y).sum()
            union = p.sum() + y.sum() - intersection
            lead_losses.append(1.0 - (intersection + 1.0) / (union + 1.0))
        losses.append(torch.stack(lead_losses).mean())
    values = torch.stack(losses)
    if threshold_weights is None:
        return values.mean()
    weights = values.new_tensor(tuple(float(v) for v in threshold_weights))
    if weights.numel() != values.numel() or torch.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("threshold_weights must be three nonnegative values with positive sum")
    return (values * weights).sum() / weights.sum()


def outside_band_trust_loss(
    prediction,
    teacher,
    target,
    thresholds=(12.0, 24.0, 32.0),
    radar_scale=70.0,
    band_width_dbz=3.0,
):
    """Keep the candidate close to the teacher away from threshold bands."""
    target_raw = target * float(radar_scale)
    threshold_tensor = target_raw.new_tensor(tuple(float(v) for v in thresholds))
    distance = (target_raw.unsqueeze(-1) - threshold_tensor).abs().amin(dim=-1)
    outside = distance > float(band_width_dbz)
    if not outside.any():
        return prediction.new_zeros(())
    return F.mse_loss(prediction[outside], teacher[outside])


def threshold_flip_loss(
    prediction,
    base,
    target,
    thresholds=(12.0, 24.0, 32.0),
    radar_scale=70.0,
    margin_dbz=0.25,
    band_width_dbz=5.0,
    threshold_weights=None,
):
    """Hinge loss for correcting only false negatives and false positives."""
    prediction_raw = prediction * float(radar_scale)
    base_raw = base * float(radar_scale)
    target_raw = target * float(radar_scale)
    losses = []
    for threshold in thresholds:
        base_event = base_raw >= float(threshold)
        target_event = target_raw >= float(threshold)
        in_band = (base_raw - float(threshold)).abs() <= float(band_width_dbz)
        false_negative = target_event & ~base_event & in_band
        false_positive = ~target_event & base_event & in_band
        push_up = F.relu(float(threshold) + float(margin_dbz) - prediction_raw)
        push_down = F.relu(prediction_raw - (float(threshold) - float(margin_dbz)))
        errors = false_negative.to(prediction.dtype) * push_up + false_positive.to(prediction.dtype) * push_down
        active = false_negative.logical_or(false_positive)
        losses.append(errors.sum() / active.sum().clamp_min(1).to(errors.dtype))
    values = torch.stack(losses)
    if threshold_weights is None:
        return values.mean()
    weights = values.new_tensor(tuple(float(v) for v in threshold_weights))
    if weights.numel() != values.numel() or torch.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("threshold_weights must be three nonnegative values with positive sum")
    return (values * weights).sum() / weights.sum()


class ConfidenceConditionedResidualCalibrator(nn.Module):
    """External, zero-initialized three-threshold residual correction for M3."""

    def __init__(
        self,
        pangu_channels=34,
        lead_times=20,
        hidden=16,
        thresholds=(12.0, 24.0, 32.0),
        radar_scale=70.0,
        max_residual_dbz=0.5,
        band_width_dbz=3.0,
    ):
        super().__init__()
        if len(thresholds) != 3:
            raise ValueError("three thresholds are required")
        self.lead_times = int(lead_times)
        self.radar_scale = float(radar_scale)
        self.max_residual = float(max_residual_dbz) / self.radar_scale
        self.band_width_dbz = float(band_width_dbz)
        self.register_buffer("thresholds_dbz", torch.tensor(thresholds, dtype=torch.float32))
        self.pangu_mapper = nn.Conv2d(pangu_channels, 8, 1)
                                                                            
        self.context = nn.Sequential(
            nn.Conv2d(18, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.residual_head = nn.Conv2d(hidden, 3, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.regularizer = torch.tensor(0.0)

    def forward(self, base, x_obs, pangu_seq, confidence):
        if base.ndim != 5 or base.shape[2] != 1:
            raise ValueError("base must have shape (B,T,1,H,W)")
        batch, leads, _, height, width = base.shape
        if leads != self.lead_times:
            raise ValueError(f"expected {self.lead_times} lead times, got {leads}")
        if confidence.shape != (batch, leads, 3, height, width):
            raise ValueError("confidence must have shape (B,T,3,H,W)")
        if x_obs.shape[1] < 2:
            raise ValueError("x_obs must contain at least two frames")
        if pangu_seq.shape[1] != leads:
            pangu_seq = F.interpolate(
                pangu_seq.permute(0, 2, 1, 3, 4),
                size=(leads, pangu_seq.shape[-2], pangu_seq.shape[-1]),
                mode="trilinear",
                align_corners=False,
            ).permute(0, 2, 1, 3, 4)
        pangu = pangu_seq.reshape(batch * leads, pangu_seq.shape[2], *pangu_seq.shape[-2:])
        pangu = F.interpolate(pangu, size=(height, width), mode="bilinear", align_corners=False)
        pangu = self.pangu_mapper(pangu).view(batch, leads, 8, height, width)
        last = x_obs[:, -1:].expand(-1, leads, -1, -1, -1)
        change = (x_obs[:, -1:] - x_obs[:, -2:-1]).expand(-1, leads, -1, -1, -1)
        lead = torch.linspace(0.0, 1.0, leads, device=base.device, dtype=base.dtype)
        lead = lead.view(1, leads, 1, 1, 1).expand(batch, -1, -1, height, width)
        probability = confidence.clamp(0.0, 1.0).to(dtype=base.dtype)
        uncertainty = 1.0 - (2.0 * probability - 1.0).abs()
        features = torch.cat((base, last, change, pangu, lead, probability, uncertainty), dim=2)
        raw = self.residual_head(self.context(features.flatten(0, 1)))
        raw = raw.view(batch, leads, 3, height, width)

        distance = (base * self.radar_scale - self.thresholds_dbz.view(1, 1, 3, 1, 1)).abs()
        band_gate = torch.sigmoid((self.band_width_dbz - distance) / 1.0)
        correction = (self.max_residual / 3.0) * torch.tanh(raw) * uncertainty * band_gate
        correction = correction.sum(dim=2, keepdim=True).clamp(-self.max_residual, self.max_residual)
        self.regularizer = correction.square().mean()
        return torch.relu(base + correction)
