import torch
import torch.nn as nn
import torch.nn.functional as F

class MeteoNetConfidenceHead(nn.Module):
    """Independent calibrated event-probability head; never changes radar output."""
    def __init__(self, pangu_channels=34, hidden=32, thresholds=(12.0,24.0,32.0), radar_scale=70.0):
        super().__init__()
        self.register_buffer("thresholds_dbz", torch.tensor(thresholds, dtype=torch.float32))
        self.radar_scale = float(radar_scale)
        self.pangu_mapper = nn.Conv2d(pangu_channels, 8, 1)
        self.context = nn.Sequential(nn.Conv2d(12, hidden, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden, len(thresholds), 1))

    def forward(self, prediction, x_obs, pangu_seq):
        batch, leads, _, height, width = prediction.shape
        if pangu_seq.shape[1] != leads:
            pangu_seq = F.interpolate(
                pangu_seq.permute(0, 2, 1, 3, 4),
                size=(leads, pangu_seq.shape[-2], pangu_seq.shape[-1]),
                mode="trilinear", align_corners=False,
            ).permute(0, 2, 1, 3, 4)
        pangu = pangu_seq.reshape(batch * leads, pangu_seq.shape[2], *pangu_seq.shape[-2:])
        pangu = F.interpolate(pangu, size=(height, width), mode="bilinear", align_corners=False)
        pangu = self.pangu_mapper(pangu).view(batch, leads, 8, height, width)
        last = x_obs[:, -1:].expand(-1, leads, -1, -1, -1)
        change = (x_obs[:, -1:] - x_obs[:, -2:-1]).expand(-1, leads, -1, -1, -1)
        lead = torch.linspace(0.0, 1.0, leads, device=prediction.device, dtype=prediction.dtype).view(1, leads, 1, 1, 1).expand(batch, -1, -1, height, width)
        features = torch.cat((prediction, last, change, pangu, lead), dim=2)
        return self.context(features.flatten(0, 1)).view(batch, leads, 3, height, width)

    def probabilities(self, prediction, x_obs, pangu_seq):
        return torch.sigmoid(self(prediction, x_obs, pangu_seq))

    def targets(self, target):
        raw = target * self.radar_scale
        return (raw >= self.thresholds_dbz.view(1, 1, 3, 1, 1)).to(target.dtype)

    def loss(self, prediction, x_obs, pangu_seq, target, pos_weight=None):
        logits = self(prediction, x_obs, pangu_seq)
        labels = self.targets(target)
        weight = None if pos_weight is None else torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype).view(1, 1, 3, 1, 1)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)

def brier_score(probabilities, labels):
    return torch.mean((probabilities - labels) ** 2)

def expected_calibration_error(probabilities, labels, bins=10):
    probabilities, labels = probabilities.detach().flatten(), labels.detach().flatten()
    total = probabilities.new_zeros(())
    for i in range(bins):
        low, high = i / bins, (i + 1) / bins
        mask = (probabilities >= low) & (probabilities <= high if i == bins - 1 else probabilities < high)
        if mask.any(): total = total + mask.float().mean() * (probabilities[mask].mean() - labels[mask].mean()).abs()
    return total
