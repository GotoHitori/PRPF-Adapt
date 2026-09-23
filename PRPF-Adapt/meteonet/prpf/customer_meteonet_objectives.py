from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomerMetricAccumulator:
    def __init__(self, thresholds: Sequence[float], lead_times: int):
        self.thresholds = tuple(float(value) for value in thresholds)
        self.lead_times = int(lead_times)
        self.batch_mse_sum = 0.0
        self.batch_count = 0
        self.counts = {
            threshold: torch.zeros(self.lead_times, 3, dtype=torch.int64)
            for threshold in self.thresholds
        }

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape or prediction.shape[1] != self.lead_times:
            raise ValueError("prediction and target must share shape and configured lead times")
        self.batch_mse_sum += F.mse_loss(prediction, target).item()
        self.batch_count += 1
        for threshold in self.thresholds:
            predicted = prediction >= threshold
            observed = target >= threshold
            for lead in range(self.lead_times):
                pred_lead = predicted[:, lead]
                obs_lead = observed[:, lead]
                self.counts[threshold][lead, 0] += (pred_lead & obs_lead).sum().cpu()
                self.counts[threshold][lead, 1] += (~pred_lead & obs_lead).sum().cpu()
                self.counts[threshold][lead, 2] += (pred_lead & ~obs_lead).sum().cpu()

    def compute(self) -> dict:
        csi_by_threshold = {}
        for threshold, values in self.counts.items():
            lead_scores = []
            for hits, misses, false_alarms in values.tolist():
                denominator = hits + misses + false_alarms
                lead_scores.append(hits / denominator if denominator else 0.0)
            csi_by_threshold[threshold] = sum(lead_scores) / self.lead_times
        return {
            "mse": self.batch_mse_sum / max(self.batch_count, 1),
            "csi_by_threshold": csi_by_threshold,
            "csi_avg": sum(csi_by_threshold.values()) / max(len(csi_by_threshold), 1),
        }


def passes_rounded_targets(metrics: Mapping[str, float], targets: Mapping[str, float],
                            tolerance: float = 5e-5) -> bool:
    if metrics["mse"] > targets["mse"] + tolerance:
        return False
    return all(
        metrics[name] + tolerance >= targets[name]
        for name in ("csi_12", "csi_24", "csi_32", "csi_avg")
    )


def strictly_dominates(metrics: Mapping[str, float], baseline: Mapping[str, float]) -> bool:
    if metrics["mse"] >= baseline["mse"]:
        return False
    return all(
        metrics[name] > baseline[name]
        for name in ("csi_12", "csi_24", "csi_32", "csi_avg")
    )


class WeightedSoftCSILoss(nn.Module):
    def __init__(self, thresholds: Sequence[float], weights: Sequence[float],
                 mean: float, std: float, temperature: float = 2.0):
        super().__init__()
        if len(thresholds) != len(weights):
            raise ValueError("thresholds and weights must have equal length")
        self.thresholds = tuple(float(value) for value in thresholds)
        self.weights = tuple(float(value) for value in weights)
        self.mean = float(mean)
        self.std = float(std)
        self.temperature = float(temperature)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_raw = prediction * self.std + self.mean
        target_raw = target * self.std + self.mean
        total = prediction.new_zeros(())
        for threshold, weight in zip(self.thresholds, self.weights):
            predicted = torch.sigmoid((prediction_raw - threshold) / self.temperature)
            observed = torch.sigmoid((target_raw - threshold) / self.temperature)
            intersection = (predicted * observed).sum()
            union = predicted.sum() + observed.sum() - intersection
            total = total + weight * (1.0 - (intersection + 1.0) / (union + 1.0))
        return total


class LeadTimeWeightedSoftCSILoss(WeightedSoftCSILoss):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_raw = prediction * self.std + self.mean
        target_raw = target * self.std + self.mean
        total = prediction.new_zeros(())
        for threshold, weight in zip(self.thresholds, self.weights):
            lead_losses = []
            for lead in range(prediction.shape[1]):
                predicted = torch.sigmoid(
                    (prediction_raw[:, lead] - threshold) / self.temperature
                )
                observed = torch.sigmoid(
                    (target_raw[:, lead] - threshold) / self.temperature
                )
                intersection = (predicted * observed).sum()
                union = predicted.sum() + observed.sum() - intersection
                lead_losses.append(1.0 - (intersection + 1.0) / (union + 1.0))
            total = total + weight * torch.stack(lead_losses).mean()
        return total


class MultiThresholdTverskyLoss(nn.Module):
    def __init__(self, thresholds: Sequence[float], mean: float, std: float,
                 temperature: float = 2.0, false_negative_weight: float = 0.7):
        super().__init__()
        if not 0.5 < false_negative_weight < 1.0:
            raise ValueError("false_negative_weight must be in (0.5, 1.0)")
        self.thresholds = tuple(float(value) for value in thresholds)
        self.mean = float(mean)
        self.std = float(std)
        self.temperature = float(temperature)
        self.beta = float(false_negative_weight)
        self.alpha = 1.0 - self.beta

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_raw = prediction * self.std + self.mean
        target_raw = target * self.std + self.mean
        losses = []
        for threshold in self.thresholds:
            predicted = torch.sigmoid((prediction_raw - threshold) / self.temperature)
            observed = torch.sigmoid((target_raw - threshold) / self.temperature)
            true_positive = (predicted * observed).sum()
            false_positive = (predicted * (1.0 - observed)).sum()
            false_negative = ((1.0 - predicted) * observed).sum()
            score = (true_positive + 1.0) / (
                true_positive + self.alpha * false_positive
                + self.beta * false_negative + 1.0
            )
            losses.append(1.0 - score)
        return torch.stack(losses).mean()


def clear_sky_distillation_loss(prediction: torch.Tensor, teacher: torch.Tensor,
                                target: torch.Tensor, threshold: float,
                                mean: float, std: float) -> torch.Tensor:
    clear = target * float(std) + float(mean) < float(threshold)
    if not clear.any():
        return prediction.new_zeros(())
    return F.mse_loss(prediction[clear], teacher[clear])


def outside_threshold_band_distillation_loss(
        prediction: torch.Tensor, teacher: torch.Tensor, target: torch.Tensor,
        thresholds: Sequence[float], band_width: float, mean: float, std: float,
) -> torch.Tensor:
    target_raw = target * float(std) + float(mean)
    threshold_tensor = target_raw.new_tensor(tuple(float(value) for value in thresholds))
    distance = (target_raw.unsqueeze(-1) - threshold_tensor).abs().amin(dim=-1)
    outside = distance > float(band_width)
    if not outside.any():
        return prediction.new_zeros(())
    return F.mse_loss(prediction[outside], teacher[outside])


def event_weights_from_maxima(maxima: Iterable[float],
                              thresholds=(12.0, 24.0, 32.0),
                              weights=(1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    if len(weights) != len(thresholds) + 1:
        raise ValueError("weights must have one more value than thresholds")
    result = []
    for maximum in maxima:
        index = sum(float(maximum) >= float(threshold) for threshold in thresholds)
        result.append(float(weights[index]))
    return torch.tensor(result, dtype=torch.double)


def customer_event_sample_weights(dataset) -> torch.Tensor:
    return event_weights_from_maxima(
        dataset.future_max_dbz(index) for index in range(len(dataset))
    )
