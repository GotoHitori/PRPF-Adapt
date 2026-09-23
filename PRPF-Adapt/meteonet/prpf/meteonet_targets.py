from dataclasses import asdict, dataclass
import math
from typing import Mapping


METRIC_NAMES = ("mse", "csi_12", "csi_24", "csi_32", "csi_avg")


@dataclass(frozen=True)
class MeteonetTargets:
    mse: float
    csi_12: float
    csi_24: float
    csi_32: float
    csi_avg: float

    @classmethod
    def customer_lr5e4(cls) -> "MeteonetTargets":
        return cls(7.6870, 0.4693, 0.3084, 0.0372, 0.2716)

    def as_metrics(self) -> dict[str, float]:
        return asdict(self)


def _validated_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    values = {}
    for name in METRIC_NAMES:
        if name not in metrics:
            raise KeyError(f"missing metric: {name}")
        value = float(metrics[name])
        if not math.isfinite(value):
            raise ValueError(f"metric {name} must be finite")
        values[name] = value
    if values["mse"] <= 0:
        raise ValueError("metric mse must be positive")
    return values


def weakest_target_score(
    metrics: Mapping[str, float], targets: MeteonetTargets
) -> float:
    values = _validated_metrics(metrics)
    return min(
        targets.mse / values["mse"],
        values["csi_12"] / targets.csi_12,
        values["csi_24"] / targets.csi_24,
        values["csi_32"] / targets.csi_32,
        values["csi_avg"] / targets.csi_avg,
    )


def passes_all_targets(
    metrics: Mapping[str, float], targets: MeteonetTargets
) -> bool:
    try:
        values = _validated_metrics(metrics)
    except (KeyError, TypeError, ValueError):
        return False
    return values["mse"] < targets.mse and all(
        values[name] > getattr(targets, name)
        for name in ("csi_12", "csi_24", "csi_32", "csi_avg")
    )
