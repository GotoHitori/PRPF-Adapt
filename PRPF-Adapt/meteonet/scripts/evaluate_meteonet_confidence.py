#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prpf.confidence_band_bias import apply_confidence_band_bias
from prpf.customer_meteonet_dataset import MeteoNetTxtPanguDataset
from prpf.meteonet_band_bias import apply_external_band_bias
from prpf.meteonet_confidence import MeteoNetConfidenceHead
from prpf.model_customer_m3 import PRPF_SetGoGAN_Generator

THRESHOLDS = (12.0, 24.0, 32.0)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_ids(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _state_dict(payload, keys):
    if not isinstance(payload, dict):
        return payload
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def _strip_prefixes(state):
    result = {}
    for key, value in state.items():
        clean = key
        for prefix in ("module.", "generator."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        result[clean] = value
    return result


def _device(args):
    if args.device:
        requested = args.device
    else:
        requested = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _load_model(checkpoint, device, model_variant):
    settings = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model = PRPF_SetGoGAN_Generator(
        model_variant=model_variant,
        att_dim=int(settings.get("att_dim", 64)),
        router_max_residual_dbz=float(settings.get("router_max_residual_dbz", 0.15)),
    ).to(device)
    state = _strip_prefixes(_state_dict(checkpoint, ("gen", "model", "state_dict")))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _load_head(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    head = MeteoNetConfidenceHead().to(device)
    state = _state_dict(payload, ("head", "state_dict"))
    head.load_state_dict(_strip_prefixes(state), strict=True)
    head.eval()
    return head


def _resolve_head(args, checkpoint):
    if args.confidence_head:
        path = Path(args.confidence_head)
    else:
        path = Path(args.checkpoint).resolve().parent / "meteonet_confidence_head_20260901.pth"
    if "confidence_band_modulation_dbz" in checkpoint and not path.is_file():
        raise FileNotFoundError(
            "confidence checkpoint requires --confidence-head pointing to meteonet_confidence_head_20260901.pth"
        )
    return path if path.is_file() else None


def _apply_calibration(prediction, x_obs, pangu, checkpoint, head):
    if "confidence_band_modulation_dbz" in checkpoint:
        if head is None:
            raise RuntimeError("confidence head is required by checkpoint")
        probability = head.probabilities(prediction, x_obs, pangu)
        return apply_confidence_band_bias(
            prediction,
            probability,
            checkpoint.get("external_band_bias_dbz", (0.0, 0.0, 0.0)),
            modulation_dbz=checkpoint["confidence_band_modulation_dbz"],
            band_width_dbz=float(checkpoint.get("external_band_width_dbz", 3.0)),
        )
    if "external_band_bias_dbz" in checkpoint:
        return apply_external_band_bias(
            prediction,
            checkpoint["external_band_bias_dbz"],
            band_width_dbz=float(checkpoint.get("external_band_width_dbz", 3.0)),
        )
    return prediction


@torch.no_grad()
def evaluate(args):
    device = _device(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _load_model(checkpoint, device, args.model_variant)
    head_path = _resolve_head(args, checkpoint)
    head = _load_head(head_path, device) if head_path is not None else None

    dataset = MeteoNetTxtPanguDataset(
        _read_ids(args.test_ids),
        str(args.radar_root),
        str(args.pangu_root),
        Tin=args.tin,
        Tout=args.tout,
        pangu_stats_path=str(args.pangu_stats),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    batch_mse_sum = 0.0
    batch_count = 0
    squared_error_sum = 0.0
    element_count = 0
    lead_squared_error = torch.zeros(args.tout, dtype=torch.float64)
    lead_element_count = torch.zeros(args.tout, dtype=torch.float64)
    counts = {threshold: torch.zeros(args.tout, 3, dtype=torch.int64) for threshold in THRESHOLDS}

    for batch_index, (x_obs, pangu, target) in enumerate(loader):
        x_obs = x_obs.to(device).float()
        pangu = pangu.to(device).float()
        target = target.to(device).float()
        prediction, _ = model(x_obs, pangu)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        if prediction.shape != target.shape:
            raise RuntimeError(f"prediction shape {tuple(prediction.shape)} does not match target shape {tuple(target.shape)}")
        prediction = torch.clamp(prediction, 0.0, 1.0)
        prediction = _apply_calibration(prediction, x_obs, pangu, checkpoint, head)
        prediction_raw = torch.clamp(prediction * 70.0, 0.0, 70.0)
        target_raw = torch.clamp(target * 70.0, 0.0, 70.0)
        error = prediction_raw - target_raw

        batch_mse_sum += float(F.mse_loss(prediction_raw, target_raw).item())
        batch_count += 1
        squared_error_sum += float(error.square().sum().item())
        element_count += error.numel()
        lead_squared_error += error.square().sum(dim=(0, 2, 3, 4)).double().cpu()
        lead_element_count += float(error.shape[0] * error.shape[2] * error.shape[3] * error.shape[4])

        for threshold in THRESHOLDS:
            predicted = prediction_raw >= threshold
            observed = target_raw >= threshold
            for lead in range(args.tout):
                p = predicted[:, lead]
                y = observed[:, lead]
                counts[threshold][lead, 0] += (p & y).sum().cpu()
                counts[threshold][lead, 1] += (~p & y).sum().cpu()
                counts[threshold][lead, 2] += (p & ~y).sum().cpu()
        if (batch_index + 1) % 25 == 0:
            print(f"batch {batch_index + 1} / {len(loader)}", flush=True)

    csi_by_threshold = {}
    csi_pooled_by_threshold = {}
    for threshold in THRESHOLDS:
        values = counts[threshold]
        denominators = values.sum(dim=1)
        lead_scores = torch.where(
            denominators > 0,
            values[:, 0].double() / denominators.double(),
            torch.zeros(args.tout, dtype=torch.float64),
        )
        csi_by_threshold[f"{int(threshold)}"] = float(lead_scores.mean().item())
        pooled = values.sum(dim=0)
        pooled_denominator = int(pooled.sum().item())
        csi_pooled_by_threshold[f"{int(threshold)}"] = float(
            pooled[0].item() / pooled_denominator if pooled_denominator else 0.0
        )

    csi_avg = sum(csi_by_threshold.values()) / len(THRESHOLDS)
    metrics = {
        "mse": batch_mse_sum / max(batch_count, 1),
        "mse_pooled": squared_error_sum / max(element_count, 1),
        "mse_lead_avg": float((lead_squared_error / lead_element_count.clamp_min(1.0)).mean().item()),
        "csi_12": csi_by_threshold["12"],
        "csi_24": csi_by_threshold["24"],
        "csi_32": csi_by_threshold["32"],
        "csi_avg": csi_avg,
        "csi_by_threshold": csi_by_threshold,
        "csi_pooled_by_threshold": csi_pooled_by_threshold,
    }
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "confidence_head": str(head_path.resolve()) if head_path is not None else None,
        "device": str(device),
        "samples": len(dataset),
        "metrics": metrics,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-variant", choices=("m0", "m3"), default="m3")
    parser.add_argument("--confidence-head", type=Path)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--pangu-root", type=Path, required=True)
    parser.add_argument("--pangu-stats", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tin", type=int, default=5)
    parser.add_argument("--tout", type=int, default=20)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
