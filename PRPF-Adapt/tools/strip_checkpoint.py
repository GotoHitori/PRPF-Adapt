"""Release copy of a training checkpoint: keeps generator weights, model args and the
calibration fields the evaluation scripts read; drops optimiser/discriminator/scheduler/RNG
states, model-selection bookkeeping and absolute paths. Re-evaluate the stripped file.
usage: python tools/strip_checkpoint.py IN.pth OUT.pth [--keep-ema]"""
import argparse
import torch

KEEP = ("gen", "args", "epoch", "val_mse", "val_csi_avg", "val_csi_133", "val_csi_by_threshold",
        "best_val_csi", "external_band_bias_dbz", "confidence_band_modulation_dbz",
        "external_band_width_dbz", "head", "metrics", "steps")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("dst"); p.add_argument("--keep-ema", action="store_true")
    a = p.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    out = {k: v for k, v in ck.items() if k in KEEP or (a.keep_ema and k == "ema")}
    cal = ck.get("confidence_calibration")
    if isinstance(cal, dict):
        out["confidence_calibration"] = {k: v for k, v in cal.items() if k in ("method", "search_metrics")}
    torch.save(out, a.dst)
    print("kept:", sorted(out)); print("dropped:", sorted(set(ck) - set(out)))

if __name__ == "__main__":
    main()
