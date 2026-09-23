import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sevir_root", type=Path, required=True)
    parser.add_argument("--pangu_root", type=Path, required=True)
    parser.add_argument("--train_periods", type=Path)
    parser.add_argument("--train_pangu", type=Path)
    parser.add_argument("--test_periods", type=Path, required=True)
    parser.add_argument("--test_pangu", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pangu_stats", type=Path)
    parser.add_argument("--confidence_head", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vis_dir", type=Path)
    parser.add_argument("--Tin", type=int, default=5)
    parser.add_argument("--Tout", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_variant", choices=("m0", "m3"), default="m3")
    parser.add_argument("--att_dim", type=int)
    parser.add_argument("--swin_depth", type=int)
    parser.add_argument("--no_ltam", action="store_true")
    parser.add_argument("--no_csmmsa", action="store_true")
    parser.add_argument("--no_crossattn", action="store_true")
    parser.add_argument("--no_channel_fusion", action="store_true")
    parser.add_argument("--no_swin", action="store_true")
    parser.add_argument("--no_temporal", action="store_true")
    args = parser.parse_args()
    from scripts.evaluate_meteonet_confidence import evaluate
    evaluate(SimpleNamespace(
        checkpoint=args.checkpoint,
        confidence_head=args.confidence_head,
        radar_root=args.sevir_root,
        pangu_root=args.pangu_root,
        pangu_stats=args.pangu_stats or Path("pangu_channel_stats.npz"),
        test_ids=args.test_periods,
        output=args.output,
        gpu=args.gpu,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tin=args.Tin,
        tout=args.Tout,
        model_variant=args.model_variant,
    ))


if __name__ == "__main__":
    main()
