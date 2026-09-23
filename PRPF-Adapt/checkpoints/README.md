# Pretrained checkpoints

Weights are published as **GitHub Release assets** (git refuses single files above 100 MB).

```bash
mkdir -p checkpoints && cd checkpoints
BASE=https://github.com/GotoHitori/PRPF-Adapt/releases/download/v1.0
for f in sevir_m0_reference.pth sevir_stf_best.pth meteonet_m0_reference.pth meteonet_m3_base.pth meteonet_m3_confidence.pth meteonet_confidence_head.pth; do
  wget -q --show-progress "$BASE/$f"
done
sha256sum *.pth
```

| File | Content | Size | SHA-256 (prefix) | Original name |
| --- | --- | ---: | --- | --- |
| `sevir_m0_reference.pth` | SEVIR M0 (same-protocol control) | 111.4 MB | `6ac97fcd063fb799…` | `SEVIR_STF_M3/checkpoints/m0_reference.pth` |
| `sevir_stf_best.pth` | SEVIR STF model | 98.2 MB | `ecbae78d05ccbe67…` | `SEVIR_STF_M3/checkpoints/m3_best_model.pth` |
| `meteonet_m0_reference.pth` | MeteoNet M0 | 30.1 MB | `1a8be1f790c604bd…` | `MeteoNet_CONFIDENCE_M3/checkpoints/m0_reference.pth` |
| `meteonet_m3_base.pth` | MeteoNet M3 base generator | 17.6 MB | `a803ac79d2e1f0b3…` | `MeteoNet_CONFIDENCE_M3/checkpoints/m3_base_reference.pth` |
| `meteonet_m3_confidence.pth` | MeteoNet M3 + confidence-modulated band bias | 17.6 MB | `2eb84d97f10b5b74…` | `MeteoNet_CONFIDENCE_M3/checkpoints/m3_confidence_best.pth` |
| `meteonet_confidence_head.pth` | MeteoNet event-probability head (required by `meteonet_m3_confidence.pth`) | – | – | `meteonet_confidence_head_20260901.pth` |

Generator weights are stored under the key `gen`; load with `torch.load(path, map_location="cpu")`.
