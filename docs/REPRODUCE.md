# Evaluation and training

## SEVIR  (run inside `sevir/stf`; use `sevir/m0` for the control)
```bash
cd sevir/stf; S=../splits
python test.py --sevir_root /path/to/sevir --pangu_root /path/to/pangu_sevir \
  --train_periods $S/sevir_train_periods.txt --train_pangu $S/sevir_train_pangufile.txt \
  --test_periods  $S/sevir_test_periods.txt  --test_pangu  $S/sevir_test_pangufile.txt \
  --checkpoint ../../checkpoints/sevir_stf_best.pth --vis_dir ../../outputs/sevir_stf --device cuda
# training: --phase 1, then --phase 2 --checkpoint <phase-1 best_model.pth>
python train.py --phase 1 <same data arguments> --save_dir ../../runs/sevir_stf_p1 --epochs 100 --batch_size 8 --device cuda
```

## MeteoNet  (run inside `meteonet/`)
```bash
cd meteonet
PYTHONPATH=. python -m prpf.evaluate --sevir_root /path/to/meteonet/radar --pangu_root /path/to/meteonet/pangu \
  --test_periods splits/meteonet_test_5to20.txt --model_variant m3 \
  --checkpoint ../checkpoints/meteonet_m3_confidence.pth --confidence_head ../checkpoints/meteonet_confidence_head.pth \
  --output ../outputs/meteonet_m3_confidence.json --device cuda
# M0: --model_variant m0 --checkpoint ../checkpoints/meteonet_m0_reference.pth

RADAR_ROOT=/path/to/meteonet/radar PANGU_ROOT=/path/to/meteonet/pangu TRAIN_IDS=splits/meteonet_train_5to20.txt \
TEST_IDS=splits/meteonet_test_5to20.txt PANGU_STATS=pangu_channel_stats.npz bash scripts/run_m3.sh   # or run_m0.sh

PYTHONPATH=. python scripts/train_meteonet_confidence.py --gpu 0 --steps 500 --data-root splits \
  --radar-root /path/to/meteonet/radar --pangu-root /path/to/meteonet/pangu --pangu-stats pangu_channel_stats.npz \
  --checkpoint ../checkpoints/meteonet_m3_base.pth --output ../outputs/meteonet_confidence_head.pth
```
