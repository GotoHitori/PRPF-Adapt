#!/usr/bin/env bash
set -euo pipefail
: "${RADAR_ROOT:?RADAR_ROOT is required}"
: "${PANGU_ROOT:?PANGU_ROOT is required}"
: "${TRAIN_IDS:?TRAIN_IDS is required}"
: "${TEST_IDS:?TEST_IDS is required}"
: "${PANGU_STATS:?PANGU_STATS is required}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m prpf.train --dataset meteonet --model_architecture customer_meteonet --sevir_root "$RADAR_ROOT" --pangu_root "$PANGU_ROOT" --train_periods "$TRAIN_IDS" --train_pangu "$TRAIN_IDS" --test_periods "$TEST_IDS" --test_pangu "$TEST_IDS" --meteonet_train_ids "$TRAIN_IDS" --meteonet_test_ids "$TEST_IDS" --meteonet_radar_root "$RADAR_ROOT" --meteonet_pangu_stats "$PANGU_STATS" --save_dir "${SAVE_DIR:-runs/m0_seed3407}" --model_variant m0 --batch_size "${BATCH_SIZE:-8}" --epochs "${EPOCHS:-10}" --num_workers "${NUM_WORKERS:-4}" --seed "${SEED:-3407}" --train_fraction "${TRAIN_FRACTION:-1.0}" --max_train_steps "${MAX_TRAIN_STEPS:-0}" --checkpoint_every "${CHECKPOINT_EVERY:-1}" --amp
