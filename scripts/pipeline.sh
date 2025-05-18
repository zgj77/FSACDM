set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:-math}"
RUN_DIR="${2:-runs/${DATASET}_$(date +%Y%m%d_%H%M%S)}"
case "$DATASET" in math|tables|music|molecules) ;; *) echo "Unknown dataset: $DATASET" >&2; exit 2;; esac
export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$PWD/.cache/datasets"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
PYTHON="${FSA_PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data/${DATASET}_full}"
if [[ ! -f "$DATA_DIR/provenance.json" || ! -f "$DATA_DIR/train.jsonl" || ! -f "$DATA_DIR/val.jsonl" || ! -f "$DATA_DIR/test.jsonl" ]]; then
  "$PYTHON" scripts/download.py --dataset "$DATASET"
  PREP_ARGS=()
  if [[ -n "${DATA_LIMIT:-}" ]]; then PREP_ARGS+=(--limit "$DATA_LIMIT"); fi
  "$PYTHON" -m fsa_cdm.run prepare --dataset "$DATASET" --parquet-dir "assets/$DATASET/data" --output "$DATA_DIR" "${PREP_ARGS[@]}"
elif [[ -z "${RESUME:-}" ]]; then
  ENCODER_DIR="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["encoder"])' "configs/$DATASET.json")"
  if [[ ! -f "$ENCODER_DIR/config.json" || ( ! -f "$ENCODER_DIR/model.safetensors" && ! -f "$ENCODER_DIR/pytorch_model.bin" ) ]]; then
    "$PYTHON" scripts/download.py --dataset "$DATASET" --model-only
  fi
fi
TRAIN_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then TRAIN_ARGS+=(--max-steps "$MAX_STEPS"); fi
if [[ -n "${RESUME:-}" ]]; then TRAIN_ARGS+=(--resume "$RESUME"); fi
if [[ "${NUM_GPUS:-1}" -gt 1 ]]; then
  "$PYTHON" -m accelerate.commands.launch --multi_gpu --num_processes "$NUM_GPUS" --num_machines 1 --mixed_precision no --dynamo_backend no -m fsa_cdm.run train --config "configs/$DATASET.json" --data "$DATA_DIR" --output "$RUN_DIR" "${TRAIN_ARGS[@]}"
else
  "$PYTHON" -m fsa_cdm.run train --config "configs/$DATASET.json" --data "$DATA_DIR" --output "$RUN_DIR" "${TRAIN_ARGS[@]}"
fi
EVAL_ARGS=()
if [[ -n "${EVAL_LIMIT:-}" ]]; then EVAL_ARGS+=(--limit "$EVAL_LIMIT"); fi
"$PYTHON" -m fsa_cdm.run generate --checkpoint "$RUN_DIR/latest.pt" --data "$DATA_DIR" --output "$RUN_DIR/evaluation" --steps "${INFERENCE_STEPS:-1000}" "${EVAL_ARGS[@]}"
