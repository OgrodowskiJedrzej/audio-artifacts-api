#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="models"
MODEL_NAME="wavegram_logmel.onnx"
MODEL_URL="https://huggingface.co/OgrodowskiJedrzej/audio-artefacts-pann-based/resolve/main/wavegram_logmel.onnx"

log() {
  echo "[setup-model] $1"
}

mkdir -p "${MODEL_DIR}"

MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

if [[ -f "${MODEL_PATH}" ]]; then
  log "Model already exists: ${MODEL_PATH}"
  log "Skipping download."
else
  log "Downloading model from:"
  log "  ${MODEL_URL}"
  curl -L --fail "${MODEL_URL}" -o "${MODEL_PATH}"
  log "Download completed."
fi

if [[ ! -s "${MODEL_PATH}" ]]; then
  log "ERROR: Model file is empty or missing."
  exit 1
fi

log "Model is ready: ${MODEL_PATH}"
