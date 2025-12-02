#!/bin/bash
# WorldMM Preprocessing Script
# Usage: ./script/2_preprocess.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

PERSON="A1_JAKE"

source .venv/bin/activate

cd "$(dirname "$0")/.."

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/preprocess/${PERSON}"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Translating DenseCaption...${NC}"
python data/EgoLife/utils/translate_densecap.py 2>&1 | tee "$LOG_DIR/translate_densecap_$TIMESTAMP.log"

echo -e "${BLUE}Generating Sync data...${NC}"
python data/EgoLife/utils/generate_sync.py 2>&1 | tee "$LOG_DIR/generate_sync_$TIMESTAMP.log"

echo -e "${BLUE}Preprocess Done! Output: output/metadata/*_memory/${PERSON}/${NC}"
