#!/bin/bash
# WorldMM Evaluation Script
# Usage: ./script/4_eval.sh [--person <person>] [--max-rounds 5]

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

PERSON="A1_JAKE"
MAX_ROUNDS=5 MAX_ERRORS=5 EPISODIC_K=3 SEMANTIC_K=10 VISUAL_K=3
OUTPUT_DIR="output" DATA_DIR="data/EgoLife"

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --person) PERSON="$2"; shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --max-errors) MAX_ERRORS="$2"; shift 2 ;;
        --episodic-top-k) EPISODIC_K="$2"; shift 2 ;;
        --semantic-top-k) SEMANTIC_K="$2"; shift 2 ;;
        --visual-top-k) VISUAL_K="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/eval/${PERSON}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_gpt_5_mini_$TIMESTAMP.log"

echo -e "${BLUE}Running EgoLifeQA evaluation: $PERSON with gpt-5-mini${NC}"
python eval/eval.py \
    --subject "$PERSON" \
    --max-rounds "$MAX_ROUNDS" \
    --max-errors "$MAX_ERRORS" \
    --episodic-top-k "$EPISODIC_K" \
    --semantic-top-k "$SEMANTIC_K" \
    --visual-top-k "$VISUAL_K" \
    --output-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" 2>&1 | tee "$LOG_FILE"

echo -e "${BLUE}Eval Done! Results: ${OUTPUT_DIR}/gpt_5_mini/egolife_eval_${PERSON}.json${NC}"
