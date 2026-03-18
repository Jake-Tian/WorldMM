#!/usr/bin/env bash
# Day-by-day EgoLifeQA evaluation with per-day video download/cleanup.
# Usage:
#   ./eval.sh [--person A1_JAKE] [--days DAY1,DAY2,...] [--retriever-model gpt-5-mini] [--respond-model gpt-5]

set -euo pipefail

PERSON="A1_JAKE"
RET_MODEL="gpt-5-mini"
RESP_MODEL="gpt-5"
MAX_ROUNDS=5
MAX_ERRORS=5
EPISODIC_K=3
SEMANTIC_K=10
VISUAL_K=3
OUTPUT_DIR="output"
DATA_DIR="data/EgoLife"
DAYS_CSV="DAY1,DAY2,DAY3,DAY4,DAY5,DAY6,DAY7"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --person) PERSON="$2"; shift 2 ;;
        --days) DAYS_CSV="$2"; shift 2 ;;
        --retriever-model) RET_MODEL="$2"; shift 2 ;;
        --respond-model) RESP_MODEL="$2"; shift 2 ;;
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

if ! command -v hf >/dev/null 2>&1; then
    echo "Error: 'hf' CLI not found. Install huggingface_hub CLI first."
    exit 1
fi

BLUE='\033[1;34m'
NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/eval/${PERSON}"
mkdir -p "$LOG_DIR"

normalize_day() {
    local d="${1^^}"
    if [[ "$d" =~ ^DAY[1-7]$ ]]; then
        echo "$d"
    elif [[ "$d" =~ ^[1-7]$ ]]; then
        echo "DAY$d"
    else
        echo ""
    fi
}

cleanup_day_videos() {
    local day="$1"
    local day_dir="${DATA_DIR}/${PERSON}/${day}"
    if [[ -d "$day_dir" ]]; then
        rm -rf "$day_dir"
        echo "Cleaned up videos: $day_dir"
    fi
}

download_one_day() {
    local day="$1"
    local log_file="$2"
    echo -e "${BLUE}Downloading ${PERSON}/${day} from HF...${NC}"
    hf download lmms-lab/EgoLife \
        --repo-type dataset \
        --local-dir "$DATA_DIR" \
        --include "${PERSON}/${day}/*" 2>&1 | tee -a "$log_file"
}

download_visual_memory() {
    local log_file="$1"
    echo -e "${BLUE}Downloading visual_memory for ${PERSON} from JakeTian/EgoLife...${NC}"
    mkdir -p "output/metadata"
    hf download JakeTian/EgoLife \
        --repo-type dataset \
        --local-dir "output/metadata" \
        --include "visual_memory/${PERSON}/*" 2>&1 | tee -a "$log_file"
}

run_eval_one_day() {
    local day="$1"
    local log_file="$2"
    echo -e "${BLUE}Evaluating ${PERSON} ${day} with day-restricted memory...${NC}"
    python eval/eval_one_day.py \
        --subject "$PERSON" \
        --day "$day" \
        --retriever-model "$RET_MODEL" \
        --respond-model "$RESP_MODEL" \
        --max-rounds "$MAX_ROUNDS" \
        --max-errors "$MAX_ERRORS" \
        --episodic-top-k "$EPISODIC_K" \
        --semantic-top-k "$SEMANTIC_K" \
        --visual-top-k "$VISUAL_K" \
        --output-dir "$OUTPUT_DIR" \
        --data-dir "$DATA_DIR" 2>&1 | tee -a "$log_file"
}

IFS=',' read -r -a DAYS_RAW <<< "$DAYS_CSV"
DAYS=()
for d in "${DAYS_RAW[@]}"; do
    d="$(echo "$d" | xargs)"
    nd="$(normalize_day "$d")"
    if [[ -z "$nd" ]]; then
        echo "Invalid day value: $d (expected DAY1..DAY7 or 1..7)"
        exit 1
    fi
    DAYS+=("$nd")
done

# BOOT_LOG="$LOG_DIR/eval_bootstrap_${TIMESTAMP}.log"
# if ! download_visual_memory "$BOOT_LOG"; then
#     echo -e "${BLUE}✗ visual_memory download failed. Abort.${NC}" | tee -a "$BOOT_LOG"
#     exit 1
# fi

for day in "${DAYS[@]}"; do
    DAY_LOG="$LOG_DIR/eval_${day}_${RESP_MODEL//-/_}_${TIMESTAMP}.log"
    echo ""
    echo "============================================================"
    echo "Processing ${PERSON} ${day}"
    echo "Log: $DAY_LOG"
    echo "============================================================"

    if download_one_day "$day" "$DAY_LOG"; then
        if run_eval_one_day "$day" "$DAY_LOG"; then
            echo -e "${BLUE}✓ ${day} evaluation completed.${NC}" | tee -a "$DAY_LOG"
        else
            echo -e "${BLUE}✗ ${day} evaluation failed.${NC}" | tee -a "$DAY_LOG"
            cleanup_day_videos "$day"
            exit 1
        fi
    else
        echo -e "${BLUE}✗ ${day} download failed.${NC}" | tee -a "$DAY_LOG"
        cleanup_day_videos "$day"
        exit 1
    fi

    cleanup_day_videos "$day"
done

echo ""
echo -e "${BLUE}All requested days completed.${NC}"
echo -e "${BLUE}Results: ${OUTPUT_DIR}/${RET_MODEL//-/_}_${RESP_MODEL//-/_}/egolife_eval_${PERSON}_DAY*.json${NC}"
