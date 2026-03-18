set -euo pipefail
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

cd "$(dirname "$0")"

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/setup"
mkdir -p "$LOG_DIR"
PERSON="A1_JAKE" STEP="all" MODEL="gpt-5-mini" NUM_FRAMES=16

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo -e "${BLUE}uv could not be found, installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tee "$LOG_DIR/uv_install_$TIMESTAMP.log"
fi

# Set up virtual environment and install dependencies
echo -e "${BLUE}Setting up virtual environment and installing dependencies...${NC}"
uv sync 2>&1 | tee "$LOG_DIR/uv_sync_$TIMESTAMP.log"

echo -e "${BLUE}Generating Sync data...${NC}"
# python data/EgoLife/utils/generate_sync.py 2>&1 | tee "$LOG_DIR/generate_sync_$TIMESTAMP.log"

echo -e "${BLUE}Preprocess Done! Output: output/metadata/*_memory/${PERSON}/${NC}"

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift 2 ;;
        --person) PERSON="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --frames) NUM_FRAMES="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Build Memory
mkdir -p output/metadata/{episodic,semantic}_memory/${PERSON}

LOG_DIR=".log/build_memory/${PERSON}"
mkdir -p "$LOG_DIR"

TOKEN_MEMORY_FILE="token_memory.json"

add_step_tokens() {
    local step_name="$1"
    local token_file="$2"
    python - "$TOKEN_MEMORY_FILE" "$step_name" "$token_file" <<'PY'
import json, os, sys
out_path, step, token_file = sys.argv[1:]
tokens = 0
if os.path.exists(token_file):
    try:
        with open(token_file, "r", encoding="utf-8") as f:
            tokens = int((json.load(f) or {}).get("tokens", 0) or 0)
    except Exception:
        tokens = 0
if tokens <= 0:
    sys.exit(0)
data = {}
if os.path.exists(out_path):
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        data = {}
data[step] = int(data.get(step, 0)) + tokens
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
PY
}

run_episodic() {
    echo -e "${BLUE}Episodic Memory: Generating fine captions...${NC}"
    local tok1="$LOG_DIR/token_generate_fine_caption_$TIMESTAMP.json"
    python preprocess/episodic_memory/generate_fine_caption.py \
        --sync-dir "data/EgoLife/EgoLifeCap/Sync" \
        --output "data/EgoLife/EgoLifeCap/${PERSON}/${PERSON}_30sec.json" \
        --token-output "$tok1" 2>&1 | tee "$LOG_DIR/generate_fine_caption_$TIMESTAMP.log"
    add_step_tokens "generate_fine_caption" "$tok1"
    echo -e "${BLUE}Episodic Memory: Generating multiscale memory...${NC}"
    local tok2="$LOG_DIR/token_multiscale_memory_$TIMESTAMP.json"
    python -m worldmm.memory.episodic.multiscale \
        --db_name "$PERSON" \
        --json_path "data/EgoLife/EgoLifeCap/${PERSON}/${PERSON}_30sec.json" \
        --diary_dir ".cache/events_diary" \
        --save_path "data/EgoLife/EgoLifeCap" \
        --token-output "$tok2" 2>&1 | tee "$LOG_DIR/multiscale_memory_$TIMESTAMP.log"
    add_step_tokens "multiscale_memory" "$tok2"
    echo -e "${BLUE}Episodic Memory: Extracting triples...${NC}"
    local tok3="$LOG_DIR/token_extract_episodic_triples_$TIMESTAMP.json"
    python preprocess/episodic_memory/extract_episodic_triples.py \
        --person "$PERSON" --model "$MODEL" --token-output "$tok3" 2>&1 | tee "$LOG_DIR/episodic_triples_$TIMESTAMP.log"
    add_step_tokens "extract_episodic_triples" "$tok3"
}

run_semantic() {
    echo -e "${BLUE}Semantic Memory: Extracting triples...${NC}"
    local tok4="$LOG_DIR/token_extract_semantic_triples_$TIMESTAMP.json"
    python preprocess/semantic_memory/extract_semantic_triples.py \
        --person "$PERSON" --model "$MODEL" --token-output "$tok4" 2>&1 | tee "$LOG_DIR/semantic_extraction_$TIMESTAMP.log"
    add_step_tokens "extract_semantic_triples" "$tok4"
    echo -e "${BLUE}Semantic Memory: Consolidating...${NC}"
    local tok5="$LOG_DIR/token_consolidate_semantic_memory_$TIMESTAMP.json"
    python preprocess/semantic_memory/consolidate_semantic_memory.py \
        --person "$PERSON" --model "$MODEL" --token-output "$tok5" 2>&1 | tee "$LOG_DIR/semantic_consolidation_$TIMESTAMP.log"
    add_step_tokens "consolidate_semantic_memory" "$tok5"
}

case $STEP in
    all) run_episodic; run_semantic ;;
    episodic) run_episodic ;;
    semantic) run_semantic ;;
    *) echo "Invalid step: $STEP"; exit 1 ;;
esac

echo -e "${BLUE}Build Memory Done! Output: output/metadata/*_memory/${PERSON}/${NC}"
