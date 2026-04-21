#!/usr/bin/env bash
set -euo pipefail

# Step 4: Reasoning for WorldMM.
# Processes data day by day: downloads, processes memory, reasons, and cleans up.

# Configuration
REPO_ID="JakeTian/EgoLife"
PERSON="A1_JAKE"
DATA_DIR="data"
MEMORY_DIR="${DATA_DIR}/memory"
EGOLIFE_DIR="${DATA_DIR}/EgoLife"

# Days to process (can be passed as arguments, e.g., ./run_daily_pipeline.sh DAY1 DAY2)
if [[ "$#" -gt 0 ]]; then
  DAYS=("$@")
else
  DAYS=("DAY1" "DAY2" "DAY3" "DAY4" "DAY5" "DAY6" "DAY7") 
fi

# Ensure base directories exist
mkdir -p "${DATA_DIR}"
mkdir -p "${MEMORY_DIR}/episodic"
mkdir -p "${MEMORY_DIR}/visual"
mkdir -p "${MEMORY_DIR}/semantic"
mkdir -p "${EGOLIFE_DIR}/EgoLifeCap/Sync"
mkdir -p "${EGOLIFE_DIR}/EgoLifeCap/${PERSON}"

cleanup_day() {
  local day="$1"
  local day_suffix="_${day}"
  
  echo "Cleaning up files for ${day}..."
  
  # Remove downloaded synced captions
  rm -f "${EGOLIFE_DIR}/EgoLifeCap/Sync/${PERSON}_${day}_"*.json
  
  # Remove intermediate processed captions
  rm -f "${EGOLIFE_DIR}/EgoLifeCap/${PERSON}/${PERSON}_30sec.json"
  rm -f "${EGOLIFE_DIR}/EgoLifeCap/${PERSON}/${PERSON}_30sec${day_suffix}.json"
  rm -f "${EGOLIFE_DIR}/EgoLifeCap/${PERSON}/${PERSON}_3min${day_suffix}.json"
  rm -f "${EGOLIFE_DIR}/EgoLifeCap/${PERSON}/${PERSON}_10min${day_suffix}.json"
  rm -f "${EGOLIFE_DIR}/EgoLifeCap/${PERSON}/${PERSON}_1h${day_suffix}.json"
  
  # Remove intermediate episodic and semantic memory
  rm -f "${MEMORY_DIR}/episodic/${PERSON}${day_suffix}_triples.json"
  rm -f "${MEMORY_DIR}/episodic/openie_results_"*.json
  rm -f "${MEMORY_DIR}/semantic/${PERSON}${day_suffix}_semantic.json"
  
  # Remove visual memory
  rm -f "${MEMORY_DIR}/visual/${PERSON}_${day}_embeddings.pkl"
  
  # Remove original video clips
  rm -rf "${DATA_DIR}/EgoLife/${PERSON}/${day}"
  
  # Remove questions file
  rm -f "${DATA_DIR}/questions.json"
  
  echo "✓ Cleanup for ${day} complete."
}

process_day() {
  local day="$1"
  local day_suffix="_${day}"
  
  echo ""
  echo "[$(date +%H:%M:%S)] Processing day: ${day}"
  echo "============================================================"

  # 1. Ensure synced caption data for the day
  echo "Checking for synced captions for ${day}..."
  if ! ls "${EGOLIFE_DIR}/EgoLifeCap/Sync/${PERSON}_${day}_"*.json >/dev/null 2>&1; then
    echo "Synced captions missing. Attempting to download..."
    hf download ${REPO_ID} --repo-type dataset \
      --include "EgoLifeCap/Sync/${PERSON}_${day}_*.json" \
      --local-dir "."
    
    mkdir -p "${EGOLIFE_DIR}/EgoLifeCap/Sync"
    mv EgoLifeCap/Sync/* "${EGOLIFE_DIR}/EgoLifeCap/Sync/" 2>/dev/null || true
    rm -rf EgoLifeCap
  fi

  # 2. Generate episodic and semantic memory
  echo "Generating memory for ${day}..."
  if python3 process_memory.py --person "${PERSON}" --day "${day}"; then
    echo "✓ Memory generated"
  else
    echo "✗ Memory generation failed"
    return 1
  fi

  # 3. Download questions, visual memory, and original video
  echo "Downloading questions, visual memory, and video for ${day}..."
  
  # Questions (Move to expected location)
  hf download ${REPO_ID} --repo-type dataset \
    --include "data/questions/EgoLifeQA_${PERSON}_${day}.json" \
    --local-dir "."
  mv "data/questions/EgoLifeQA_${PERSON}_${day}.json" "${DATA_DIR}/questions.json"
    
  # Visual Memory (Move to expected location)
  hf download ${REPO_ID} --repo-type dataset \
    --include "visual_memory/${PERSON}_${day}_embeddings.pkl" \
    --local-dir "."
  mv "visual_memory/${PERSON}_${day}_embeddings.pkl" "${MEMORY_DIR}/visual/${PERSON}_${day}_embeddings.pkl"
  rm -rf visual_memory

  # Original Video
  echo "Downloading video clips for ${day}..."
  hf download ${REPO_ID} --repo-type dataset \
    --include "data/EgoLife/${PERSON}/${day}/*.mp4" \
    --local-dir "."

  # 4. Run reasoning
  echo "Running reasoning for ${day}..."
  if python3 reason.py --person "${PERSON}" --day "${day}"; then
    echo "✓ Reasoning complete"
  else
    echo "✗ Reasoning failed"
    return 1
  fi

  # 5. Cleanup to save space (Keep results.json)
  cleanup_day "${day}"
  
  echo "✓ [${day}] Done (results saved in ${DATA_DIR}/results${day_suffix}.json)"
  return 0
}

# Main Loop
for day in "${DAYS[@]}"; do
  process_day "${day}"
done

echo ""
echo "Pipeline complete."
