# WorldMM Simplified Data Structure

The WorldMM project has been reorganized to a simpler, flatter structure inspired by HVM.

## Root Directory Structure

- `worldmm/`: Core unified memory logic (LLM, MLLM, Embedding, Memory).
- `hipporag/`: Core retrieval logic for episodic memory.
- `data/`: Centralized storage for all datasets, memories, and questions.
- `process_memory.py`: Unified script for Step 3 (Episodic Memory Construction).
- `extract_visual_memory.py`: Independent script for extracting visual embeddings for DAY1 to DAY7.
- `reason.py`: Unified script for Step 4 (Iterative Reasoning and Evaluation).
- `pyproject.toml`, `uv.lock`: Dependency management.

## Data Directory Structure (`data/`)

- `data/questions.json`: Merged simple questions (no cross-day) for evaluation.
- `data/memory/`: Precomputed memory results.
  - `data/memory/episodic/`: Episodic triples and multiscale data (stored as `{person}_{day}_triples.json`).
  - `data/memory/visual/`: Visual embeddings (stored as `{person}_{day}_embeddings.pkl`).
- `data/EgoLife/`: Raw dataset from Hugging Face.
  - `data/EgoLife/A1_JAKE/`, ...: Raw video files split by DAY1..DAY7.
  - `data/EgoLife/EgoLifeCap/`: Episodic captions (30sec, 3min, 10min, 1h).
  - `data/EgoLife/EgoLifeCap/Sync/`: Synchronized caption/transcript files.

## Workflow

### Step 3: Memory Construction

#### Episodic Memory
Run `process_memory.py` to build episodic memories (multiscale + triples) for a specific person/day.
```bash
python process_memory.py --person A1_JAKE --day 1
```

#### Visual Memory
Run `extract_visual_memory.py` to extract visual embeddings for all 7 days. This script handles resumption automatically if interrupted.
```bash
python extract_visual_memory.py --person A1_JAKE --gpu 0
```

### Step 4: Iterative Reasoning
Run `reason.py` to answer the simple questions in `data/questions.json`.
```bash
python reason.py --person A1_JAKE --day 1 --model gpt-5-mini
```

Results will be saved to `data/results_DAY1.json`.
