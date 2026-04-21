#!/usr/bin/env python3
"""
Step 3: Process Memory for WorldMM.
Constructs episodic and visual memories for A1_JAKE.
"""

import os
import json
import argparse
import logging
import concurrent.futures
from tqdm import tqdm
from worldmm.memory.episodic.multiscale import generate_multiscale_memory
from worldmm.memory.episodic.openie import OpenIE
from worldmm.memory.episodic.utils import compute_mdhash_id
from worldmm.llm import PromptTemplateManager, generate_text_response, update_token_memory_json
from collections import defaultdict
from worldmm.memory.semantic import SemanticExtraction, SemanticConsolidation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_memory(person="A1_JAKE", day=None, num_workers=10):
    # Paths
    data_dir = "data/EgoLife"
    cap_dir = os.path.join(data_dir, "EgoLifeCap", person)
    memory_dir = "data/memory"
    
    # Normalize day string (e.g., "1" -> "DAY1")
    day_suffix = ""
    if day:
        if day.isdigit():
            day = f"DAY{day}"
        day_suffix = f"_{day}"

    os.makedirs(memory_dir, exist_ok=True)
    os.makedirs(os.path.join(memory_dir, "episodic"), exist_ok=True)
    os.makedirs(os.path.join(memory_dir, "visual"), exist_ok=True)

    # 1. Load Captions (30sec)
    # If day is specified, we filter for that day
    caption_30sec_path = os.path.join(cap_dir, f"{person}_30sec.json")
    with open(caption_30sec_path, 'r') as f:
        full_caption_data = json.load(f)
    
    if day:
        caption_data = [item for item in full_caption_data if item.get('date') == day]
        logger.info(f"Filtered to {len(caption_data)} captions for {day}")
    else:
        caption_data = full_caption_data

    if not caption_data:
        logger.warning(f"No caption data found for {person} {day if day else ''}")
        return

    # 2. Episodic Memory: Multiscale Memory (Always runs on full data or specific day if implemented in utility)
    # generate_multiscale_memory utility usually handles the splitting internally or we pass filtered path.
    # To keep it simple, we use the filtered data if day is provided.
    day_caption_path = caption_30sec_path
    if day:
        day_caption_path = os.path.join(cap_dir, f"{person}_30sec{day_suffix}.json")
        with open(day_caption_path, 'w') as f:
            json.dump(caption_data, f, indent=2)

    logger.info(f"Generating multiscale memory (3min, 10min, 1h) for {day if day else 'all'}...")
    generate_multiscale_memory(
        db_name=person,
        json_path=day_caption_path,
        diary_dir=".cache/events_diary",
        save_path=cap_dir
    )

    # 3. Episodic Memory: OpenIE Triple Extraction
    logger.info(f"Extracting episodic triples for {day if day else 'all'}...")
    openie = OpenIE(model_name="gpt-5-mini")
    passages = [item['text'] for item in caption_data if 'text' in item]
    ner_results, triple_results = openie.batch_openie(passages, output_dir=os.path.join(memory_dir, "episodic"))
    
    episodic_triples = {}
    for item in caption_data:
        timestamp = item['date'][-1] + item['end_time'].zfill(8)
        text_hash = compute_mdhash_id(item['text'], prefix="chunk-")
        episodic_triples[timestamp] = triple_results.get(text_hash, [])
    
    triples_filename = f"{person}{day_suffix}_triples.json"
    with open(os.path.join(memory_dir, "episodic", triples_filename), 'w') as f:
        json.dump(episodic_triples, f, indent=2)

    # 4. Semantic Memory: Extraction and Consolidation (10-min intervals)
    logger.info(f"Extracting and consolidating semantic memory for {day if day else 'all'}...")
    os.makedirs(os.path.join(memory_dir, "semantic"), exist_ok=True)
    
    semantic_extractor = SemanticExtraction(model_name="gpt-5-mini")
    semantic_consolidator = SemanticConsolidation(model_name="gpt-5-mini")
    
    # Group episodic triples by 10-minute intervals
    # timestamp format: DHHMMSSxx (e.g. 111095800). 10-min bucket: DHHM (e.g. 1110)
    buckets = defaultdict(list)
    for timestamp, triples in episodic_triples.items():
        if triples:
            bucket_key = timestamp[:4]
            buckets[bucket_key].extend(triples)
            
    ongoing_semantic_triples = []
    semantic_history = {}
    
    # Parallel Semantic Extraction
    bucket_keys = sorted(buckets.keys())
    extraction_results = {}
    
    logger.info(f"Extracting semantic triples for {len(bucket_keys)} buckets in parallel using {num_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_bucket = {
            executor.submit(semantic_extractor.extract, buckets[k]): k 
            for k in bucket_keys
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_bucket), total=len(bucket_keys), desc="Semantic Extraction"):
            bucket_key = future_to_bucket[future]
            try:
                result = future.result()
                extraction_results[bucket_key] = result.get("semantic_triples", [])
            except Exception as e:
                logger.error(f"Failed to extract semantic triples for bucket {bucket_key}: {e}")
                extraction_results[bucket_key] = []

    # Sequential Semantic Consolidation
    logger.info("Consolidating semantic triples sequentially...")
    for bucket_key in tqdm(bucket_keys, desc="Semantic Consolidation"):
        new_semantic_triples = extraction_results[bucket_key]
        
        # Consolidate each new semantic triple
        for new_triple in new_semantic_triples:
            consolidation_result = semantic_consolidator.consolidate(new_triple, ongoing_semantic_triples)
            updated_triple = consolidation_result.get("updated_triple", new_triple)
            to_remove = consolidation_result.get("triples_to_remove", [])
            
            # Remove outdated triples in reverse order to keep indices valid
            for idx in sorted(to_remove, reverse=True):
                if 0 <= idx < len(ongoing_semantic_triples):
                    ongoing_semantic_triples.pop(idx)
                    
            # Add the consolidated triple
            ongoing_semantic_triples.append(updated_triple)
            
        # Save the state for this timestamp
        max_ts_in_bucket = max([ts for ts in episodic_triples.keys() if ts.startswith(bucket_key)])
        
        semantic_history[max_ts_in_bucket] = {
            "consolidated_semantic_triples": list(ongoing_semantic_triples)
        }
        
    semantic_filename = f"{person}{day_suffix}_semantic.json"
    with open(os.path.join(memory_dir, "semantic", semantic_filename), 'w') as f:
        json.dump(semantic_history, f, indent=2)

    # Token Monitoring
    token_log_path = os.path.join("data", "token_memory.json")
    day_key = day if day else "ALL"
    
    # Episodic tokens (OpenIE)
    update_token_memory_json(token_log_path, day_key, "episodic_openie", openie.total_tokens)
    
    # Semantic tokens
    semantic_tokens = semantic_extractor.total_tokens + semantic_consolidator.total_tokens
    update_token_memory_json(token_log_path, day_key, "semantic_memory", semantic_tokens)

    logger.info(f"Memory processing complete for {person} {day if day else ''}. Data saved in {memory_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", type=str, default="A1_JAKE")
    parser.add_argument("--day", type=str, default=None, help="Specific day to process (e.g. 1 or DAY1)")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of parallel workers for semantic extraction")
    args = parser.parse_args()
    
    process_memory(person=args.person, day=args.day, num_workers=args.num_workers)
