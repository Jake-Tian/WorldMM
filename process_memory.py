#!/usr/bin/env python3
"""
Step 3: Process Memory for WorldMM.
Constructs episodic and semantic memories for A1_JAKE starting from Sync files.
"""

import os
import json
import glob
import argparse
import logging
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from worldmm.memory.episodic.multiscale import generate_multiscale_memory
from worldmm.memory.episodic.openie import OpenIE
from worldmm.memory.episodic.utils import compute_mdhash_id
from worldmm.llm import generate_text_response, update_token_memory_json
from worldmm.memory.semantic import SemanticExtraction, SemanticConsolidation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONSOLIDATION_PROMPT = """You are an expert egocentric video captioner. Transform these raw video segments (captions/transcripts) into a concise, high-quality first-person narrative for this 30-second window. Focus on key actions and objects. Use "I/me/my".

Input:
{segments}

Output:
"""

def consolidate_30s(segments, model="gpt-5-mini"):
    """Consolidate segments into a narrative using LLM."""
    if not segments: return "", 0
    seg_text = json.dumps(segments, indent=2)
    try:
        content, tokens = generate_text_response([{"role": "user", "content": CONSOLIDATION_PROMPT.format(segments=seg_text)}], model=model)
        return content.strip(), int(tokens or 0)
    except Exception as e:
        logger.error(f"Consolidation error: {e}")
        return " ".join([s['text'] for s in segments]), 0

def build_30sec_from_sync(person, day, data_dir, output_path, model="gpt-5-mini", num_workers=10):
    """Builds the 30sec.json by grouping Sync data into 30s buckets and consolidating via LLM."""
    sync_dir = os.path.join(data_dir, "EgoLifeCap", "Sync")
    sync_files = sorted(glob.glob(os.path.join(sync_dir, f"{person}_{day}_*.json")))
    if not sync_files:
        logger.error(f"No sync files found for {person} {day}")
        return 0

    logger.info(f"Processing {len(sync_files)} sync files for {day}...")
    buckets = defaultdict(list) # key: bucket_start_sec
    video_map = {} # bucket_start_sec -> video_path

    for fpath in sync_files:
        with open(fpath, 'r') as f:
            sync_data = json.load(f)
        for entry in sync_data:
            v_path = os.path.join(data_dir, person, day, entry['video_file'])
            for item in entry.get('data', []):
                start = item['start']
                # Convert HHMMSS to total seconds
                sec = (start // 10000) * 3600 + ((start // 100) % 100) * 60 + (start % 100)
                b_start = (sec // 30) * 30
                buckets[b_start].append(item)
                if b_start not in video_map: video_map[b_start] = v_path

    # Parallel Consolidation
    sorted_keys = sorted(buckets.keys())
    results = []
    total_tokens = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(consolidate_30s, buckets[k], model): k for k in sorted_keys}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Consolidating 30s"):
            k = futures[f]
            text, tokens = f.result()
            total_tokens += tokens
            # Convert bucket_start back to HHMMSS string
            h, m, s = k // 3600, (k % 3600) // 60, k % 60
            ts_str = f"{h:02d}{m:02d}{s:02d}"
            results.append({
                "start_time": ts_str,
                "end_time": ts_str, # Simplification
                "text": text,
                "date": day,
                "video_path": video_map.get(k, "")
            })
    
    results.sort(key=lambda x: x['start_time'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f: json.dump(results, f, indent=2)
    return total_tokens

def process_memory(person="A1_JAKE", day=None, num_workers=10, model="gpt-5-mini"):
    data_dir = "data/EgoLife"
    cap_dir = os.path.join(data_dir, "EgoLifeCap", person)
    memory_dir = "data/memory"
    day_suffix = f"_{day}" if day else ""

    # 1. Build/Load 30s captions
    caption_30sec_path = os.path.join(cap_dir, f"{person}_30sec{day_suffix}.json")
    consolidation_tokens = 0
    if not os.path.exists(caption_30sec_path):
        if not day: raise ValueError("Day required to build memory from Sync")
        consolidation_tokens = build_30sec_from_sync(person, day, data_dir, caption_30sec_path, model, num_workers)
    
    with open(caption_30sec_path, 'r') as f:
        caption_data = json.load(f)

    # 2. Multiscale Memory
    generate_multiscale_memory(db_name=person, json_path=caption_30sec_path, save_path=os.path.dirname(cap_dir))
    if day: # Suffix the generated files
        for g in ["3min", "10min", "1h"]:
            old, new = os.path.join(cap_dir, f"{person}_{g}.json"), os.path.join(cap_dir, f"{person}_{g}{day_suffix}.json")
            if os.path.exists(old):
                if os.path.exists(new): os.remove(new)
                os.rename(old, new)

    # 3. OpenIE Extraction
    openie = OpenIE(model_name=model)
    ner, triples = openie.batch_openie([c['text'] for c in caption_data], output_dir=os.path.join(memory_dir, "episodic"))
    
    episodic_triples = {}
    for item in caption_data:
        # Key: DayChar + HHMMSSxx
        ts = item['date'][-1] + item['start_time'] + "00" 
        text_hash = compute_mdhash_id(item['text'], prefix="chunk-")
        episodic_triples[ts] = triples.get(text_hash, [])
    
    with open(os.path.join(memory_dir, "episodic", f"{person}{day_suffix}_triples.json"), 'w') as f:
        json.dump(episodic_triples, f, indent=2)

    # 4. Semantic Memory
    extractor = SemanticExtraction(model_name=model)
    consolidator = SemanticConsolidation(model_name=model)
    
    # Group into 10min buckets (DHHM)
    buckets = defaultdict(list)
    for ts, tlist in episodic_triples.items():
        if tlist: buckets[ts[:4]].extend(tlist)
    
    sorted_buckets = sorted(buckets.keys())
    ongoing_triples, history = [], {}
    
    # Parallel Extraction
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        fut = {executor.submit(extractor.extract, buckets[b]): b for b in sorted_buckets}
        extracted = {fut[f]: f.result().get("semantic_triples", []) for f in tqdm(concurrent.futures.as_completed(fut), total=len(fut), desc="Semantic Extraction")}

    # Sequential Consolidation
    for b in tqdm(sorted_buckets, desc="Semantic Consolidation"):
        for nt in extracted[b]:
            res = consolidator.consolidate(nt, ongoing_triples)
            for idx in sorted(res.get("triples_to_remove", []), reverse=True): ongoing_triples.pop(idx)
            ongoing_triples.append(res.get("updated_triple", nt))
        max_ts = max([ts for ts in episodic_triples if ts.startswith(b)])
        history[max_ts] = {"consolidated_semantic_triples": list(ongoing_triples)}

    with open(os.path.join(memory_dir, "semantic", f"{person}{day_suffix}_semantic.json"), 'w') as f:
        json.dump(history, f, indent=2)

    # Token Logs
    token_log = os.path.join("data", "token_memory.json")
    d_key = day or "ALL"
    update_token_memory_json(token_log, d_key, "caption_consolidation", consolidation_tokens)
    update_token_memory_json(token_log, d_key, "episodic_openie", openie.total_tokens)
    update_token_memory_json(token_log, d_key, "semantic_memory", extractor.total_tokens + consolidator.total_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", default="A1_JAKE")
    parser.add_argument("--day", help="e.g. DAY1")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()
    process_memory(args.person, args.day, args.num_workers, args.model)
