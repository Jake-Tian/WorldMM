#!/usr/bin/env python3
"""
Step 4: Reasoning for WorldMM.
Answers simple questions from data/questions.json using constructed memories.
"""

import os
import json
import argparse
import logging
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from worldmm.embedding import EmbeddingModel
from worldmm.llm import PromptTemplateManager, update_token_eval_json
from worldmm.memory import WorldMemory, transform_timestamp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize(text: str) -> str:
    return text.lower().strip().rstrip(".,)")

def extract_choice_letter(text: str) -> str:
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else ""

def evaluate_prediction(prediction: str, gold_letter: str, choices: dict) -> bool:
    pred_letter = extract_choice_letter(prediction)
    if pred_letter == gold_letter:
        return True
    
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices.get(f"choice_{gold_letter.lower()}", ""))
    if pred_norm == gold_candidate:
        return True
    
    return False

def process_single_question(item, world_memory):
    question = item['question']
    gold_answer = item['answer']  # A, B, C, or D
    choices = {
        'choice_a': item.get('choice_a', ''),
        'choice_b': item.get('choice_b', ''),
        'choice_c': item.get('choice_c', ''),
        'choice_d': item.get('choice_d', '')
    }

    # Format question with choices
    question_with_choices = f"{question}\n"
    for letter in ['A', 'B', 'C', 'D']:
        choice_text = choices[f'choice_{letter.lower()}']
        if choice_text:
            question_with_choices += f"({letter}) {choice_text}\n"

    # Ask WorldMemory
    try:
        qa_result = world_memory.ask(question_with_choices)
        prediction = qa_result.answer
        
        # Track Tokens per Channel
        channel_tokens = {"episodic": 0, "semantic": 0, "visual": 0, "total": 0}
        for r in qa_result.round_history:
            t = r.get('token', 0)
            channel_tokens["total"] += t
            if r['decision'] == 'search':
                mem_type = r['memory_type']
                if mem_type in channel_tokens:
                    channel_tokens[mem_type] += t
        
        is_correct = evaluate_prediction(prediction, gold_answer, choices)
        
        return {
            "id": item.get('ID', 'unknown'),
            "is_correct": is_correct,
            "channel_tokens": channel_tokens,
            "result_entry": {
                'ID': item.get('ID'),
                'question': question,
                'gold_answer': gold_answer,
                'prediction': prediction,
                'is_correct': is_correct,
                'reasoning_rounds': qa_result.round_history
            }
        }
    except Exception as e:
        logger.error(f"Error processing question {item.get('ID')}: {e}")
        return None

def reason(person="A1_JAKE", day=None, model="gpt-5-mini", num_workers=4):
    # Paths
    data_dir = "data"
    memory_dir = os.path.join(data_dir, "memory")
    questions_path = os.path.join(data_dir, "questions.json")
    
    # Normalize day string
    day_suffix = ""
    if day:
        if day.isdigit():
            day = f"DAY{day}"
        day_suffix = f"_{day}"
    
    results_path = os.path.join(data_dir, f"results{day_suffix}.json")
    token_eval_path = os.path.join(data_dir, f"token_eval{day_suffix}.json")

    # Load Questions
    if not os.path.exists(questions_path):
        logger.error(f"Questions file not found: {questions_path}")
        return
    with open(questions_path, 'r') as f:
        full_questions = json.load(f)
    
    if day:
        questions = [q for q in full_questions if q.get('query_time', {}).get('date') == day]
        logger.info(f"Filtered to {len(questions)} questions for {day}")
    else:
        questions = full_questions

    # Initialize WorldMemory
    logger.info("Initializing WorldMemory...")
    embed_model = EmbeddingModel()
    world_memory = WorldMemory(
        embedding_model=embed_model,
        retriever_llm_model=model,
        respond_llm_model=model
    )

    # Load Memories
    logger.info("Loading memories...")
    
    # Episodic
    cap_dir = os.path.join(data_dir, "EgoLife/EgoLifeCap", person)
    granularities = ["30sec", "3min", "10min", "1h"]
    caption_files = {}
    for g in granularities:
        path = os.path.join(cap_dir, f"{person}_{g}{day_suffix}.json")
        if not os.path.exists(path):
            path = os.path.join(cap_dir, f"{person}_{g}.json")
        caption_files[g] = path
        
    world_memory.episodic_memory.load_captions_from_files(caption_files)
    
    # Episodic Triples
    triples_filename = f"{person}{day_suffix}_triples.json"
    triples_path = os.path.join(memory_dir, "episodic", triples_filename)
    if os.path.exists(triples_path):
        with open(triples_path, 'r') as f:
            world_memory.episodic_memory.openie_data = json.load(f)
    else:
        logger.warning(f"Triples file not found: {triples_path}. Using empty triples.")

    # Visual
    visual_filename = f"{person}{day_suffix}_embeddings.pkl"
    visual_path = os.path.join(memory_dir, "visual", visual_filename)
    if os.path.exists(visual_path):
        clips_path = os.path.join(cap_dir, f"{person}_30sec{day_suffix}.json")
        if not os.path.exists(clips_path):
            clips_path = os.path.join(cap_dir, f"{person}_30sec.json")
            
        with open(clips_path, 'r') as f:
            clips_data = json.load(f)
        world_memory.visual_memory.load_clips(embeddings_path=visual_path, clips_data=clips_data)
    else:
        logger.warning(f"Visual embeddings not found: {visual_path}. Skipping visual memory.")

    # Semantic
    semantic_filename = f"{person}{day_suffix}_semantic.json"
    semantic_path = os.path.join(memory_dir, "semantic", semantic_filename)
    if os.path.exists(semantic_path):
        world_memory.load_semantic_triples(file_path=semantic_path)
    else:
        logger.warning(f"Semantic memory file not found: {semantic_path}. Skipping semantic memory.")

    # Pre-index all memories to ensure thread-safety during retrieval
    logger.info("Indexing memories...")
    world_memory.index(999999999) # Index everything available

    # Evaluation
    logger.info(f"Starting reasoning for {len(questions)} questions using {num_workers} workers...")
    results = []
    correct_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_question, item, world_memory): item for item in questions}
        
        for future in tqdm(as_completed(futures), total=len(questions), desc="Reasoning"):
            res = future.result()
            if res:
                results.append(res["result_entry"])
                if res["is_correct"]:
                    correct_count += 1
                
                # Update token log sequentially in the main thread
                update_token_eval_json(token_eval_path, res["id"], res["channel_tokens"])

    # Save Results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    accuracy = correct_count / len(questions) if questions else 0
    logger.info(f"Reasoning complete. Accuracy: {accuracy:.2%} ({correct_count}/{len(questions)})")
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", type=str, default="A1_JAKE")
    parser.add_argument("--day", type=str, default=None, help="Specific day to reason about (e.g. 1 or DAY1)")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for reasoning")
    args = parser.parse_args()
    
    reason(person=args.person, day=args.day, model=args.model, num_workers=args.num_workers)

