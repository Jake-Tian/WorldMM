#!/usr/bin/env python3
"""
EgoLifeQA single-day evaluation script using WorldMM unified memory system.
This version restricts both questions and memory to one specific day.
"""

import os
import json
import re
import argparse
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory, QAResult
from worldmm.llm.token_monitor import update_token_eval_json


def load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().rstrip(".,)")


def extract_choice_letter(text: str) -> Optional[str]:
    """Extracts A, B, C... from a prediction like (C), B. Bryan, etc."""
    match = re.match(r"\(?([A-Za-z])[\.\)]?\s*", text.strip())
    return match.group(1).upper() if match else None


def evaluate_prediction(prediction: str, gold_letter: str, choices: Dict[str, str]) -> bool:
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices[gold_letter])

    if pred_norm == gold_candidate:
        return True

    pred_letter = extract_choice_letter(prediction)
    if pred_letter == gold_letter:
        return True

    full_patterns = [
        normalize(f"{gold_letter}. {choices[gold_letter]}"),
        normalize(f"({gold_letter}) {choices[gold_letter]}")
    ]
    if pred_norm in full_patterns:
        return True

    return False


def normalize_day(day: str) -> str:
    """Normalize day input into DAY{N} format."""
    day = str(day).strip().upper()
    if day.startswith("DAY"):
        return day
    return f"DAY{day}"


def day_to_digit(day: str) -> str:
    """Convert DAYN to N."""
    return normalize_day(day).replace("DAY", "")


def filter_caption_data_by_day(caption_data: List[Dict[str, Any]], day: str) -> List[Dict[str, Any]]:
    """Filter caption entries by date field."""
    day_norm = normalize_day(day)
    return [x for x in caption_data if str(x.get("date", "")).upper() == day_norm]


def filter_semantic_data_by_day(semantic_data: Dict[str, Any], day: str) -> Dict[str, Any]:
    """
    semantic_data format:
    {
      "111111111": {"consolidated_semantic_triples": [...]},
      ...
    }
    Filter keys whose first digit matches day number.
    """
    day_digit = day_to_digit(day)
    filtered = {}
    for ts, content in semantic_data.items():
        ts_str = str(ts)
        if ts_str.startswith(day_digit):
            filtered[ts_str] = content
    return filtered


def find_30s_segment(target_timestamp: int, segments_30s: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Find the 30s segment that contains the target timestamp."""
    for segment in segments_30s:
        date = segment.get('date', '')
        start_time_raw = segment.get('start_time', 0)
        end_time_raw = segment.get('end_time', 0)

        day = date.replace('DAY', '').replace('Day', '') if isinstance(date, str) else str(date)

        if isinstance(start_time_raw, str):
            start_time = int(day + start_time_raw.zfill(8))
        elif isinstance(start_time_raw, int):
            start_time = int(day + str(start_time_raw).zfill(8))
        else:
            continue

        if isinstance(end_time_raw, str):
            end_time = int(day + end_time_raw.zfill(8))
        elif isinstance(end_time_raw, int):
            end_time = int(day + str(end_time_raw).zfill(8))
        else:
            continue

        if start_time <= target_timestamp <= end_time:
            return (start_time, end_time)

    return (0, 0)


def parse_target_time(row: Dict[str, Any], segments_30s: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    target_time_list = []

    if "time" in row['target_time'] and row['target_time']["time"]:
        time_str = row['target_time']["time"]
        time_str_upper = time_str.upper()

        if "DAY" in time_str_upper:
            parts = re.split(r'DAY|Day', time_str, maxsplit=1)
            if len(parts) == 2:
                start_time_str = parts[0]
                day_and_end = parts[1].split("_")
                if len(day_and_end) == 2:
                    end_day = day_and_end[0]
                    end_time_str = day_and_end[1]
                    start_day = row['target_time']["date"].replace('DAY', '').replace('Day', '')

                    start_time = int(start_day + start_time_str.zfill(8))
                    end_time = int(end_day + end_time_str.zfill(8))
                    target_time_list.append((start_time, end_time))
        else:
            day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)

    elif "time_list" in row['target_time'] and row['target_time']["time_list"]:
        day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
        for time_str in row['target_time']["time_list"]:
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)

    return target_time_list


def write_day_filtered_caption_files(
    subject: str,
    day: str,
    episodic_caption_files: Dict[str, str],
) -> Dict[str, str]:
    """
    Create day-filtered caption jsons under .cache/eval_one_day and return the new file map.
    This keeps changes minimal while reusing existing world_memory.load_episodic_captions interface.
    """
    cache_dir = os.path.join(".cache", "eval_one_day", subject, normalize_day(day))
    os.makedirs(cache_dir, exist_ok=True)

    out_files: Dict[str, str] = {}
    for granularity, in_path in episodic_caption_files.items():
        data = load_json(in_path)
        filtered = filter_caption_data_by_day(data, day)
        out_path = os.path.join(cache_dir, f"{subject}_{granularity}.json")
        with open(out_path, "w") as f:
            json.dump(filtered, f, indent=2)
        out_files[granularity] = out_path
    return out_files


def main():
    parser = argparse.ArgumentParser(description="EgoLifeQA Day-Specific Evaluation with WorldMM")
    parser.add_argument("--subject", type=str, default="A1_JAKE", help="Subject ID")
    parser.add_argument("--day", type=str, required=True, help="Day to evaluate (e.g., DAY1 or 1)")
    parser.add_argument("--retriever-model", type=str, default="gpt-5-mini", help="LLM model for retrieval (NER, OpenIE)")
    parser.add_argument("--respond-model", type=str, default="gpt-5", help="LLM model for iterative reasoning and generating answers")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum retrieval rounds")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum errors before forcing answer")
    parser.add_argument("--episodic-top-k", type=int, default=3, help="Top-k for episodic retrieval")
    parser.add_argument("--semantic-top-k", type=int, default=10, help="Top-k for semantic retrieval")
    parser.add_argument("--visual-top-k", type=int, default=3, help="Top-k for visual retrieval")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data/EgoLife", help="Data directory")
    args = parser.parse_args()

    day_norm = normalize_day(args.day)

    logger.info("Initializing models...")
    embedding_model = EmbeddingModel()
    retriever_llm_model = LLMModel(model_name=args.retriever_model)
    respond_llm_model = LLMModel(model_name=args.respond_model, fps=1)
    prompt_template_manager = PromptTemplateManager()

    logger.info("Initializing WorldMemory...")
    world_memory = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm_model,
        respond_llm_model=respond_llm_model,
        prompt_template_manager=prompt_template_manager,
        max_rounds=args.max_rounds,
        max_errors=args.max_errors,
    )

    world_memory.set_retrieval_top_k(
        episodic=args.episodic_top_k,
        semantic=args.semantic_top_k,
        visual=args.visual_top_k,
    )

    logger.info("Loading data...")
    subject = args.subject
    data_dir = args.data_dir

    eval_data_path = os.path.join(data_dir, f"EgoLifeQA/EgoLifeQA_{subject}.json")
    eval_data = load_json(eval_data_path)
    eval_data = [row for row in eval_data if str(row.get("query_time", {}).get("date", "")).upper() == day_norm]
    logger.info(f"Filtered questions for {day_norm}: {len(eval_data)}")

    episodic_caption_dir = os.path.join(data_dir, f"EgoLifeCap/{subject}")
    granularities = ["30sec", "3min", "10min", "1h"]
    episodic_caption_files = {
        g: os.path.join(episodic_caption_dir, f"{subject}_{g}.json")
        for g in granularities
    }

    # Build day-filtered episodic caption files
    day_caption_files = write_day_filtered_caption_files(subject, day_norm, episodic_caption_files)
    episodic_captions_30sec = load_json(day_caption_files["30sec"])

    semantic_path = os.path.join(f"output/metadata/semantic_memory/{subject}/semantic_consolidation_results_gpt-5-mini.json")
    semantic_results = load_json(semantic_path)
    semantic_results = filter_semantic_data_by_day(semantic_results, day_norm)
    logger.info(f"Filtered semantic timestamps for {day_norm}: {len(semantic_results)}")

    visual_path = os.path.join(f"output/metadata/visual_memory/{subject}/visual_embeddings.pkl")

    logger.info("Loading day-filtered data into WorldMemory...")
    world_memory.load_episodic_captions(caption_files=day_caption_files)
    world_memory.load_semantic_triples(data=semantic_results)
    world_memory.load_visual_clips(embeddings_path=visual_path, clips_data=episodic_captions_30sec)

    logger.info(f"Starting evaluation on {len(eval_data)} samples...")
    results = []
    evaluate_true = 0
    token_eval_path = "token_eval.json"

    for row in tqdm(eval_data):
        ID = row['ID']
        query_type = row['type']
        question = row['question']
        answer = row['answer']

        choices = {}
        for key, label in [('choice_a', 'A'), ('choice_b', 'B'), ('choice_c', 'C'), ('choice_d', 'D')]:
            if key in row and row[key]:
                choices[label] = row[key]

        query_time = int(row['query_time']["date"][-1] + row['query_time']["time"].zfill(8))
        target_time_list = parse_target_time(row, episodic_captions_30sec)

        logger.info(f"Processing ID {ID}: {question[:50]}...")

        qa_result: Optional[QAResult] = None
        try:
            qa_result = world_memory.answer(
                query=question,
                choices=choices,
                until_time=query_time,
            )
            response = qa_result.answer
        except Exception as e:
            logger.error(f"Error processing ID {ID}: {e}")
            response = "Error"

        evaluate = evaluate_prediction(response, answer, choices)
        evaluate_true += int(evaluate)

        result_entry = {
            "ID": ID,
            "type": query_type,
            "question": question,
            "choices": choices,
            "answer": answer,
            "response": response,
            "round_history": qa_result.round_history if qa_result else [],
            "num_rounds": qa_result.num_rounds if qa_result else 0,
            "evaluate": evaluate,
            "query_time": query_time,
            "target_time": target_time_list,
            "day": day_norm,
        }
        results.append(result_entry)

        if qa_result:
            round_token_map = {
                str(item.get("round_num")): int(item.get("token", 0) or 0)
                for item in qa_result.round_history
            }
            update_token_eval_json(token_eval_path, str(ID), round_token_map)

        logger.info(
            f"ID {ID} Answer: {response}, Gold: {answer}, Correct: {evaluate} "
            f"// Accuracy: {evaluate_true}/{len(results)} = {evaluate_true/len(results):.4f}"
        )

    output_path = os.path.join(
        args.output_dir,
        f"{args.retriever_model.replace('-', '_')}_{args.respond_model.replace('-', '_')}",
        f"egolife_eval_{subject}_{day_norm}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    final_accuracy = evaluate_true / len(results) if results else 0
    logger.info(f"\n{'='*50}")
    logger.info("Evaluation Complete")
    logger.info(f"Subject: {subject}")
    logger.info(f"Day: {day_norm}")
    logger.info(f"Total: {len(results)}")
    logger.info(f"Correct: {evaluate_true}")
    logger.info(f"Accuracy: {final_accuracy:.4f}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*50}")

    world_memory.cleanup()


if __name__ == "__main__":
    main()
