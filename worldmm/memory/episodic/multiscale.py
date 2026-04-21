import json
import os
import logging

logger = logging.getLogger(__name__)

def group_captions(captions, interval_minutes):
    """Group 30s captions into larger intervals (concatenating text)."""
    if not captions:
        return []
    
    grouped = []
    # Interval in seconds: 3min=180, 10min=600, 1h=3600
    interval_seconds = interval_minutes * 60
    
    current_group = []
    current_start_sec = -1
    
    for cap in captions:
        # timestamp format: HHMMSS
        ts = int(cap['start_time']) // 100
        ss = ts % 100
        mm = (ts // 100) % 100
        hh = (ts // 10000)
        total_seconds = hh * 3600 + mm * 60 + ss
        
        if current_start_sec == -1 or total_seconds >= current_start_sec + interval_seconds:
            if current_group:
                # Finish previous group
                text = " ".join([c['text'] for c in current_group])
                grouped.append({
                    "start_time": current_group[0]['start_time'],
                    "end_time": current_group[-1]['end_time'],
                    "text": text,
                    "date": current_group[0]['date'],
                    "video_path": current_group[0]['video_path']
                })
            current_group = [cap]
            current_start_sec = (total_seconds // interval_seconds) * interval_seconds
        else:
            current_group.append(cap)
            
    if current_group:
        text = " ".join([c['text'] for c in current_group])
        grouped.append({
            "start_time": current_group[0]['start_time'],
            "end_time": current_group[-1]['end_time'],
            "text": text,
            "date": current_group[0]['date'],
            "video_path": current_group[0]['video_path']
        })
        
    return grouped

def generate_multiscale_memory(db_name: str = "A1_JAKE", 
                          json_path: str = "data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_30sec.json",
                          diary_dir: str = ".cache/events_diary",
                          save_path: str = "data/EgoLife/EgoLifeCap"):
    """
    Generate multiscale episodic memory files (3min, 10min, 1h) by grouping 30s captions.
    """
    if not os.path.exists(json_path):
        logger.error(f"Base 30sec caption file not found: {json_path}")
        return 0
        
    logger.info(f"Generating multiscale files from {json_path}")
    
    with open(json_path, 'r') as f:
        base_captions = json.load(f)
        
    intervals = {
        "3min": 3,
        "10min": 10,
        "1h": 60
    }
    
    for label, minutes in intervals.items():
        grouped = group_captions(base_captions, minutes)
        out_path = os.path.join(save_path, db_name, f"{db_name}_{label}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(grouped, f, indent=2)
        logger.info(f"Generated {out_path} with {len(grouped)} segments.")
        
    return 0


# Entry point for the script
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize RagAgent with database and JSON path."
    )
    parser.add_argument("--db_name", default="A1_JAKE", help="Name of the Chroma database")
    parser.add_argument("--json_path", default="data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_30sec.json", help="Path to the JSON file for database creation")
    parser.add_argument("--diary_dir", default=".cache/events_diary", help="Path to the diary directory")
    parser.add_argument("--save_path", default="data/EgoLife/EgoLifeCap", help="Path to save the generated events")
    parser.add_argument("--token-output", default="", help="Optional path to save token count JSON")

    args = parser.parse_args()

    total_tokens = generate_multiscale_memory(
        db_name=args.db_name,
        json_path=args.json_path,
        diary_dir=args.diary_dir,
        save_path=args.save_path
    )
    if args.token_output:
        os.makedirs(os.path.dirname(args.token_output) or ".", exist_ok=True)
        with open(args.token_output, "w", encoding="utf-8") as f:
            json.dump({"tokens": int(total_tokens)}, f, indent=2)
