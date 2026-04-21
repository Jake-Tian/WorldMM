import json
import os
import logging

logger = logging.getLogger(__name__)


def generate_multiscale_memory(db_name: str = "A1_JAKE", 
                          json_path: str = "data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_30sec.json",
                          diary_dir: str = ".cache/events_diary",
                          save_path: str = "data/EgoLife/EgoLifeCap"):
    """
    Check for existing multiscale episodic memory files.
    
    NOTE: Multiscale generation requires EgoRAG which is no longer in this repo.
    This function verifies existence of precomputed files.
    """
    logger.info(f"Checking for precomputed multiscale files (3min, 10min, 1h) in {save_path}/{db_name}")
    
    granularities = ["3min", "10min", "1h"]
    missing = []
    for g in granularities:
        path = os.path.join(save_path, db_name, f"{db_name}_{g}.json")
        if not os.path.exists(path):
            missing.append(path)
            
    if missing:
        logger.warning(f"Missing precomputed files: {missing}")
        return 0
        
    logger.info("All multiscale files found.")
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
