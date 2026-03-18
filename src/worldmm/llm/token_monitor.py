from __future__ import annotations

import json
import os
from typing import Any, Dict

from filelock import FileLock


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_json(path: str, data: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def update_token_memory_json(path: str, day: str, step: str, tokens: int) -> None:
    tokens = int(tokens or 0)
    if tokens <= 0:
        return
    lock = FileLock(path + ".lock")
    with lock:
        data = _load_json(path)
        if day not in data or not isinstance(data.get(day), dict):
            data[day] = {}
        data[day][step] = int(data[day].get(step, 0)) + tokens
        _save_json(path, data)


def update_token_eval_json(path: str, qid: str, round_tokens: Dict[str, int]) -> None:
    lock = FileLock(path + ".lock")
    with lock:
        data = _load_json(path)
        data[str(qid)] = {str(k): int(v) for k, v in round_tokens.items()}
        _save_json(path, data)
