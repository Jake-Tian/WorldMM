from __future__ import annotations

import os
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

def _default_mllm_model() -> str:
    return os.getenv("WORLDMM_OPENAI_MLLM_MODEL", "gpt-5-mini")

def get_response(
    messages: List[Dict[str, Any]],
    text_format: Optional[type] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, int]:
    """
    Minimal OpenAI responses wrapper for multimodal inputs.
    """
    client = OpenAI()
    model_name = model or _default_mllm_model()

    if text_format is None:
        response = client.responses.create(
            model=model_name,
            input=messages,
            **kwargs,
        )
        tokens = int(getattr(getattr(response, "usage", None), "total_tokens", None) or 0)
        return response.output_text, tokens

    response = client.responses.parse(
        model=model_name,
        input=messages,
        text_format=text_format,
        **kwargs,
    )
    tokens = int(getattr(getattr(response, "usage", None), "total_tokens", None) or 0)
    return response.output_parsed, tokens

def generate_messages(
    images: Union[Any, str, Path, np.ndarray, Image.Image, List[Any]],
    prompt: str,
) -> List[Dict[str, Any]]:
    """
    Build OpenAI `responses` input messages for multimodal (text + images).
    """
    if isinstance(images, (str, Path, np.ndarray, Image.Image)):
        images = [images]

    imgs: List[np.ndarray] = []
    for item in images:
        if isinstance(item, Image.Image):
            rgb = np.array(item.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            imgs.append(bgr)
            continue
        if isinstance(item, np.ndarray):
            imgs.append(item)
            continue
        
        p = Path(item)
        if p.is_dir():
            paths = sorted([x for x in p.iterdir() if x.suffix.lower() in [".jpg", ".jpeg"]])
            for img_path in paths:
                img = cv2.imread(str(img_path))
                if img is not None:
                    imgs.append(img)
        else:
            img = cv2.imread(str(p))
            if img is not None:
                imgs.append(img)

    if not imgs:
        raise ValueError("No images provided.")

    base64_frames: List[str] = []
    for img in imgs:
        success, buffer = cv2.imencode(".jpg", img)
        if not success:
            raise ValueError("Failed to encode image array to JPG.")
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
        *[
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{frame}"}
            for frame in base64_frames
        ],
    ]

    return [{"role": "user", "content": content}]
