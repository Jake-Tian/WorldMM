from __future__ import annotations

import os
import json
import logging
import re
from string import Template
from typing import Any, Dict, List, Optional, Tuple, Union
import importlib
import functools

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from filelock import FileLock

logger = logging.getLogger(__name__)

PromptLike = Union[str, List[Dict[str, Any]]]

def _default_text_model() -> str:
    # Keep a single place to override model selection.
    return os.getenv("WORLDMM_OPENAI_TEXT_MODEL", "gpt-5-mini")

def generate_text_response(
    prompt: PromptLike,
    text_format: Optional[type] = None,
    *,
    model: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, int]:
    """
    HVM-style OpenAI wrapper.
    """
    client = OpenAI()
    model_name = model or _default_text_model()

    if text_format is None:
        response = client.responses.create(
            model=model_name,
            input=prompt,
            **kwargs,
        )
        tokens = int(getattr(getattr(response, "usage", None), "total_tokens", None) or 0)
        return response.output_text, tokens

    response = client.responses.parse(
        model=model_name,
        input=prompt,
        text_format=text_format,
        **kwargs,
    )
    tokens = int(getattr(getattr(response, "usage", None), "total_tokens", None) or 0)
    return response.output_parsed, tokens

def get_embedding(text: str) -> List[float]:
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding  # type: ignore[no-any-return]

def get_multiple_embeddings(texts: List[str]) -> List[List[float]]:
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [response.data[i].embedding for i in range(len(response.data))]

def dynamic_retry_decorator(func):
    """Decorator that applies retry logic with exponential backoff."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, 'max_retries', 5)
        return retry(
            stop=stop_after_attempt(max_retries), 
            wait=wait_exponential(multiplier=1, min=1, max=10)
        )(func)(self, *args, **kwargs)
    return wrapper

class PromptTemplateManager:
    def __init__(self, role_mapping: Optional[Dict[str, str]] = None):
        self.role_mapping = role_mapping or {"system": "system", "user": "user", "assistant": "assistant"}
        self.templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = {}
        
        # Templates are now in worldmm/templates
        current_file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_file_path)
        self.templates_dir = os.path.join(package_dir, "templates")
        
        self._load_templates()

    def _load_templates(self) -> None:
        if not os.path.exists(self.templates_dir):
            return
        
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                script_name = os.path.splitext(filename)[0]
                try:
                    module_name = f"worldmm.templates.{script_name}"
                    module = importlib.import_module(module_name)
                    prompt_template = getattr(module, "prompt_template")
                    
                    if isinstance(prompt_template, str):
                        self.templates[script_name] = Template(prompt_template)
                    elif isinstance(prompt_template, list):
                        rendered_template = []
                        for item in prompt_template:
                            role = self.role_mapping.get(item["role"], item["role"])
                            content = item["content"]
                            if isinstance(content, str):
                                content = Template(content)
                            rendered_template.append({"role": role, "content": content})
                        self.templates[script_name] = rendered_template
                except Exception as e:
                    logger.error(f"Failed to load template {script_name}: {e}")

    def render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found.")
        
        template = self.templates[name]
        if isinstance(template, Template):
            return template.substitute(**kwargs)
        else:
            return [
                {"role": item["role"], "content": item["content"].substitute(**kwargs)}
                for item in template
            ]

def update_token_memory_json(path: str, day: str, step: str, tokens: int) -> None:
    def _load_json(path: str) -> Dict[str, Any]:
        if not os.path.exists(path): return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception: return {}

    def _save_json(path: str, data: Dict[str, Any]) -> None:
        parent = os.path.dirname(path)
        if parent: os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    tokens = int(tokens or 0)
    if tokens <= 0: return
    lock = FileLock(path + ".lock")
    with lock:
        data = _load_json(path)
        if day not in data or not isinstance(data.get(day), dict):
            data[day] = {}
        data[day][step] = int(data[day].get(step, 0)) + tokens
        _save_json(path, data)


def update_token_eval_json(path: str, qid: str, round_tokens: Dict[str, int]) -> None:
    def _load_json(path: str) -> Dict[str, Any]:
        if not os.path.exists(path): return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception: return {}

    def _save_json(path: str, data: Dict[str, Any]) -> None:
        parent = os.path.dirname(path)
        if parent: os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    lock = FileLock(path + ".lock")
    with lock:
        data = _load_json(path)
        data[str(qid)] = {str(k): int(v) for k, v in round_tokens.items()}
        _save_json(path, data)

def convert_format_to_template(original_string: str, placeholder_mapping: Optional[dict] = None, static_values: Optional[dict] = None) -> str:
    """
    Converts a .format() style string to a Template-style string.

    Args:
        original_string (str): The original string using .format() placeholders.
        placeholder_mapping (dict, optional): Mapping from original placeholder names to new placeholder names.
        static_values (dict, optional): Mapping from original placeholders to static values to be replaced in the new template.

    Returns:
        str: The converted string in Template-style format.
    """
    # Initialize mappings
    placeholder_mapping = placeholder_mapping or {}
    static_values = static_values or {}

    # Regular expression to find .format() style placeholders
    placeholder_pattern = re.compile(r'\{(\w+)\}')

    # Substitute placeholders in the string
    def replace_placeholder(match):
        original_placeholder = match.group(1)

        # If the placeholder is in static_values, substitute its value directly
        if original_placeholder in static_values:
            return str(static_values[original_placeholder])

        # Otherwise, rename the placeholder if needed, or keep it as is
        new_placeholder = placeholder_mapping.get(original_placeholder, original_placeholder)
        return f'${{{new_placeholder}}}'

    # Replace all placeholders
    template_string = placeholder_pattern.sub(replace_placeholder, original_string)

    return template_string
