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
    return os.getenv("OPENAI_TEXT_MODEL", "gpt-5-mini")

def generate_text_response(
    prompt: PromptLike,
    model: str | None = None,
    text_format: type | None = None,
    **kwargs,
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

def update_token_memory_json(path: str, day: str, task: str, tokens: int) -> None:
    """Thread-safe update of token_memory.json."""
    lock = FileLock(path + ".lock")
    with lock:
        data = {}
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                pass
        
        if day not in data:
            data[day] = {}
        
        data[day][task] = data[day].get(task, 0) + tokens
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

def update_token_eval_json(path: str, question_id: str, channel_tokens: Dict[str, int]) -> None:
    """Thread-safe update of token_eval.json."""
    lock = FileLock(path + ".lock")
    with lock:
        data = {}
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                pass
        
        data[question_id] = channel_tokens
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

def convert_format_to_template(original_string: str, placeholder_mapping: Optional[dict] = None, static_values: Optional[dict] = None) -> str:
    """
    Converts a .format() style string to a Template-style string.
    """
    placeholder_mapping = placeholder_mapping or {}
    static_values = static_values or {}
    placeholder_pattern = re.compile(r'\{(\w+)\}')

    def replace_placeholder(match):
        original_placeholder = match.group(1)
        if original_placeholder in static_values:
            return str(static_values[original_placeholder])
        new_placeholder = placeholder_mapping.get(original_placeholder, original_placeholder)
        return f'${{{new_placeholder}}}'

    template_string = placeholder_pattern.sub(replace_placeholder, original_string)
    return template_string
