from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from worldmm.llm import generate_text_response


PromptLike = Union[str, List[Dict[str, Any]]]


class SimpleOpenAILLM:
    """
    Minimal LLM shim for HippoRAG to avoid depending on worldmm.llm.LLMModel/llm_wrapper.

    Supports:
    - generate(messages) -> str
    - generate_with_tokens(messages) -> (str|parsed, tokens)
    - infer(messages=...) -> (raw_response_text, metadata, cache_hit)
      (used by HippoRAG's online OpenIE adapter)
    """

    def __init__(self, model_name: str = "gpt-5-mini"):
        self.model_name = model_name

    def generate(self, prompt: PromptLike, **kwargs: Any) -> Any:
        out, _tok = generate_text_response(prompt, model=self.model_name, **kwargs)
        return out

    def generate_with_tokens(self, prompt: PromptLike, text_format: Optional[type] = None, **kwargs: Any) -> Tuple[Any, int]:
        out, tok = generate_text_response(prompt, text_format=text_format, model=self.model_name, **kwargs)
        return out, int(tok or 0)

    def infer(self, *, messages: PromptLike, text_format: Optional[type] = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any], bool]:
        out, tok = generate_text_response(messages, text_format=text_format, model=self.model_name, **kwargs)
        # Provide a minimal metadata dict compatible with HippoRAG expectations.
        metadata = {
            "finish_reason": "stop",
            "total_tokens": int(tok or 0),
        }
        cache_hit = False
        # HippoRAG's OpenIE adapter expects raw_response to be string-like when not parsed.
        return out, metadata, cache_hit


__all__ = ["SimpleOpenAILLM"]

