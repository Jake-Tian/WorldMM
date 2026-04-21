from __future__ import annotations

import os
from typing import List, Union

import numpy as np
from openai import OpenAI


class GPTEmbeddingModel:
    """Wrapper for OpenAI text embedding models."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Please export it before using GPT embeddings."
                )
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def encode_text(
        self, texts: Union[str, List[str]], batch_size: int = 128
    ) -> np.ndarray:
        """Encode text into embeddings using OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            embeddings.extend([item.embedding for item in resp.data])

        return np.asarray(embeddings, dtype=np.float32)

    def encode(self, content: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Universal encode method for text."""
        return self.encode_text(content, **kwargs)
