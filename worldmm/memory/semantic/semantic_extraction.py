"""
Semantic Extraction module for WorldMM.
"""

import json
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union

from ...llm import PromptTemplateManager, generate_text_response
from .utils import SemanticRawOutput

logger = logging.getLogger(__name__)

class SemanticExtraction:
    def __init__(self, model_name: str = "gpt-5-mini"):
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.model_name = model_name
        self.total_tokens = 0
        self._tokens_lock = threading.Lock()

    def extract(self, episodic_triples: List[List[str]]) -> Dict[str, Any]:
        """
        Extract semantic triples from episodic triples.
        
        Args:
            episodic_triples: List of [subject, predicate, object] triples
            
        Returns:
            Dict containing semantic_triples and episodic_evidence
        """
        if not episodic_triples:
            return {"semantic_triples": [], "episodic_evidence": []}
            
        # Format episodic triples for the prompt
        formatted_episodic = "\n".join([f"{i}. {json.dumps(t)}" for i, t in enumerate(episodic_triples)])
        
        messages = self.prompt_template_manager.render(
            name='semantic_extraction',
            episodic_triples=formatted_episodic
        )
        
        try:
            response, tokens = generate_text_response(
                messages,
                text_format=SemanticRawOutput,
                model=self.model_name,
            )
            with self._tokens_lock:
                self.total_tokens += int(tokens or 0)
            return {
                "semantic_triples": response.semantic_triples,
                "episodic_evidence": response.episodic_evidence
            }
        except Exception as e:
            logger.error(f"Failed to extract semantic memory: {e}")
            return {"semantic_triples": [], "episodic_evidence": []}
