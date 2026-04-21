"""
Semantic Consolidation module for WorldMM.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

from ...llm import PromptTemplateManager, generate_text_response
from .utils import ConsolidationRawOutput

logger = logging.getLogger(__name__)

class SemanticConsolidation:
    def __init__(self, model_name: str = "gpt-5-mini"):
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.model_name = model_name
        self.total_tokens = 0

    def consolidate(self, new_triple: List[str], existing_triples: List[List[str]]) -> Dict[str, Any]:
        """
        Consolidate a new semantic triple with existing ones.
        
        Args:
            new_triple: The new [subject, predicate, object] triple
            existing_triples: List of existing triples
            
        Returns:
            Dict containing updated_triple and triples_to_remove (indices in existing_triples)
        """
        if not existing_triples:
            return {"updated_triple": new_triple, "triples_to_remove": []}
            
        formatted_existing = "\n".join([f"{i}. {json.dumps(t)}" for i, t in enumerate(existing_triples)])
        
        messages = self.prompt_template_manager.render(
            name='semantic_consolidation',
            new_triple=json.dumps(new_triple),
            existing_triples=formatted_existing
        )
        
        try:
            response, tokens = generate_text_response(
                messages,
                text_format=ConsolidationRawOutput,
                model=self.model_name,
            )
            self.total_tokens += int(tokens or 0)
            return {
                "updated_triple": response.updated_triple,
                "triples_to_remove": response.triples_to_remove
            }
        except Exception as e:
            logger.error(f"Failed to consolidate semantic memory: {e}")
            return {"updated_triple": new_triple, "triples_to_remove": []}
