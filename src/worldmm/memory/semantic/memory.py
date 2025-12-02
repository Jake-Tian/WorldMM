"""
Semantic Memory module for WorldMM.
"""

import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
import igraph as ig
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass

from ...embedding import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class SemanticTripleEntry:
    """Represents a single semantic triple entry with its metadata."""
    id: str
    subject: str
    predicate: str
    object: str
    timestamp: int  # The timestamp when this triple was consolidated
    
    @property
    def triple(self) -> List[str]:
        """Return the triple as a list."""
        return [self.subject, self.predicate, self.object]
    
    @property
    def text(self) -> str:
        """Return the triple as a joined text string for embedding."""
        return " ".join(self.triple)
    
    def to_display_str(self) -> str:
        """Format triple for display."""
        return f"({self.subject}, {self.predicate}, {self.object})"


def _transform_timestamp(ts_str: str) -> str:
    """Transform timestamp string to human-readable format."""
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"


class SemanticMemory:
    """
    Semantic memory for general knowledge using Personalized PageRank.
    
    This class manages semantic triples (subject, predicate, object) that represent
    consolidated knowledge. It uses a graph-based retrieval approach where:
    - Entities (subjects and objects) are vertices in the graph
    - Triples define edges between entities
    - Retrieval uses Personalized PageRank (PPR) to find relevant triples
    
    The retrieval process:
    1. Index triples up to a given timestamp, building a graph and embeddings
    2. For a query, find top-k similar triples using embedding similarity
    3. Extract entities from those triples for PPR personalization
    4. Run PPR on the entity graph
    5. Score triples by summing PPR scores of their subject and object entities
    6. Return top-k triples by PPR-based score
    
    Attributes:
        embedding_model: Model for computing triple embeddings
        triples: List of all loaded SemanticTripleEntry objects
        indexed_entries: List of entries indexed up to indexed_time
        indexed_time: Timestamp boundary for indexed triples
        graph: igraph Graph with entities as vertices
        embeddings: Tensor of triple embeddings for indexed entries
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
    ):
        """
        Initialize SemanticMemory.
        
        Args:
            embedding_model: Embedding model for computing triple embeddings
        """
        self.embedding_model = embedding_model
        
        # Storage for triples
        self.triples: List[SemanticTripleEntry] = []
        self.triple_id_to_entry: Dict[str, SemanticTripleEntry] = {}
        
        # Indexed state
        self.indexed_entries: List[SemanticTripleEntry] = []
        self.indexed_time: int = 0
        
        # Graph and embeddings for retrieval
        self.graph: Optional[ig.Graph] = None
        self.embeddings: Optional[torch.Tensor] = None
        self.index_to_pos: Dict[str, int] = {}
        self.triple_to_entities: Dict[str, Tuple[str, str]] = {}
        self.entity_to_vertex: Dict[str, int] = {}
    
    def load_triples_from_file(self, file_path: str) -> None:
        """
        Load semantic triples from a JSON file.
        
        Expected format:
        {
            "timestamp1": {
                "consolidated_semantic_triples": [[subj, pred, obj], ...]
            },
            ...
        }
        
        Args:
            file_path: Path to JSON file containing consolidated semantic triples
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.load_triples_from_data(data)
    
    def load_triples_from_data(
        self,
        data: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Load semantic triples from in-memory data.
        
        Args:
            data: Dict mapping timestamp -> {consolidated_semantic_triples}
        """
        for timestamp_str, content in data.items():
            timestamp = int(timestamp_str)
            triples = content.get("consolidated_semantic_triples", [])
            
            for idx, triple in enumerate(triples):
                if len(triple) < 3:
                    logger.warning(f"Skipping invalid triple at {timestamp_str}[{idx}]: {triple}")
                    continue
                
                triple_id = f"semantic_{timestamp}_{idx}"
                
                entry = SemanticTripleEntry(
                    id=triple_id,
                    subject=triple[0],
                    predicate=triple[1],
                    object=triple[2] if len(triple) > 2 else "",
                    timestamp=timestamp,
                )
                self.triples.append(entry)
                self.triple_id_to_entry[triple_id] = entry
        
        # Sort triples by timestamp for efficient indexing
        self.triples.sort(key=lambda e: e.timestamp)
        logger.info(f"Loaded {len(self.triples)} semantic triples")
    
    def index(self, until_time: int) -> None:
        """
        Index semantic triples up to the specified timestamp.
        
        This builds the entity graph and computes embeddings for triples
        with timestamp <= until_time.
        
        Args:
            until_time: Timestamp boundary - index all triples with timestamp <= this value
        """
        # Skip if already indexed beyond this time
        if self.indexed_time >= until_time:
            logger.debug(f"Already indexed up to {self.indexed_time}, skipping index for {until_time}")
            return
        
        # Get entries to index
        entries_to_index = [
            entry for entry in self.triples
            if entry.timestamp <= until_time
        ]
        
        if not entries_to_index:
            logger.debug(f"No entries to index up to {until_time}")
            return
        
        # Collect all unique entities
        all_entities: Set[str] = set()
        self.triple_to_entities = {}
        
        for entry in entries_to_index:
            subj, obj = entry.subject, entry.object
            if subj:
                all_entities.add(subj)
            if obj:
                all_entities.add(obj)
            self.triple_to_entities[entry.id] = (subj, obj)
        
        # Build graph with entities as vertices
        self.graph = ig.Graph()
        entity_list = list(all_entities)
        self.graph.add_vertices(entity_list)
        
        # Create entity to vertex index mapping
        self.entity_to_vertex = {entity: i for i, entity in enumerate(entity_list)}
        
        # Add edges for each triple (connecting subject to object)
        edges_to_add = []
        for entry in entries_to_index:
            subj, obj = self.triple_to_entities[entry.id]
            if subj and obj and subj in self.entity_to_vertex and obj in self.entity_to_vertex:
                subj_vertex = self.entity_to_vertex[subj]
                obj_vertex = self.entity_to_vertex[obj]
                if subj_vertex != obj_vertex:
                    edges_to_add.append((subj_vertex, obj_vertex))
        
        if edges_to_add:
            self.graph.add_edges(edges_to_add)
        
        # Compute embeddings for triples
        all_embeddings = []
        self.index_to_pos = {}
        
        for pos, entry in enumerate(entries_to_index):
            self.index_to_pos[entry.id] = pos
            
            # Compute embedding
            emb = self.embedding_model.encode_text(entry.text)
            if len(emb.shape) == 2:
                emb = emb[0]
            all_embeddings.append(emb)
        
        self.embeddings = torch.tensor(
            np.array(all_embeddings), 
            dtype=torch.float32, 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.indexed_entries = entries_to_index
        self.indexed_time = until_time
        
        logger.info(f"Indexed {len(entries_to_index)} semantic triples up to {until_time}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        as_context: bool = True,
    ) -> Union[List[SemanticTripleEntry], str]:
        """
        Retrieve top-k semantic triples using Personalized PageRank.
        
        The retrieval process:
        1. Compute query embedding and find top-k similar triples
        2. Extract entities from those triples for PPR personalization
        3. Run PPR on the entity graph
        4. Score triples by summing PPR scores of subject and object entities
        5. Return top-k triples by PPR score
        
        Args:
            query: Search query text
            top_k: Number of triples to retrieve
            as_context: If True, return formatted string instead of entries
            
        Returns:
            List of SemanticTripleEntry objects or formatted context string
        """
        if not self.indexed_entries or self.embeddings is None or self.graph is None:
            logger.warning("No triples indexed. Call index(until_time) before retrieve().")
            return "" if as_context else []
        
        device = self.embeddings.device
        
        # Encode query
        query_embedding = self.embedding_model.encode_text(query)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)
        
        # Compute similarities with triple embeddings
        similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
        
        # Get top-k similar triples for personalization
        num_available = len(self.indexed_entries)
        top_k_sim = min(top_k, num_available)
        top_values, top_pos_indices = torch.topk(similarities, top_k_sim)
        
        top_sim_entries = [self.indexed_entries[pos] for pos in top_pos_indices.cpu().tolist()]
        
        # Extract entities from top similar triples for PPR personalization
        personalization_entities: Set[str] = set()
        for entry in top_sim_entries:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            if subj:
                personalization_entities.add(subj)
            if obj:
                personalization_entities.add(obj)
        
        if not personalization_entities:
            # Fallback: return top similar triples by embedding similarity
            if as_context:
                return self.retrieve_triples_as_str(top_sim_entries)
            return top_sim_entries
        
        # Set reset vector for PPR (personalize on entities from top similar triples)
        num_entities = self.graph.vcount()
        entity_list = [self.graph.vs[i]['name'] for i in range(num_entities)]
        reset = [
            1.0 / len(personalization_entities) if entity in personalization_entities else 0.0
            for entity in entity_list
        ]
        
        # Run Personalized PageRank on entities
        ppr_scores = self.graph.personalized_pagerank(
            directed=False,
            damping=0.85,
            reset=reset,
            implementation='prpack'
        )
        
        # Create entity to PPR score mapping
        entity_to_ppr = {entity_list[i]: ppr_scores[i] for i in range(num_entities)}
        
        # Score triples as sum of subject and object PPR scores
        triple_scores: Dict[str, float] = {}
        for entry in self.indexed_entries:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            subj_score = entity_to_ppr.get(subj, 0.0) if subj else 0.0
            obj_score = entity_to_ppr.get(obj, 0.0) if obj else 0.0
            triple_scores[entry.id] = subj_score + obj_score
        
        # Get top-k triples by PPR score
        sorted_entries = sorted(
            self.indexed_entries,
            key=lambda e: triple_scores.get(e.id, 0.0),
            reverse=True
        )[:top_k]
        
        if as_context:
            return self.retrieve_triples_as_str(sorted_entries)
        
        return sorted_entries
    
    def retrieve_triples_as_str(self, entries: List[SemanticTripleEntry]) -> str:
        """
        Format a list of triple entries as context string.
        
        Args:
            entries: List of SemanticTripleEntry objects
            
        Returns:
            Formatted context string
        """
        lines = []
        for entry in entries:
            lines.append(entry.to_display_str())
        return "\n".join(lines)
    
    def cleanup(self) -> None:
        """Explicitly free GPU memory."""
        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def reset_index(self) -> None:
        """Reset the indexed state, clearing graph and embeddings."""
        self.graph = None
        self.embeddings = None
        self.indexed_entries = []
        self.indexed_time = 0
        self.index_to_pos = {}
        self.triple_to_entities = {}
        self.entity_to_vertex = {}
        logger.info("Index reset - graph and embeddings cleared")
    
    def get_indexed_time(self) -> str:
        """Get the current indexed time boundary as human-readable string."""
        return _transform_timestamp(str(self.indexed_time))
    
    def get_triple_by_id(self, triple_id: str) -> Optional[SemanticTripleEntry]:
        """Get a triple entry by its ID."""
        return self.triple_id_to_entry.get(triple_id)
    
    def get_triples_count(self) -> int:
        """Get the total number of loaded triples."""
        return len(self.triples)
    
    def get_indexed_count(self) -> int:
        """Get the number of indexed triples."""
        return len(self.indexed_entries)
