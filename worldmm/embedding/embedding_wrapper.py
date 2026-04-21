import numpy as np
from typing import Union, List, Optional
from PIL import Image
from .gpt_embedding import GPTEmbeddingModel

class EmbeddingModel:
    """Universal embedding wrapper that routes different modalities to appropriate models"""

    def __init__(self, 
                text_model_name: str = "text-embedding-3-small",
                vis_model_name: str = "VLM2Vec/VLM2Vec-V2.0",
                device: str = "cuda"):
        """
        Initialize embedding models for different modalities

        Args:
            text_model_name: Model name for text embeddings
            vis_model_name: Model name for visual embeddings (defaults to VLM2Vec V2.0)
            device: Device to run models on
        """
        self.device = device
        self.text_model_name = text_model_name
        self.vis_model_name = vis_model_name

        # Lazy load models to avoid heavy dependencies or API key checks if not used
        self._text_model = None
        self._vis_model = None

    @property
    def text_model(self):
        """Lazy loading of text model"""
        if self._text_model is None:
            self._text_model = GPTEmbeddingModel(model_name=self.text_model_name)
        return self._text_model

    @property
    def vis_model(self):
        """Lazy loading of visual model"""
        if self._vis_model is None:
            from .vlm2vecv2 import VLM2VecV2EmbeddingModel
            self._vis_model = VLM2VecV2EmbeddingModel(
                model_name=self.vis_model_name,
                device=self.device
            )
        return self._vis_model

    def encode_text(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode text using configured text embedding model."""
        return self.text_model.encode_text(texts, **kwargs)

    def encode_vis_query(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode visual query using VLM2VecV2 model"""
        return self.vis_model.encode_text(texts, **kwargs)

    def encode_image(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> np.ndarray:
        """Encode images using VLM2VecV2 model"""
        return self.vis_model.encode_image(images, **kwargs)

    def encode_video(self, video_paths: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode videos using VLM2VecV2 model"""
        return self.vis_model.encode_video(video_paths, **kwargs)

    def encode(self, content: Union[str, List[str], Image.Image, List[Image.Image]], 
               modality: str = "text", **kwargs) -> np.ndarray:
        """
        Universal encode method that routes to appropriate model based on modality
        """
        if modality == "text":
            return self.encode_text(content, **kwargs)  # type: ignore
        elif modality == "image":
            return self.encode_image(content, **kwargs)  # type: ignore
        elif modality == "video":
            return self.encode_video(content, **kwargs)  # type: ignore
        elif modality == "vis_query":
            return self.encode_vis_query(content, **kwargs)  # type: ignore
        else:
            raise ValueError(f"Unsupported modality: {modality}")