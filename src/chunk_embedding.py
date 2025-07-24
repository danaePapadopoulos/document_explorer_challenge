import logging
from typing import Union, List

from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)


class ChunkEmbedder:
    """
    Wrapper for embedding text chunks using various embedding models.

    Currently supports:
    - BGE-M3 (dense vectors)

    Future extensions:
    - Support for sparse vectors
    - Other embedding models (e.g., E5, OpenAI, etc.)
    """

    def __init__(self, model_name: str, vector_type: str = "dense"):
        """
        Initialize the ChunkEmbedder.

        Parameters
        ----------
        model_name : str
            Name of the embedding model to load.
            Currently only 'bge-m3' is supported.
        vector_type : str
            Type of vector representation to return. Options:
            - 'dense' (default)
            - 'sparse' (not yet implemented)
        """
        self.model_name = model_name
        self.vector_type = vector_type
        self.model = None  # lazy-loaded model instance

    def _load_model(self) -> None:
        """
        Load the embedding model if it hasn't been loaded already.
        """
        if self.model_name == "bge-m3" and self.model is None:
            logger.info("Loading embedding model: BGE-M3")
            self.model = BGEM3FlagModel(
                "BAAI/bge-m3",
                use_fp16=False,  # Set to True for faster inference with minor performance trade-off
            )
            logger.info("Model loaded successfully.")

    def embed_query(self, query: str) -> Union[List[float], None]:
        """
        Generate an embedding vector for a given query string.

        Parameters
        ----------
        query : str
            Input query or text chunk to be embedded.

        Returns
        -------
        Union[List[float], None]
            Dense vector representation of the input query.

        Raises
        ------
        NotImplementedError
            If sparse embeddings are requested but not implemented.
        ValueError
            If an unsupported model or vector type is used.
        """
        self._load_model()

        if self.model_name == "bge-m3":
            if self.vector_type == "dense":
                result = self.model.encode(
                    query,
                    batch_size=12,
                    max_length=8192,  # Reduce if shorter context is sufficient, speeds up encoding
                )
                return result["dense_vecs"]
            elif self.vector_type == "sparse":
                raise NotImplementedError("Sparse embedding not yet supported.")
            else:
                raise ValueError(f"Unsupported vector_type: {self.vector_type}")

        raise ValueError(f"Unsupported model_name: {self.model_name}")
