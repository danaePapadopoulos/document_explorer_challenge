import logging
from typing import List, Optional, Sequence, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissIndex:
    def __init__(
        self,
        chunk_texts: List[str],
        chunk_embeddings: np.ndarray,
        index_type: str,
        quantizer_type: Optional[str] = None,
    ):
        """
        Initialize FAISS index with given chunk texts and embeddings.
        """
        self.chunk_texts = chunk_texts
        self.chunk_embeddings = chunk_embeddings
        self.index_type = index_type
        self.quantizer_type = quantizer_type

        self.index = self.create_index(
            embeddings=self.chunk_embeddings,
            index_type=self.index_type,
            quantizer_type=self.quantizer_type,
        )

    def create_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "IndexFlatL2",
        quantizer_type: Optional[str] = None,
        nlist: int = 50,
        m: int = 8,
        bits: int = 8,
    ) -> faiss.Index:
        """
        Create and train a FAISS index using the given settings.
        Supports Flat, IVF, and IVFPQ index types.

        Parameters
        ----------
        embeddings : np.ndarray
            2D array of shape (n_samples, dim) containing the vectors to index.
        index_type : str, optional
            Type of index to create. Supported values:
            - 'IndexFlatL2'
            - 'IndexFlatIP'
            - 'IndexIVFFlat'
            - 'IndexIVFPQ'
            Default is 'IndexFlatL2'.
        quantizer_type : str, optional
            Quantizer type for IVF indexes. Supported:
            - 'IndexFlatL2'
            - 'IndexFlatIP'
            Ignored for flat indexes.
        nlist : int, optional
            Number of Voronoi cells (clusters) for IVF indexes. Default is 50.
        m : int, optional
            Number of subquantizers for IVFPQ. Default is 8.
        bits : int, optional
            Number of bits per code for IVFPQ. Default is 8.

        Returns
        -------
        faiss.Index
            A trained FAISS index containing the provided embeddings.

        Raises
        ------
        ValueError
            If an unsupported index_type or quantizer_type is provided,
            or if embeddings array is not 2D.
        """
        # Validate input
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got {embeddings.ndim}D.")

        nb_vectors, dim = embeddings.shape

        # Define supported types
        flat_index_classes = {
            "IndexFlatL2": faiss.IndexFlatL2,
            "IndexFlatIP": faiss.IndexFlatIP,
        }

        # Helper: instantiate a flat index or quantizer
        def _make_flat(name: str) -> faiss.Index:
            if name not in flat_index_classes:
                raise ValueError(f"Unsupported quantizer_type: {name!r}")
            return flat_index_classes[name](dim)

        # Build/instantiate the appropriate FAISS index
        if index_type in flat_index_classes:
            index = _make_flat(index_type)

        elif index_type == "IndexIVFFlat":
            if quantizer_type is None:
                raise ValueError("quantizer_type is required for IndexIVFFlat.")
            quantizer = _make_flat(quantizer_type)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)

        elif index_type == "IndexIVFPQ":
            if quantizer_type is None:
                raise ValueError("quantizer_type is required for IndexIVFPQ.")
            quantizer = _make_flat(quantizer_type)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)

        else:
            raise ValueError(f"Unsupported index_type: {index_type!r}")

        logger.debug(
            "Created FAISS index %s (trained=%s)", index_type, index.is_trained
        )

        # Train index if necessary (non-flat types)
        if not index.is_trained:
            index.train(embeddings)
            logger.debug("Trained index on %d vectors (dim=%d)", nb_vectors, dim)

        index.add(embeddings)  # add vectors
        logger.info("Added %d vectors to the index", nb_vectors)

        return index

    def get_top_k_vectors(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        n_probe: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the FAISS index for the top_k nearest vectors to the query.

        Parameters
        ----------
        query_embedding : array-like or np.matrix
            Either:
            - A 1-D sequence of floats (length d), or
            - A 2-D ndarray of shape (1, d), or
            - An existing np.matrix of shape (1, d).
        top_k : int, optional
            Number of nearest neighbors to retrieve. Defaults to 5.
        n_probe : int, optional
            Number of inverted lists to probe (FAISS `nprobe`). If `None`, leaves
            the index’s default unchanged.
            (An "inverted list” is exactly the set of vectors that fell into
            one Voronoi cell (cluster))

        Returns
        -------
        distances : np.ndarray
            Array of shape (1, top_k) with distances (or similarity scores).
        indices : np.ndarray
            Array of shape (1, top_k) with the indices of nearest neighbors.
        """
        # Normalize input to 2D matrix shape (1, d)
        # If user already gave us a np.matrix, just check its shape
        if isinstance(query_embedding, np.matrix):
            qmat = query_embedding
        else:
            # Convert lists or ndarrays into a float32 ndarray
            arr = np.asarray(query_embedding, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim == 2 and arr.shape[0] == 1:
                pass  # already ok
            else:
                raise ValueError(
                    f"`query_embedding` must be shape (d,) or (1, d); got {arr.shape}"
                )
            # Now wrap into a true numpy.matrix
            qmat = np.matrix(arr)

        # Final check
        if qmat.ndim != 2 or qmat.shape[0] != 1:
            raise ValueError(f"Internal error: query matrix has shape {qmat.shape}")

        # Set nprobe if applicable (for IVF-based indexes)
        if n_probe is not None and hasattr(self.index, "nprobe"):
            # type: ignore[attr-defined]
            self.index.nprobe = n_probe
            logger.debug("Set self.index.nprobe = %d", n_probe)

        distances, indices = self.index.search(qmat, top_k)
        logger.debug("Index.search returned %d neighbors", top_k)

        return distances, indices

    def get_chunks_by_indices(
        self,
        texts: Sequence[str],
        indices: np.ndarray,
        distances: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        Return the text chunks corresponding to retrieved indices, with optional scores.

        Parameters
        ----------
        texts : Sequence[str]
            The full list of chunk texts.
        indices : np.ndarray
            Array of shape (1, k) or (k,) containing the retrieved indices.
        distances : Optional[np.ndarray]
            If provided, an array of the same shape as `indices` containing distance or similarity scores.

        Returns
        -------
        List[str]
            A list of formatted strings like "42: (score=0.0234) The chunk text…"
        """
        flat_idxs = indices.ravel()
        flat_dists = distances.ravel() if distances is not None else None

        results: List[str] = []
        for pos, idx in enumerate(flat_idxs):
            if idx < 0 or idx >= len(texts):
                continue  # Skip invalid index

            snippet = texts[idx]
            if flat_dists is not None:
                results.append(
                    f"Chunk top {idx}: (score={flat_dists[pos]:.4f}) {snippet}"
                )
            else:
                results.append(f"{idx}: {snippet}")

        return results

    def retrieve_top_k_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        n_probe: Optional[int] = None,
    ) -> List[str]:
        """
        Full retrieval pipeline: query → search → format top results.

        Parameters
        ----------
        query_embedding : array-like or np.matrix
            Either:
            - A 1-D sequence of floats (length d), or
            - A 2-D ndarray of shape (1, d), or
            - An existing np.matrix of shape (1, d).
        top_k : int, optional
            Number of nearest neighbors to retrieve. Defaults to 5.
        n_probe : int, optional
            Number of inverted lists to probe (FAISS `nprobe`). If `None`, leaves
            the index’s default unchanged.
            (An "inverted list” is exactly the set of vectors that fell into
            one Voronoi cell (cluster))

        Returns
        -------
        List[str]
            A list of formatted strings like "42: (score=0.0234) The chunk text…"
        """
        dists, idxs = self.get_top_k_vectors(query_embedding, top_k, n_probe)
        chunks = self.get_chunks_by_indices(self.chunk_texts, idxs, dists)

        return chunks
