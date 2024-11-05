import hashlib
from typing import Any, Optional

import numpy as np
import tiktoken
import torch
from ml_dtypes import bfloat16, float4_e2m1fn, float8_e4m3fn, float8_e5m2
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from core.configs import PCAConfig, QuantizationType


class EmbeddingEngine:
    def __init__(
        self,
        model_name: str,
        quant_type: Optional[QuantizationType] = None,
        pca_config: PCAConfig = None,
        device: str = "cuda",
        benchmark: Optional[str] = None,
        count_queries: bool = False,
    ):
        self.model: SentenceTransformer = torch.compile(
            SentenceTransformer(model_name, device=device)
        )
        self.quant_type = quant_type
        self.query_count = 0
        self.token_count = 0
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.model_card_data = ""
        self.benchmark = benchmark
        self.qdrant_client = QdrantClient(
            url="localhost"
        )  # TODO: esto puede ser location=:memory:
        self.collection_stats = {}  # Store min/max values per collection
        self.count_queries = count_queries

        self.pca_config = pca_config
        self.model_card_data = self.model.model_card_data
        self.similarity_fn_name = self.model.similarity_fn_name

        if benchmark:
            self._create_collection(benchmark)

            if pca_config.n_components:
                self._fit_pca()

    def set_pca_config(self, pca_config: PCAConfig) -> None:
        """Change the current PCA configuration."""
        self.pca_config = pca_config

        if pca_config.n_components:
            self._fit_pca()
        else:
            self.pca = None

    def _fit_pca(self) -> None:
        """Fit PCA to the benchmark collection."""
        if not self.benchmark:
            raise ValueError("Benchmark is required for PCA fitting")

        embeddings = self._return_all_embeddings(self.benchmark)

        if self.pca_config.n_components <= 1:
            n_components = round(
                self.model.get_sentence_embedding_dimension()
                * self.pca_config.n_components
            )
        else:
            n_components = self.pca_config.n_components

        # Ensure n_components doesn't exceed the minimum dimension
        # this only happends in AILA benchmarks
        max_components = min(embeddings.shape[0], embeddings.shape[1])
        n_components = min(n_components, max_components)

        self.pca = PCA(
            n_components=n_components, random_state=self.pca_config.random_state
        ).fit(embeddings)
        # explained_variance = sum(self.pca.explained_variance_ratio_) * 100
        # print(f"Total explained variance: {explained_variance:.2f}%")

    def _return_all_embeddings(self, collection_name: str) -> np.ndarray:
        """Return all embeddings from a collection."""
        embeddings = []
        offset = None
        batch_size = 1000  # Process 1000 vectors at a time

        while True:

            points, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_vectors=True,
            )

            if not points:
                break

            # Process this batch
            batch_embeddings = np.vstack([np.array(p.vector) for p in points])
            embeddings.append(batch_embeddings)

            # Update offset for next iteration
            if next_offset is None:
                break
            offset = next_offset

        return np.vstack(embeddings)

    def _create_collection(self, collection_name: str) -> None:
        """Create a Qdrant collection if it doesn't exist."""
        try:
            self.qdrant_client.get_collection(collection_name)
        except (UnexpectedResponse, ValueError):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE,
                    datatype="float32",
                ),
            )

    def set_quant_type(self, quant_type: QuantizationType) -> None:
        """Change the current quantization type being used."""
        self.quant_type = quant_type

    def set_benchmark(self, benchmark: str) -> None:
        """Change the current benchmark being used."""
        self.benchmark = benchmark
        self._create_collection(benchmark)

    def _calculate_collection_stats(self, collection_name: str) -> tuple[float, float]:
        """Calculate and cache min/max values for a collection."""
        if collection_name not in self.collection_stats:
            global_min = float("inf")
            global_max = float("-inf")

            # Process collection in batches
            offset = None  # Start from beginning
            batch_size = 100  # Process 100 vectors at a time

            while True:
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_vectors=True,
                )

                if not points:
                    break

                # Process this batch
                batch_embeddings = np.vstack([np.array(p.vector) for p in points])

                # Update global min/max
                batch_min = np.min(batch_embeddings)
                batch_max = np.max(batch_embeddings)
                global_min = min(global_min, batch_min)
                global_max = max(global_max, batch_max)

                # Update offset for next iteration
                if next_offset is None:
                    break
                offset = next_offset

            if global_min != float("inf"):
                self.collection_stats[collection_name] = (global_min, global_max)

        return self.collection_stats.get(collection_name, (None, None))

    def quantize_tensor(self, x: np.ndarray) -> np.ndarray:
        if self.quant_type is None:
            return x.astype(np.float32)

        if self.quant_type == QuantizationType.FLOAT32:
            return x.astype(np.float32)

        if self.quant_type == QuantizationType.FLOAT16:
            return x.astype(np.float16).astype(np.float32)

        elif self.quant_type == QuantizationType.BFLOAT16:
            return x.astype(bfloat16).astype(np.float32)

        elif self.quant_type == QuantizationType.FLOAT8_E4M3:
            return x.astype(float8_e4m3fn).astype(np.float32)

        elif self.quant_type == QuantizationType.FLOAT8_E5M2:
            return x.astype(float8_e5m2).astype(np.float32)

        elif self.quant_type == QuantizationType.FLOAT4_E2M1:
            return x.astype(float4_e2m1fn).astype(np.float32)

        elif self.quant_type == QuantizationType.INT8:
            if not self.benchmark:
                raise ValueError("Benchmark is required for INT8 quantization")

            global_min, global_max = self._calculate_collection_stats(self.benchmark)
            if global_min is not None and global_max is not None:
                # Scale to int8 range [0, 255]
                x_scaled = 255 * (x - global_min) / (global_max - global_min)
                x_int8 = np.clip(np.round(x_scaled), 0, 255).astype(np.int8)
                return x_int8.astype(np.float32)

        elif self.quant_type == QuantizationType.BINARY:
            # por el momento esto se almacena como int8 pero es binario (0, 1)
            return (x > 0).astype(np.float32)

        return x

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:

        # TODO: aqui llega todo a la vez, hay que dividir en batches porque si no no se sube nada al qdrant hasta que se acabe de procesar absolutamente todo

        if self.count_queries:
            self.query_count += len(sentences)

            for sentence in sentences:
                self.token_count += len(self.tokenizer.encode(sentence))

        # Calculate hashes for all sentences
        sentence_hashes = [
            int(hashlib.sha256(s.encode()).hexdigest()[:16], 16) for s in sentences
        ]

        embeddings = []
        uncached_sentences = []
        uncached_indices = []

        # Try to get embeddings from cache
        if self.benchmark:
            search_results = self.qdrant_client.retrieve(
                collection_name=self.benchmark, ids=sentence_hashes, with_vectors=True
            )

            # Create mapping of found embeddings
            found_embeddings = {r.id: np.array(r.vector) for r in search_results}

            # Process sentences in order
            for i, sentence_hash in enumerate(sentence_hashes):
                if sentence_hash in found_embeddings:
                    embeddings.append(found_embeddings[sentence_hash])
                else:
                    uncached_sentences.append(sentences[i])
                    uncached_indices.append(i)
        else:
            uncached_sentences = sentences
            uncached_indices = list(range(len(sentences)))

        # Calculate embeddings for uncached sentences
        if uncached_sentences:
            with torch.no_grad():
                batch_embeddings = self.model.encode(uncached_sentences, **kwargs)
                if isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = batch_embeddings.cpu().numpy()
                else:
                    batch_embeddings = np.array(batch_embeddings)

                # Store in cache if benchmark is set
                if self.benchmark:
                    # Process in batches of 1000
                    batch_size = 1000
                    for i in range(0, len(uncached_indices), batch_size):
                        batch_indices = uncached_indices[i : i + batch_size]
                        batch_embs = batch_embeddings[i : i + batch_size]

                        points = [
                            PointStruct(id=sentence_hashes[idx], vector=emb.tolist())
                            for idx, emb in zip(batch_indices, batch_embs)
                        ]
                        self.qdrant_client.upsert(
                            collection_name=self.benchmark, points=points
                        )

                # Insert uncached embeddings in correct positions
                for i, emb in zip(uncached_indices, batch_embeddings):
                    while len(embeddings) < i:
                        embeddings.append(None)  # Pad with None if needed
                    embeddings.insert(i, emb)

        embeddings = np.stack([e for e in embeddings if e is not None])

        # Apply PCA if configured, before quantization
        if self.pca_config and self.pca:
            embeddings = self.pca.transform(embeddings)

        return self.quantize_tensor(embeddings)
