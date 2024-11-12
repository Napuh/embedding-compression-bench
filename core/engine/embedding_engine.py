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
from sentence_transformers.quantization import quantize_embeddings
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
        cache_location: str = ":memory:",
    ):
        self.model: SentenceTransformer = torch.compile(
            SentenceTransformer(model_name, device=device, trust_remote_code=True)
        )
        self.quant_type = quant_type
        self.query_count = 0
        self.token_count = 0
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.model_card_data = ""
        self.benchmark = benchmark
        self.qdrant_client = QdrantClient(cache_location)
        self.collection_stats = {}  # Store min/max values per collection
        self.count_queries = count_queries

        self.pca_config = pca_config
        self.model_card_data = self.model.model_card_data
        self.similarity_fn_name = self.model.similarity_fn_name
        self.calibration_dataset = None

        # Initialize quantization ranges for INT8
        self.int8_ranges = None
        if benchmark:
            self._create_collection(benchmark)

            if pca_config and pca_config.n_components:
                self._fit_pca()

            if quant_type == QuantizationType.INT8:
                self._init_int8_ranges()

    def set_calibration_dataset(self, calibration_dataset: str) -> None:
        self.calibration_dataset = calibration_dataset

    def _get_calibration_embeddings(self) -> np.ndarray:
        if self.calibration_dataset:
            return self._return_all_embeddings(self.calibration_dataset)
        return self._return_all_embeddings(self.benchmark)

    def _init_int8_ranges(self) -> None:
        """Initialize ranges for INT8 quantization using calibration embeddings"""
        calibration_embeddings = self._get_calibration_embeddings()
        if self.pca_config and self.pca:
            calibration_embeddings = self.pca.transform(calibration_embeddings)
        # Calculate ranges (min, max) for each dimension
        self.int8_ranges = np.vstack(
            [
                np.min(calibration_embeddings, axis=0),
                np.max(calibration_embeddings, axis=0),
            ]
        )
        # print(f"Calibration embeddings: {calibration_embeddings.shape}")

    def set_pca_config(self, pca_config: PCAConfig) -> None:
        """Change the current PCA configuration."""
        self.pca_config = pca_config

        if pca_config and pca_config.n_components:
            self._fit_pca()
            if self.quant_type == QuantizationType.INT8:
                self._init_int8_ranges()
        else:
            self.pca = None
            if self.quant_type == QuantizationType.INT8:
                self._init_int8_ranges()

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

    def _return_all_embeddings(self, collection_name: str, limit=250_000) -> np.ndarray:
        """Return all embeddings from a collection."""
        embeddings = []
        offset = None
        batch_size = 1000  # Process 1000 vectors at a time
        total_vectors = 0

        while total_vectors < limit:
            # Adjust batch size for last iteration if needed
            current_batch_size = min(batch_size, limit - total_vectors)

            points, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=current_batch_size,
                offset=offset,
                with_vectors=True,
            )

            if not points:
                break

            # Process this batch
            batch_embeddings = np.vstack([np.array(p.vector) for p in points])
            embeddings.append(batch_embeddings)
            total_vectors += len(points)

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
            return quantize_embeddings(
                x, precision="uint8", ranges=self.int8_ranges
            ).astype(np.float32)

        elif self.quant_type == QuantizationType.BINARY:
            # por el momento esto se almacena como int8 pero es binario (0, 1)
            return (x > 0).astype(np.float32)

        return x

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        BATCH_SIZE = 1024
        all_embeddings = []

        if self.count_queries:
            self.query_count += len(sentences)
            for sentence in sentences:
                self.token_count += len(self.tokenizer.encode(sentence))

        # Process sentences in batches
        for batch_start in range(0, len(sentences), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(sentences))
            batch_sentences = sentences[batch_start:batch_end]

            # Calculate hashes for batch sentences using faster MD5
            batch_hashes = [
                int(hashlib.md5(s.encode()).hexdigest()[:16], 16)
                for s in batch_sentences
            ]

            batch_embeddings = []
            uncached_sentences = []
            uncached_indices = []

            # Try to get embeddings from cache
            if self.benchmark:
                search_results = self.qdrant_client.retrieve(
                    collection_name=self.benchmark, ids=batch_hashes, with_vectors=True
                )

                # Create mapping of found embeddings
                found_embeddings = {r.id: np.array(r.vector) for r in search_results}

                # Process sentences in order
                for i, sentence_hash in enumerate(batch_hashes):
                    if sentence_hash in found_embeddings:
                        batch_embeddings.append(found_embeddings[sentence_hash])
                    else:
                        uncached_sentences.append(batch_sentences[i])
                        uncached_indices.append(i)
            else:
                uncached_sentences = batch_sentences
                uncached_indices = list(range(len(batch_sentences)))

            # Calculate embeddings for uncached sentences
            if uncached_sentences:
                with torch.no_grad():
                    new_embeddings = self.model.encode(uncached_sentences, **kwargs)
                    if isinstance(new_embeddings, torch.Tensor):
                        new_embeddings = new_embeddings.cpu().numpy()
                    else:
                        new_embeddings = np.array(new_embeddings)

                    # Store in cache if benchmark is set
                    if self.benchmark:
                        points = [
                            PointStruct(id=batch_hashes[idx], vector=emb.tolist())
                            for idx, emb in zip(uncached_indices, new_embeddings)
                        ]
                        self.qdrant_client.upsert(
                            collection_name=self.benchmark, points=points
                        )

                    # Insert uncached embeddings in correct positions
                    for i, emb in zip(uncached_indices, new_embeddings):
                        while len(batch_embeddings) < i:
                            batch_embeddings.append(None)  # Pad with None if needed
                        batch_embeddings.insert(i, emb)

            batch_embeddings = np.stack([e for e in batch_embeddings if e is not None])

            # Apply PCA if configured, before quantization
            if self.pca_config and self.pca:
                batch_embeddings = self.pca.transform(batch_embeddings)

            batch_embeddings = self.quantize_tensor(batch_embeddings)
            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        return np.concatenate(all_embeddings, axis=0)
