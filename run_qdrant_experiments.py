import argparse
import json
import time
from typing import Any, Dict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Datatype as QdrantDatatype
from qdrant_client.http.models import Distance, VectorParams

from core.configs import ExperimentConfig, QuantizationType
from utils.config_utils import create_experiment_configs, load_config
from utils.experiment_utils import save_experiment_results


def generate_random_embeddings(
    num_vectors: int, vector_dim: int, dtype: np.dtype = np.float32
) -> np.ndarray:
    """Generate random embeddings with normal distribution."""
    if dtype == np.int8:
        # For int8, generate values between 0 and 255
        return np.random.randint(0, 256, (num_vectors, vector_dim), dtype=dtype)
    return np.random.normal(-5, 5, (num_vectors, vector_dim)).astype(dtype)


def calculate_memory_size(
    num_vectors: int, vector_dim: int, quant_type: QuantizationType
) -> float:
    """Calculate estimated memory size in GB based on quantization type."""
    bytes_per_element = {
        QuantizationType.FLOAT32: 4,
        QuantizationType.FLOAT16: 2,
        QuantizationType.INT8: 1,
        QuantizationType.BINARY: 1 / 8,  # 1 bit per element
    }
    overhead_factor = 1.5
    memory_bytes = (
        num_vectors * vector_dim * bytes_per_element[quant_type] * overhead_factor
    )
    return memory_bytes / (1024**3)  # Convert to GB


def get_quantization_params(experiment: ExperimentConfig):
    """Get Qdrant quantization parameters based on experiment configuration."""
    params = {
        QuantizationType.FLOAT32: (
            QdrantDatatype.FLOAT32,
            np.float32,
            Distance.COSINE,
            None,
            None,
        ),
        QuantizationType.FLOAT16: (
            QdrantDatatype.FLOAT16,
            np.float16,
            Distance.COSINE,
            None,
            None,
        ),
        QuantizationType.INT8: (
            QdrantDatatype.UINT8,
            np.uint8,
            Distance.COSINE,
            None,
            None,
        ),
        QuantizationType.BINARY: (
            QdrantDatatype.FLOAT32,
            np.float32,
            Distance.DOT,
            models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False, rescore=False
                )
            ),
            models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            ),
        ),
    }
    return params[experiment.quantization_type]


def calculate_payload_sizes(
    client: QdrantClient, collection_name: str, batch_size: int, embeddings: np.ndarray
) -> tuple[float, float]:
    """Calculate average payload sizes for upload and retrieval."""
    # Calculate upload request size
    sample_request = {
        "batch": {
            "ids": list(range(batch_size)),
            "payloads": [{"metadata": f"point_{i}"} for i in range(batch_size)],
            "vectors": [emb.tolist() for emb in embeddings],
        }
    }
    avg_upload_size = len(json.dumps(sample_request).encode()) / batch_size

    # Calculate point payload size
    points = client.scroll(
        collection_name=collection_name, with_vectors=True, limit=batch_size
    )[0]
    total_size = sum(
        len(json.dumps(p.model_dump(mode="json")).encode()) for p in points
    )
    avg_point_size = total_size / len(points) if points else 0

    return avg_upload_size, avg_point_size


def run_qdrant_experiments(
    model_name: str,
    experiment_configs: list[ExperimentConfig],
    num_vectors: int = 250_000,
    num_queries: int = 100_000,
    vector_dim: int = 768,
    batch_size: int = 100,
    output_dir: str = "",
    location: str = "localhost",
) -> Dict[str, Any]:

    if not output_dir:
        output_dir = f"results/qdrant/{model_name}"

    # Initialize Qdrant client
    client = QdrantClient(location, prefer_grpc=True)

    for experiment in experiment_configs:
        print(f"\nRunning experiments with {experiment.name}")
        collection_name = f"benchmark_{str(experiment.name)}"

        # Apply PCA if configured
        current_vector_dim = vector_dim
        if experiment.pca_config and experiment.pca_config.n_components:
            if isinstance(experiment.pca_config.n_components, float):
                # Convert percentage to number of components
                current_vector_dim = int(
                    vector_dim * experiment.pca_config.n_components
                )
            else:
                current_vector_dim = experiment.pca_config.n_components

        # Calculate theoretical memory requirements
        memory_size_gb = calculate_memory_size(
            num_vectors, current_vector_dim, experiment.quantization_type
        )
        print(f"Estimated memory requirement: {memory_size_gb:.2f} GB")

        # Get quantization parameters
        qdrant_dtype, numpy_dtype, distance_type, search_params, quantization_config = (
            get_quantization_params(experiment)
        )

        # Generate sample embeddings for payload size calculation
        sample_embeddings = generate_random_embeddings(
            batch_size, current_vector_dim, dtype=numpy_dtype
        )

        avg_request_size, avg_payload_size = calculate_payload_sizes(
            client, collection_name, batch_size, sample_embeddings
        )

        # Delete collection if exists
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        # Create new collection with indexing disabled
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=current_vector_dim, distance=distance_type, datatype=qdrant_dtype
            ),
            quantization_config=quantization_config,
            on_disk_payload=False,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )

        # Upload vectors in batches, generating them on the fly
        print("Uploading vectors to Qdrant...")
        start_time = time.time()

        for i in range(0, num_vectors, batch_size):
            # Generate batch of embeddings
            current_batch_size = min(batch_size, num_vectors - i)
            batch_embeddings = generate_random_embeddings(
                current_batch_size, current_vector_dim, dtype=numpy_dtype
            )

            points = [
                models.PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={"metadata": f"point_{idx}"},
                )
                for idx, embedding in enumerate(batch_embeddings, start=i)
            ]

            client.upsert(collection_name=collection_name, points=points)

            if (i + batch_size) % 10000 == 0:
                print(f"Uploaded {i + batch_size} vectors...")

        upload_duration = time.time() - start_time

        # Re-enable indexing after upload
        print("Re-enabling indexing...")
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )

        # Wait for collection status to become green
        print("Waiting for collection to be ready...")
        while True:
            if client.get_collection(collection_name).status == "green":
                print("Collection is ready!")
                break
            time.sleep(1)

        # Measure retrieval speed 5 times
        print("Measuring retrieval speed (5 runs)...")
        query_embeddings = generate_random_embeddings(
            num_queries, current_vector_dim, dtype=numpy_dtype
        )

        all_query_times = []
        for run in range(5):
            print(f"Run {run + 1}/5...")

            # Execute searches sequentially
            for query in query_embeddings:
                start_time = time.time()
                client.search(
                    collection_name=collection_name,
                    query_vector=query.tolist(),
                    limit=1000,
                    search_params=search_params,
                    with_vectors=False,
                    with_payload=False,
                )
                query_time = time.time() - start_time
                all_query_times.append(query_time)

        # Calculate percentiles
        p50, p90, p95, p99 = np.percentile(all_query_times, [50, 90, 95, 99])
        mean_retrieval_time = np.mean(all_query_times)

        # Save results for this experiment
        experiment_results = {
            "num_vectors": num_vectors,
            "vector_dim": current_vector_dim,
            "theoretical_memory_gb": memory_size_gb,
            "upload_duration_seconds": upload_duration,
            "uploaded_points_per_second": num_vectors / upload_duration,
            "mean_retrieval_time": mean_retrieval_time,
            "retrieval_p50": p50,
            "retrieval_p90": p90,
            "retrieval_p95": p95,
            "retrieval_p99": p99,
            "queries_per_second": 1 / mean_retrieval_time,
            "avg_payload_size_bytes": avg_payload_size,
            "avg_request_payload_size_bytes": avg_request_size,
            "collection_info": client.get_collection(collection_name).model_dump(),
        }

        save_experiment_results(output_dir, experiment.name, experiment_results)

        print(f"Deleting collection {collection_name}...")
        client.delete_collection(collection_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yml",
        help="Path to config file",
    )
    parser.add_argument("--num-vectors", type=int, default=10_000)
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--vector-dim", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--location", type=str, default="localhost")

    args = parser.parse_args()

    config = load_config(args.config)
    experiment_configs = create_experiment_configs(config["experiments"])

    run_qdrant_experiments(
        model_name=config["model_name"],
        experiment_configs=experiment_configs,
        num_vectors=args.num_vectors,
        num_queries=args.num_queries,
        vector_dim=args.vector_dim,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        location=args.location,
    )
