import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Datatype as QdrantDatatype
from qdrant_client.http.models import Distance, VectorParams

from core.configs import ExperimentConfig, PCAConfig, QuantizationType


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_experiment_configs(experiments_config: list) -> list[ExperimentConfig]:

    # Filter for Qdrant-supported quantization types
    supported_types = {
        QuantizationType.FLOAT32,
        QuantizationType.FLOAT16,
        QuantizationType.INT8,
        QuantizationType.BINARY,
    }

    experiment_configs = []
    for exp in experiments_config:
        quant_type = QuantizationType[exp["quantization_type"]]
        if quant_type in supported_types:
            experiment_configs.append(
                ExperimentConfig(
                    exp["name"],
                    quant_type,
                    PCAConfig(exp["pca_config"]),
                )
            )
    return experiment_configs


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

    results = {}

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

        # Map quantization type to Qdrant datatype and numpy dtype
        datatype = QdrantDatatype.FLOAT32  # Default
        dtype = np.float32  # Default
        distance = Distance.COSINE  # Default
        search_params = None  # Default
        quantization_config = None

        if experiment.quantization_type == QuantizationType.FLOAT16:
            datatype = QdrantDatatype.FLOAT16
            dtype = np.float16
        elif experiment.quantization_type == QuantizationType.INT8:
            datatype = QdrantDatatype.UINT8
            dtype = np.uint8
        elif experiment.quantization_type == QuantizationType.BINARY:
            # Use scalar quantization config for binary quantization
            distance = Distance.DOT
            search_params = models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=False,
                )
            )
            quantization_config = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            )

        # Generate a small batch of embeddings to calculate payload size
        sample_embeddings = generate_random_embeddings(
            1, current_vector_dim, dtype=dtype
        )
        sample_vector = sample_embeddings[0].tolist()
        sample_request = {
            "batch": {
                "ids": [0],
                "payloads": [{"metadata": "point_0"}],
                "vectors": [sample_vector],
            }
        }
        payload_size = len(json.dumps(sample_request).encode("utf-8"))
        print(f"Sample request payload size: {payload_size} bytes")

        # Create collection
        vector_params = VectorParams(
            size=current_vector_dim, distance=distance, datatype=datatype
        )

        # Delete collection if exists
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        # Create new collection with indexing disabled
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params,
            quantization_config=quantization_config,
            on_disk_payload=False,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0,
            ),
        )

        # Upload vectors in batches, generating them on the fly
        print("Uploading vectors to Qdrant...")
        start_time = time.time()

        for i in range(0, num_vectors, batch_size):
            # Generate batch of embeddings
            current_batch_size = min(batch_size, num_vectors - i)
            batch_embeddings = generate_random_embeddings(
                current_batch_size, current_vector_dim, dtype=dtype
            )

            # Create points with minimal payload
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
            collection_info = client.get_collection(collection_name)
            if collection_info.status == "green":
                print("Collection is ready!")
                break
            time.sleep(1)  # Wait 1 second before checking again

        # Measure retrieval speed 5 times
        print("Measuring retrieval speed (5 runs)...")
        query_embeddings = generate_random_embeddings(
            num_queries, current_vector_dim, dtype=dtype
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
        all_query_times.sort()
        p50 = np.percentile(all_query_times, 50)
        p90 = np.percentile(all_query_times, 90)
        p95 = np.percentile(all_query_times, 95)
        p99 = np.percentile(all_query_times, 99)

        # Calculate mean retrieval time
        mean_retrieval_time = np.mean(all_query_times)

        # Calculate queries per second - divide 1 by mean retrieval time since that's queries/second
        queries_per_second = 1 / mean_retrieval_time

        # Measure payload size
        sample_point = client.retrieve(
            collection_name=collection_name,
            ids=[0],
            with_vectors=True,
            with_payload=True,
        )[0]

        payload = sample_point.model_dump(mode="json")
        payload_size = len(json.dumps(payload).encode("utf-8"))

        # Get collection info
        collection_info = client.get_collection(collection_name)

        results[experiment.name] = {
            "num_vectors": num_vectors,
            "vector_dim": current_vector_dim,
            "theoretical_memory_gb": memory_size_gb,
            "upload_duration_seconds": upload_duration,
            "points_per_second": num_vectors / upload_duration,
            "mean_retrieval_time": mean_retrieval_time,
            "retrieval_p50": p50,
            "retrieval_p90": p90,
            "retrieval_p95": p95,
            "retrieval_p99": p99,
            "queries_per_second": queries_per_second,
            "payload_size_bytes": payload_size,
            "request_payload_size_bytes": payload_size,
            "collection_info": collection_info.model_dump(),
        }

        # Save results after each experiment
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{output_dir}/results_{experiment.name}.json", "w") as f:
            json.dump(results[experiment.name], f, indent=2)

        # Delete collection after experiment
        print(f"Deleting collection {collection_name}...")
        client.delete_collection(collection_name)

    return results


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

    results = run_qdrant_experiments(
        model_name=config["model_name"],
        experiment_configs=experiment_configs,
        num_vectors=args.num_vectors,
        num_queries=args.num_queries,
        vector_dim=args.vector_dim,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        location=args.location,
    )
