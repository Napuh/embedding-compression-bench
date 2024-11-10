import argparse
import hashlib
import time

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer


def should_use_cache(
    model_name: str,
    benchmark_size: int = 1000,
    batch_size: int = 32,
    collection_name: str = "benchmark_cache",
    location: str = ":memory:",
) -> bool:
    """
    Benchmark whether using cache is faster than direct embedding for a given model.

    Args:
        model: SentenceTransformer model to benchmark
        benchmark_size: Number of sentences to use for benchmarking
        batch_size: Batch size for embedding generation
        collection_name: Name of temporary collection for benchmarking
        location: Location of Qdrant server ("localhost" or URL)

    Returns:
        bool: True if cache retrieval is faster than direct embedding
    """

    model = SentenceTransformer(model_name)

    # Generate random sentences for benchmarking
    sentences = [
        f"This is a long benchmark sentence that contains multiple words and phrases to better simulate real-world usage scenario number {i}:"
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus mollis vulputate porta. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Aliquam finibus dui sed porta scelerisque. Praesent laoreet sapien erat, elementum posuere urna tempus at. Ut quis eleifend neque. Aenean aliquet venenatis sagittis. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis ut ex posuere, fermentum lectus nec, lobortis massa. Sed mauris libero, malesuada ac mauris id, pellentesque pulvinar tortor. Mauris ante ligula, mollis quis lacus porta, condimentum blandit sem. Mauris a sapien at nunc convallis ullamcorper."
        "Quisque lobortis blandit cursus. Maecenas pellentesque pellentesque nunc eu dignissim. Cras malesuada mollis libero non rhoncus. Donec nec velit vitae orci sagittis viverra. Curabitur condimentum a enim et mattis. Mauris facilisis ex at congue lacinia. Proin tempor interdum tellus vitae hendrerit."
        "Cras ornare est id enim ultricies, in volutpat sapien fringilla. Donec quam ligula, finibus in velit vel, dignissim rhoncus odio. Sed tempor semper turpis in suscipit. Nulla fermentum turpis vitae metus posuere, ac dictum ex viverra. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Nullam fringilla est vel elit mollis eleifend. Pellentesque ac libero sed ex commodo tincidunt ac non enim. Suspendisse potenti. Nulla at consequat nibh. Pellentesque bibendum enim id orci finibus dapibus. In quis est posuere, malesuada nulla eu, efficitur quam. Sed a arcu interdum, dignissim libero vel, placerat risus."
        "Cras bibendum lacinia magna nec tincidunt. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Ut finibus diam at porta lobortis. Suspendisse enim enim, pharetra suscipit nunc eget, imperdiet scelerisque lectus. Interdum et malesuada fames ac ante ipsum primis in faucibus. Etiam leo leo, iaculis id cursus non, pellentesque non magna. Fusce pharetra est vitae placerat aliquam. In ut augue nunc. Pellentesque sed lacus vehicula, lobortis nibh vitae, interdum magna. Fusce scelerisque dui eu viverra suscipit. Ut eu suscipit lorem, quis vestibulum nisi. Morbi et mi et sem vehicula commodo. Suspendisse ac ligula erat."
        "Quisque viverra mauris ut massa euismod, ut luctus arcu aliquam. In lacinia dui ac eleifend sodales. Phasellus eleifend mollis neque at hendrerit. Proin lacinia vitae arcu eget placerat. Donec aliquet, elit a pellentesque rhoncus, quam neque interdum risus, id malesuada dolor nulla ut justo. Integer pretium est ut malesuada condimentum. Sed sed elit ut est gravida mattis id vitae tellus. Integer faucibus elementum mattis. Sed porta imperdiet augue non dapibus. Duis ac mi nec massa luctus porta. Mauris congue lectus ut tortor molestie, a mollis libero sollicitudin. Curabitur magna odio, volutpat at nulla ullamcorper, sollicitudin luctus risus. Maecenas consequat eros in finibus volutpat. Maecenas at lacus dignissim, sagittis arcu sed, facilisis arcu."
        for i in range(benchmark_size)
    ]

    # Calculate hashes
    sentence_hashes = [
        int(hashlib.md5(s.encode()).hexdigest()[:16], 16) for s in sentences
    ]

    # Setup temporary Qdrant collection
    qdrant_client = QdrantClient(location)
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
                datatype="float32",
            ),
        )
    except Exception:
        # Collection may already exist
        pass

    # Generate and store embeddings
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)

    # Store in Qdrant
    points = [
        {"id": h, "vector": e.tolist()} for h, e in zip(sentence_hashes, embeddings)
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

    # Benchmark direct embedding time
    start_time = time.time()
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        _ = model.encode(batch)
    embedding_time = time.time() - start_time

    # Benchmark cache retrieval time
    start_time = time.time()
    for i in range(0, len(sentence_hashes), batch_size):
        batch_hashes = sentence_hashes[i : i + batch_size]
        _ = qdrant_client.retrieve(
            collection_name=collection_name, ids=batch_hashes, with_vectors=True
        )
    cache_time = time.time() - start_time

    # Cleanup
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception:
        pass

    # Return True if cache is faster
    return cache_time < embedding_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark caching vs direct embedding for a model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name or path of the sentence transformer model",
    )
    parser.add_argument(
        "--benchmark-size",
        type=int,
        default=1000,
        help="Number of sentences to use for benchmarking",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="benchmark_cache",
        help="Name of temporary collection for benchmarking",
    )
    parser.add_argument(
        "--location", type=str, default="localhost", help="Location of Qdrant server"
    )

    args = parser.parse_args()

    result = should_use_cache(
        model_name=args.model_name,
        benchmark_size=args.benchmark_size,
        batch_size=args.batch_size,
        collection_name=args.collection_name,
        location=args.location,
    )

    if result:
        print(
            "Cache retrieval is faster than direct embedding - caching is recommended"
        )
    else:
        print(
            "Direct embedding is faster than cache retrieval - caching is not recommended"
        )
