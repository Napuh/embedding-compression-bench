import argparse
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import tiktoken
from mteb import MTEB, get_tasks
from sentence_transformers import SentenceTransformerModelCardData

from utils.config_utils import load_config


class QueryCounter:
    def __init__(self):
        self.query_counts = defaultdict(int)
        self.token_counts = defaultdict(int)
        self.current_benchmark = None
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.model_card_data = SentenceTransformerModelCardData()
        self.similarity_fn_name = "cosine"

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        """Count queries and tokens for current benchmark and return random embeddings."""
        self.query_counts[self.current_benchmark] += len(sentences)
        for sentence in sentences:
            self.token_counts[self.current_benchmark] += len(
                self.tokenizer.encode(sentence)
            )
        # Return random embeddings of shape (len(sentences), 64)
        return np.random.randn(len(sentences), 16)

    def set_benchmark(self, benchmark: str) -> None:
        """Set current benchmark being processed."""
        self.current_benchmark = benchmark


def count_queries(tasks: list[str], batch_size: int = 32) -> Dict[str, Dict[str, int]]:
    """Count queries and tokens for each benchmark task."""
    counter = QueryCounter()
    results = {}

    for benchmark in tasks:
        print(f"Counting queries for benchmark: {benchmark}")
        counter.set_benchmark(benchmark)

        mteb_tasks = get_tasks(tasks=[benchmark], languages=["eng"])
        evaluation = MTEB(tasks=mteb_tasks)

        # Run evaluation just to count queries
        evaluation.run(
            counter,
            output_folder=None,
            encode_kwargs={"batch_size": batch_size},
            verbosity=0,
        )

        results[benchmark] = {
            "queries": counter.query_counts[benchmark],
            "tokens": counter.token_counts[benchmark],
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_complete.yml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    results = count_queries(config["tasks"])

    # Create markdown table with results
    markdown = "# Benchmark Query Counts\n\n"
    markdown += "| Benchmark | Queries | Tokens |\n"
    markdown += "|-----------|----------|--------|\n"

    for benchmark, counts in results.items():
        markdown += f"| {benchmark} | {counts['queries']} | {counts['tokens']} |\n"

    with open("BENCHMARKS.md", "w") as f:
        f.write(markdown)

    print("Results saved to BENCHMARKS.md")
