from typing import Any

import numpy as np
import tiktoken


class DummyEmbeddingEngine:
    def __init__(self):
        self.query_count = 0
        self.token_count = 0
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        """Returns random numpy embeddings of dimension 384 for the given sentences.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments (ignored).

        Returns:
            Random numpy embeddings of shape (len(sentences), 384).
        """
        self.query_count += len(sentences)
        for sentence in sentences:
            self.token_count += len(self.tokenizer.encode(sentence))
        return np.random.rand(len(sentences), 384).astype(np.float32)
