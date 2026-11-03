"""In-memory example retrieval: MiniLM sentence embeddings + numpy cosine similarity."""

from functools import lru_cache

import numpy as np
from langfuse import get_client, observe
from sentence_transformers import SentenceTransformer

from messaging_bot.config import EMBEDDING_MODEL_NAME
from messaging_bot.models import TestCase
from messaging_bot.records import retrieval_text


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load the local embedding model once per process."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, so a dot product equals cosine similarity."""
    embedding_model = load_embedding_model()
    return embedding_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


class ExampleIndex:
    """Holds every labelled record and finds the ones most similar to a query record."""

    def __init__(self, records: list[TestCase]):
        # Only records with an expected outcome are useful as worked examples.
        self.examples = [record for record in records if record.expected is not None]
        if self.examples:
            self.embeddings = embed_texts([retrieval_text(record) for record in self.examples])
        else:
            self.embeddings = np.empty((0, 0))

    @observe(name="retrieve", capture_input=False, capture_output=False)
    def top_k(self, record: TestCase, k: int, exclude_task_id: str | None = None) -> list[tuple[TestCase, float]]:
        """Return up to k (example, cosine similarity) pairs, most similar first.

        `exclude_task_id` removes the query record itself from the candidates, which is
        what makes leave-one-out evaluation honest.
        """
        if not self.examples:
            return []

        query_embedding = embed_texts([retrieval_text(record)])[0]
        similarities = self.embeddings @ query_embedding
        ranked_indices = np.argsort(-similarities)

        results: list[tuple[TestCase, float]] = []
        for example_index in ranked_indices:
            example = self.examples[example_index]
            if example.task_id == exclude_task_id:
                continue
            results.append((example, float(similarities[example_index])))
            if len(results) == k:
                break

        get_client().update_current_span(
            input={"query_task_id": record.task_id, "k": k, "excluded_task_id": exclude_task_id},
            output=[{"task_id": example.task_id, "similarity": round(similarity, 4)} for example, similarity in results],
        )
        return results
