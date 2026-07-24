"""Provider de embeddings densos con sentence-transformers."""

from typing import Literal

from sentence_transformers import SentenceTransformer


class EmbeddingProvider:
    """Genera embeddings con intfloat/multilingual-e5-large (1024 dims).
    La familia E5 requiere prefijos "passage: "/"query: " en el texto de
    entrada — sin ellos, la similitud coseno degrada notablemente."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.requires_prefix = "e5" in model_name.lower()

    def _add_prefix(self, text: str, mode: Literal["query", "passage"]) -> str:
        if not self.requires_prefix:
            return text

        prefix = "query: " if mode == "query" else "passage: "
        if text.startswith(prefix):
            return text
        return f"{prefix}{text}"

    def encode(
        self,
        texts: list[str],
        mode: Literal["query", "passage"] = "passage",
    ) -> list[list[float]]:
        prefixed_texts = [self._add_prefix(text, mode) for text in texts]

        # normalize_embeddings=True: requerido para Distance.COSINE en Qdrant.
        embeddings = self.model.encode(
            prefixed_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def encode_single(
        self,
        text: str,
        mode: Literal["query", "passage"] = "passage",
    ) -> list[float]:
        return self.encode([text], mode=mode)[0]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
