"""
Provider de embeddings sparse (BM25) usando fastembed.

Genera vectores sparse para la señal léxica del retrieval híbrido.
Complementa a `EmbeddingProvider` (denso, semántico): juntos alimentan
la fusión RRF nativa de Qdrant.
"""

from typing import Literal

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector


class SparseEmbeddingProvider:
    """
    Genera vectores sparse BM25 a partir de texto usando fastembed.

    A diferencia de e5 (que usa prefijos "query:"/"passage:"), el modelo
    Qdrant/bm25 distingue documento vs consulta mediante dos métodos
    distintos:
    - embed() (documentos): aplica ponderación por término, pensada para
      combinarse con Modifier.IDF del lado de Qdrant (que calcula el IDF
      real sobre el corpus indexado al momento de la búsqueda).
    - query_embed() (consultas): solo presencia/conteo de términos, sin
      ponderación adicional.
    """

    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model = SparseTextEmbedding(model_name)
        self.model_name = model_name

    def encode(
        self,
        texts: list[str],
        mode: Literal["query", "passage"] = "passage",
    ) -> list[SparseVector]:
        """
        Convierte una lista de textos en una lista de vectores sparse.

        Args:
            texts: Lista de strings a convertir
            mode: "passage" para chunks de documentos (default),
                  "query" para consultas del usuario

        Returns:
            Lista de SparseVector (indices + values) de qdrant_client
        """
        embed_fn = self.model.query_embed if mode == "query" else self.model.embed
        sparse_embeddings = list(embed_fn(texts))

        return [
            SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            )
            for embedding in sparse_embeddings
        ]

    def encode_single(
        self,
        text: str,
        mode: Literal["query", "passage"] = "passage",
    ) -> SparseVector:
        """Convierte un solo texto en vector sparse."""
        return self.encode([text], mode=mode)[0]
