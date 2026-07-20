"""
Provider de embeddings usando sentence-transformers.

Este módulo encapsula la lógica de generación de embeddings (vectores)
a partir de texto usando el modelo multilingual-e5-large.
"""

from sentence_transformers import SentenceTransformer


class EmbeddingProvider:
    """
    Genera embeddings (vectores) a partir de texto.
    
    Usa el modelo intfloat/multilingual-e5-large que:
    - Genera vectores de 1024 dimensiones
    - Soporta múltiples idiomas (español + inglés)
    - Optimizado para búsqueda semántica
    
    Explicación:
        - La primera vez que se carga el modelo se descarga (~2GB)
        - Luego queda en caché local (~/.cache/huggingface/)
        - El modelo corre en CPU por defecto, si tienes GPU CUDA lo usa automáticamente
    """

    def __init__(self, model_name: str):
        """
        Inicializa el provider cargando el modelo.
        
        Args:
            model_name: Nombre del modelo en Hugging Face Hub
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Convierte lista de textos en lista de vectores.
        
        Args:
            texts: Lista de strings a convertir
            
        Returns:
            Lista de vectores, cada uno con 1024 dimensiones
            
        Explicación:
            - convert_to_numpy=False retorna listas Python (más compatible con Qdrant)
            - normalize_embeddings=True hace que todos los vectores tengan magnitud 1
              (necesario para Distance.COSINE en Qdrant)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        """
        Convierte un solo texto en vector (útil para queries).
        
        Args:
            text: String a convertir
            
        Returns:
            Vector de 1024 dimensiones
        """
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        """Retorna la dimensión de los vectores que genera este modelo."""
        return self.model.get_sentence_embedding_dimension()
