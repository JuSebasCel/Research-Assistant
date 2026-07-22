"""
Provider de embeddings usando sentence-transformers.

Este módulo encapsula la lógica de generación de embeddings (vectores)
a partir de texto usando el modelo multilingual-e5-large.
"""

from typing import Literal

from sentence_transformers import SentenceTransformer


class EmbeddingProvider:
    """
    Genera embeddings (vectores) a partir de texto.
    
    Usa el modelo intfloat/multilingual-e5-large que:
    - Genera vectores de 1024 dimensiones
    - Soporta múltiples idiomas (español + inglés)
    - Optimizado para búsqueda semántica
    - REQUIERE prefijo "passage: " o "query: " según el modelo E5
    
    Explicación:
        - La primera vez que se carga el modelo se descarga (~2GB)
        - Luego queda en caché local (~/.cache/huggingface/)
        - El modelo corre en CPU por defecto, si tienes GPU CUDA lo usa automáticamente
        - CRÍTICO: La familia E5 fue entrenada con prefijos obligatorios
          ("passage: " para documentos, "query: " para consultas)
    """

    def __init__(self, model_name: str):
        """
        Inicializa el provider cargando el modelo.
        
        Args:
            model_name: Nombre del modelo en Hugging Face Hub
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Detectar si es modelo E5 que requiere prefijos
        self.requires_prefix = "e5" in model_name.lower()

    def _add_prefix(self, text: str, mode: Literal["query", "passage"]) -> str:
        """
        Añade el prefijo apropiado según el modo (solo para modelos E5).
        
        Args:
            text: Texto a prefijar
            mode: "query" para consultas del usuario, "passage" para documentos
            
        Returns:
            Texto con prefijo si el modelo lo requiere
            
        Explicación:
            Los modelos E5 fueron entrenados con estos prefijos como parte
            del contrastive learning. Sin ellos, el espacio de embeddings
            opera fuera de su distribución de entrenamiento, degradando
            significativamente la calidad de similitud coseno.
        """
        if not self.requires_prefix:
            return text
        
        prefix = "query: " if mode == "query" else "passage: "
        
        # Evitar añadir prefijo duplicado
        if text.startswith(prefix):
            return text
        
        return f"{prefix}{text}"

    def encode(
        self,
        texts: list[str],
        mode: Literal["query", "passage"] = "passage",
    ) -> list[list[float]]:
        """
        Convierte lista de textos en lista de vectores.
        
        Args:
            texts: Lista de strings a convertir
            mode: "passage" para chunks de documentos (default),
                  "query" para consultas del usuario
            
        Returns:
            Lista de vectores, cada uno con 1024 dimensiones
            
        Explicación:
            - Añade prefijo E5 automáticamente según el modo
            - convert_to_numpy=True para compatibilidad
            - normalize_embeddings=True para Distance.COSINE en Qdrant
        """
        # Aplicar prefijo según el modo
        prefixed_texts = [self._add_prefix(text, mode) for text in texts]
        
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
        """
        Convierte un solo texto en vector.
        
        Args:
            text: String a convertir
            mode: "passage" para documentos, "query" para consultas
            
        Returns:
            Vector de 1024 dimensiones
        """
        return self.encode([text], mode=mode)[0]

    @property
    def dimension(self) -> int:
        """Retorna la dimensión de los vectores que genera este modelo."""
        return self.model.get_sentence_embedding_dimension()
