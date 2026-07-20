"""Script de prueba para embeddings provider."""

from rag_app.core.config import get_settings
from rag_app.providers.embeddings import EmbeddingProvider


def main():
    settings = get_settings()
    print(f"Cargando modelo: {settings.embedding_model_name}")
    print("(Primera vez descarga ~2GB, puede tardar...)")
    
    provider = EmbeddingProvider(settings.embedding_model_name)
    
    print(f"\n✓ Modelo cargado")
    print(f"✓ Dimensión: {provider.dimension}")
    
    # Probar con texto de ejemplo
    test_texts = [
        "Los exosomas son vesículas extracelulares pequeñas.",
        "Exosomes are small extracellular vesicles.",
    ]
    
    embeddings = provider.encode(test_texts)
    
    print(f"\n✓ Embeddings generados:")
    print(f"  - Cantidad: {len(embeddings)}")
    print(f"  - Dimensión de cada uno: {len(embeddings[0])}")
    print(f"  - Primeros 5 valores del vector 1: {embeddings[0][:5]}")
    print("\n¡Todo funciona! 🎉")


if __name__ == "__main__":
    main()
