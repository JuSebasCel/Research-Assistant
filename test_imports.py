"""
Script simple para verificar que todos los imports funcionan correctamente.
"""

print("Verificando imports...")

try:
    print("  - Config...")
    from rag_app.core.config import get_settings
    
    print("  - Embeddings provider...")
    from rag_app.providers.embeddings import EmbeddingProvider
    
    print("  - Qdrant client...")
    from rag_app.providers.qdrant_client import create_qdrant_client
    
    print("  - Vector repository...")
    from rag_app.repositories.vector_repository import VectorRepository
    
    print("  - Indexing service...")
    from rag_app.services.indexing_service import IndexingService
    
    print("\n✓ Todos los imports exitosos!")
    
except Exception as e:
    print(f"\n✗ Error en imports: {e}")
    import traceback
    traceback.print_exc()
