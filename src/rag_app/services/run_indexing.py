"""
Script CLI para indexación de documentos en Qdrant.

Orquesta el proceso de generación de embeddings e indexación vectorial
de documentos que ya tienen chunks en cache.

Pipeline:
    chunks.json → EmbeddingProvider → VectorRepository → Qdrant

Usage:
    # Indexar documento específico
    python src/rag_app/services/run_indexing.py paper
    
    # Indexar todos los documentos en cache
    python src/rag_app/services/run_indexing.py --all
    
    # Re-indexar (elimina e indexa nuevamente)
    python src/rag_app/services/run_indexing.py paper --reindex
    
    # Buscar (testing)
    python src/rag_app/services/run_indexing.py --search "exosomes wound healing"
    
    # Ver estadísticas
    python src/rag_app/services/run_indexing.py --stats
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_app.core.config import get_settings
from rag_app.providers.embeddings import EmbeddingProvider
from rag_app.providers.sparse_embeddings import SparseEmbeddingProvider
from rag_app.providers.qdrant_client import create_qdrant_client
from rag_app.repositories.vector_repository import VectorRepository
from rag_app.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    """Configura logging del script."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )
    
    if not verbose:
        # Silenciar librerías verbosas
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def create_services() -> tuple[IndexingService, VectorRepository]:
    """
    Inicializa todos los servicios necesarios.
    
    Returns:
        Tupla de (IndexingService, VectorRepository)
    """
    settings = get_settings()
    
    # Embedding providers (dense + sparse)
    embedding_provider = EmbeddingProvider(
        model_name=settings.embedding_model_name
    )
    sparse_embedding_provider = SparseEmbeddingProvider()

    # Qdrant client
    qdrant_client = create_qdrant_client(settings)

    # Vector repository
    vector_repository = VectorRepository(
        client=qdrant_client,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.embedding_dimension,
    )

    # Asegurar que la colección existe
    vector_repository.ensure_collection_exists()

    # Indexing service
    indexing_service = IndexingService(
        embedding_provider=embedding_provider,
        sparse_embedding_provider=sparse_embedding_provider,
        vector_repository=vector_repository,
        batch_size=32,  # Batch size óptimo para e5-large
    )
    
    return indexing_service, vector_repository


def index_document(
    document_name: str,
    cache_dir: Path,
    reindex: bool = False,
) -> int:
    """
    Indexa un documento específico.
    
    Args:
        document_name: Nombre del documento (sin extensión)
        cache_dir: Directorio base de cache
        reindex: Si True, elimina e indexa nuevamente
    
    Returns:
        0 si éxito, 1 si error
    """
    chunks_json = cache_dir / document_name / "chunks" / "chunks.json"
    
    if not chunks_json.exists():
        logger.error(f"✗ No se encontró chunks.json para '{document_name}'")
        logger.error(f"  Ruta esperada: {chunks_json}")
        logger.error(f"  Ejecuta primero: python src/rag_app/services/run_chunking.py")
        return 1
    
    logger.info("=" * 80)
    logger.info(f"{'Re-indexando' if reindex else 'Indexando'}: {document_name}")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        start_time = datetime.now()
        
        # Crear servicios
        indexing_service, _ = create_services()
        
        # Indexar o re-indexar
        if reindex:
            result = indexing_service.reindex_document(
                chunks_json_path=chunks_json,
                document_name=document_name,
            )
        else:
            result = indexing_service.index_document(
                chunks_json_path=chunks_json,
                document_name=document_name,
            )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Mostrar resultados
        if "error" in result:
            logger.error(f"✗ Error: {result['error']}")
            return 1
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ Indexación completada")
        logger.info("=" * 80)
        logger.info(f"  Documento:        {result['document_name']}")
        logger.info(f"  Chunks procesados: {result['chunks_processed']}")
        logger.info(f"  Chunks indexados:  {result['chunks_indexed']}")
        logger.info(f"  Batches:           {result['batches']}")
        logger.info(f"  Tiempo:            {elapsed:.2f}s")
        logger.info("")
        
        return 0
    
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        if logger.level == logging.DEBUG:
            import traceback
            traceback.print_exc()
        return 1


def index_all(cache_dir: Path) -> int:
    """
    Indexa todos los documentos en cache.
    
    Args:
        cache_dir: Directorio base de cache
    
    Returns:
        0 si éxito, 1 si error
    """
    logger.info("=" * 80)
    logger.info("Indexando todos los documentos en cache")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        start_time = datetime.now()
        
        # Crear servicios
        indexing_service, _ = create_services()
        
        # Indexar todos
        results = indexing_service.index_all_cached_documents(cache_dir)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if not results:
            logger.info("No se encontraron documentos con chunks en cache.")
            logger.info(f"Cache dir: {cache_dir}")
            logger.info("")
            logger.info("Ejecuta primero:")
            logger.info("  python src/rag_app/services/run_chunking.py data/uploads/paper.pdf")
            return 0
        
        # Mostrar resumen
        success_count = sum(1 for r in results if "error" not in r)
        error_count = len(results) - success_count
        total_chunks = sum(r.get("chunks_indexed", 0) for r in results if "error" not in r)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("Resumen de indexación")
        logger.info("=" * 80)
        
        for result in results:
            if "error" in result:
                logger.info(f"  ✗ {result['document_name']}: {result['error']}")
            else:
                logger.info(
                    f"  ✓ {result['document_name']}: "
                    f"{result['chunks_indexed']} chunks"
                )
        
        logger.info("")
        logger.info(f"Total documentos: {len(results)}")
        logger.info(f"  Exitosos:       {success_count}")
        logger.info(f"  Con errores:    {error_count}")
        logger.info(f"Total chunks:     {total_chunks}")
        logger.info(f"Tiempo total:     {elapsed:.2f}s")
        logger.info("")
        
        return 0 if error_count == 0 else 1
    
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        if logger.level == logging.DEBUG:
            import traceback
            traceback.print_exc()
        return 1


def search(
    query: str,
    top_k: int = 5,
    page_filter: Optional[int] = None,
    heading_contains: Optional[str] = None,
) -> int:
    """
    Búsqueda de prueba (híbrida: semántica + léxica con fusión RRF).

    Args:
        query: Consulta del usuario
        top_k: Número de resultados
        page_filter: Opcional, restringe a una página específica
        heading_contains: Opcional, restringe por texto libre en headings

    Returns:
        0 si éxito, 1 si error
    """
    logger.info("=" * 80)
    logger.info(f"Búsqueda: '{query}'")
    logger.info("=" * 80)
    logger.info("")

    try:
        # Crear servicios
        indexing_service, _ = create_services()

        # Buscar
        results = indexing_service.search(
            query=query,
            top_k=top_k,
            page_filter=page_filter,
            heading_contains=heading_contains,
        )
        
        if not results:
            logger.info("No se encontraron resultados.")
            logger.info("Verifica que hay documentos indexados:")
            logger.info("  python src/rag_app/services/run_indexing.py --stats")
            return 0
        
        # Mostrar resultados
        logger.info(f"Encontrados {len(results)} resultados:\n")
        
        for i, result in enumerate(results, 1):
            logger.info(f"--- Resultado {i} ---")
            logger.info(f"Score:     {result['score']:.4f}")
            logger.info(f"Documento: {result['document_name']}")
            logger.info(f"Chunk:     {result['chunk_id']}")
            
            if result['headings']:
                logger.info(f"Headings:  {' → '.join(result['headings'])}")
            
            if result['pages']:
                logger.info(f"Páginas:   {', '.join(map(str, result['pages']))}")
            
            if result['image_paths']:
                logger.info(f"Imágenes:  {', '.join(result['image_paths'])}")
            
            logger.info(f"\nContenido:\n{result['content'][:300]}...")
            logger.info("")
        
        return 0
    
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        if logger.level == logging.DEBUG:
            import traceback
            traceback.print_exc()
        return 1


def show_stats() -> int:
    """
    Muestra estadísticas de la colección.
    
    Returns:
        0 si éxito, 1 si error
    """
    logger.info("=" * 80)
    logger.info("Estadísticas de Qdrant")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Crear servicios
        indexing_service, _ = create_services()
        
        # Stats de colección
        stats = indexing_service.get_stats()
        
        logger.info("Colección:")
        logger.info(f"  Nombre:          {stats['collection_name']}")
        logger.info(f"  Puntos:          {stats['points_count']}")
        logger.info(f"  Estado:          {stats['status']}")
        logger.info("")
        
        # Lista de documentos
        documents = indexing_service.list_indexed_documents()
        
        if documents:
            logger.info(f"Documentos indexados ({len(documents)}):")
            for doc in documents:
                logger.info(f"  - {doc}")
        else:
            logger.info("No hay documentos indexados todavía.")
            logger.info("")
            logger.info("Para indexar:")
            logger.info("  python src/rag_app/services/run_indexing.py --all")
        
        logger.info("")
        
        return 0
    
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        if logger.level == logging.DEBUG:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Entry point del script."""
    parser = argparse.ArgumentParser(
        description="Indexar documentos en Qdrant",
        epilog="""
Ejemplos:
  python src/rag_app/services/run_indexing.py paper
  python src/rag_app/services/run_indexing.py --all
  python src/rag_app/services/run_indexing.py paper --reindex
  python src/rag_app/services/run_indexing.py --search "exosomes wound healing"
  python src/rag_app/services/run_indexing.py --stats
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "document_name",
        nargs="?",
        help="Nombre del documento a indexar (sin extensión)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Indexar todos los documentos en cache",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Re-indexar (elimina e indexa nuevamente)",
    )
    parser.add_argument(
        "--search",
        type=str,
        metavar="QUERY",
        help="Buscar documentos (testing)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Número de resultados para búsqueda (default: 5)",
    )
    parser.add_argument(
        "--page-filter",
        type=int,
        default=None,
        help="Restringir búsqueda a una página específica",
    )
    parser.add_argument(
        "--heading-contains",
        type=str,
        default=None,
        help="Restringir búsqueda por texto libre en headings",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostrar estadísticas de la colección",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Directorio de cache (default: data/cache)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        # Determinar acción
        if args.stats:
            return show_stats()
        
        elif args.search:
            return search(
                args.search,
                args.top_k,
                page_filter=args.page_filter,
                heading_contains=args.heading_contains,
            )
        
        elif args.all:
            if args.document_name:
                logger.warning("--all especificado, ignorando nombre de documento\n")
            return index_all(args.cache_dir)
        
        elif args.document_name:
            return index_document(
                args.document_name,
                args.cache_dir,
                args.reindex,
            )
        
        else:
            parser.error(
                "Especifica un documento, --all, --search, o --stats"
            )
    
    except KeyboardInterrupt:
        logger.info("\n\nInterrumpido por usuario")
        return 130
    
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
