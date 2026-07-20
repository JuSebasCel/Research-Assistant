"""
Configuración centralizada de logging.

Por qué no usamos `print()`: print no tiene niveles (info/warning/error),
no se puede filtrar ni redirigir fácilmente, y no incluye contexto como
timestamp, módulo de origen, o nombre del logger.
"""

import logging
import sys

from rag_app.core.config import get_settings


def configure_logging() -> None:
    """
    Configura el logging raíz de la aplicación.

    Se llama una única vez, al arrancar la app (en el lifespan de FastAPI,
    que vamos a ver en el próximo paso). Cada módulo luego obtiene su
    propio logger con `logging.getLogger(__name__)`, que hereda esta
    configuración pero identifica de qué archivo viene cada mensaje.
    """
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Librerías de terceros suelen ser muy verbosas en nivel INFO/DEBUG;
    # las bajamos a WARNING para no ensuciar nuestros propios logs.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)