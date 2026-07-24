"""Configuración centralizada de logging."""

import logging
import sys

from rag_app.core.config import get_settings


def configure_logging() -> None:
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