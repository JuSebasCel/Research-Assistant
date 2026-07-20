"""
Script de verificación manual del logging.

Objetivo: confirmar que
1. configure_logging() aplica el formato esperado.
2. Cada logger identifica correctamente su módulo de origen.
3. LOG_LEVEL desde el .env realmente filtra los mensajes.

DEBUG < INFO < WARNING < ERROR < CRITICAL
"""

import logging

from rag_app.core.logging_config import configure_logging

configure_logging()

# Cada módulo debería crear su propio logger así, usando __name__.
logger = logging.getLogger("scripts.check_logging")

print("--- Probando los distintos niveles ---")
logger.debug("Este es un mensaje DEBUG (no debería verse si LOG_LEVEL=INFO)")
logger.info("Este es un mensaje INFO")
logger.warning("Este es un mensaje WARNING")
logger.error("Este es un mensaje ERROR")

print("\n--- Simulando un logger de otro módulo ---")
other_logger = logging.getLogger("rag_app.services.fake_service")
other_logger.info("Mensaje desde un servicio simulado")