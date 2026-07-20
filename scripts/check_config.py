"""
Script de verificación manual: confirma que Settings carga correctamente
desde el .env y que get_settings() efectivamente cachea la instancia.

No es un test automatizado (eso vendrá con pytest más adelante) — es una
forma rápida de "tocar" el objeto Settings con los ojos, como quien
hace un print de depuración pero de forma más estructurada.
"""

from rag_app.core.config import get_settings

settings_1 = get_settings()
settings_2 = get_settings()

print("=== Valores cargados ===")
print(f"environment:  {settings_1.environment}")
print(f"log_level:    {settings_1.log_level}")
print(f"database_url: {settings_1.database_url}")

print("\n=== Verificando el cacheo de lru_cache ===")
print(f"¿settings_1 es la misma instancia que settings_2? {settings_1 is settings_2}")