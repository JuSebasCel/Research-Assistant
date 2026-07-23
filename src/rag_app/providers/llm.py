"""
Provider de LLM usando Gemini (google-genai).

Gemini 2.5 Flash: tier gratuito real (sin tarjeta, ~1500 req/día vía Google
AI Studio), a diferencia de OpenAI (solo crédito único que expira).
"""

from collections.abc import Iterator

from google import genai
from google.genai import types


class GeminiProvider:
    """
    Genera respuestas de texto en streaming a partir de un system prompt y
    un prompt de usuario.

    Sin manejo de historial de conversación: cada llamada es independiente,
    coherente con que el retrieval reconstruye el contexto relevante en
    cada pregunta (multi-turno queda fuera de alcance por ahora).
    """

    def __init__(self, api_key: str, model_name: str = "gemini-flash-latest"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_stream(
        self,
        system_instruction: str,
        user_prompt: str,
        images: list[tuple[bytes, str]] | None = None,
    ) -> Iterator[str]:
        """
        Genera la respuesta en chunks de texto según van llegando.

        Args:
            system_instruction: Instrucciones de comportamiento del modelo
            user_prompt: Contexto recuperado + pregunta del usuario
            images: Lista opcional de (bytes, mime_type) para las figuras
                asociadas a los chunks recuperados (RAG multimodal)

        Yields:
            Fragmentos de texto de la respuesta
        """
        config = types.GenerateContentConfig(system_instruction=system_instruction)

        contents = [user_prompt]
        for image_bytes, mime_type in images or []:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        response = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text
