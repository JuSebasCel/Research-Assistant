"""Provider de LLM usando Gemini (google-genai)."""

from collections.abc import Iterator

from google import genai
from google.genai import types


class GeminiProvider:
    """Genera respuestas en streaming. Sin historial de conversación: cada
    llamada es independiente, el retrieval reconstruye el contexto por
    pregunta."""

    def __init__(self, api_key: str, model_name: str = "gemini-flash-latest"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def set_api_key(self, api_key: str) -> None:
        self.client = genai.Client(api_key=api_key)

    def generate_stream(
        self,
        system_instruction: str,
        user_prompt: str,
        images: list[tuple[bytes, str]] | None = None,
    ) -> Iterator[str]:
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
