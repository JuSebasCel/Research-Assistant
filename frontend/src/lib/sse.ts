// EventSource nativo no soporta POST, así que el parseo de SSE se hace a
// mano sobre un fetch() + ReadableStream. Compartido entre chat e ingesta.
export async function parseSSEStream<T>(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: T) => void
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const line = chunk.trim();
      if (!line.startsWith("data:")) continue;

      const payload = line.slice("data:".length).trim();
      if (!payload) continue;

      onEvent(JSON.parse(payload) as T);
    }
  }
}
