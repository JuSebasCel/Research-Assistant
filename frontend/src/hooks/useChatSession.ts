import { useCallback, useRef, useState } from "react";
import { API_BASE, type ChatEvent, type ChatRequest, type Citation } from "../lib/api";

export type MessageStatus = "streaming" | "done" | "no_results" | "error";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  status?: MessageStatus;
  error?: string;
}

export interface UseChatSessionResult {
  messages: Message[];
  isStreaming: boolean;
  ask: (request: Omit<ChatRequest, "query"> & { query: string }) => Promise<void>;
}

let nextId = 0;
const newId = () => `msg_${Date.now()}_${nextId++}`;

/**
 * Mantiene el hilo completo de la conversación (a diferencia de un hook que
 * solo guarda la última respuesta). Cada `ask()` agrega el mensaje del
 * usuario y un mensaje de asistente "placeholder" que se va llenando en
 * vivo a medida que llegan los chunks del streaming.
 *
 * EventSource nativo no sirve para POST /chat (solo soporta GET), así que
 * el parseo de SSE se hace a mano sobre un fetch() + ReadableStream.
 */
export function useChatSession(): UseChatSessionResult {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const updateAssistantMessage = useCallback((id: string, patch: Partial<Message>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }, []);

  const ask = useCallback(
    async (request: ChatRequest) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const userMessage: Message = { id: newId(), role: "user", content: request.query };
      const assistantId = newId();
      const assistantMessage: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        status: "streaming",
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsStreaming(true);

      try {
        const res = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(request),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          throw new Error(`El servidor respondió ${res.status}`);
        }

        const reader = res.body.getReader();
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

            const event = JSON.parse(payload) as ChatEvent;

            if (event.type === "chunk") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, content: m.content + event.text } : m
                )
              );
            } else if (event.type === "done") {
              updateAssistantMessage(assistantId, { citations: event.citations, status: "done" });
            } else if (event.type === "no_results") {
              updateAssistantMessage(assistantId, { status: "no_results" });
            }
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        updateAssistantMessage(assistantId, {
          status: "error",
          error: err instanceof Error ? err.message : String(err),
        });
      } finally {
        setIsStreaming(false);
      }
    },
    [updateAssistantMessage]
  );

  return { messages, isStreaming, ask };
}
