import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, type ChatEvent, type ChatRequest, type Citation } from "../lib/api";
import { parseSSEStream } from "../lib/sse";

export type MessageStatus = "streaming" | "done" | "no_results" | "error";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  status?: MessageStatus;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export interface UseConversationsResult {
  conversations: Conversation[];
  activeConversationId: string | null;
  messages: Message[];
  isStreaming: boolean;
  createConversation: () => void;
  switchConversation: (id: string) => void;
  ask: (request: Omit<ChatRequest, "query"> & { query: string }) => Promise<void>;
}

const CONVERSATIONS_KEY = "research-assistant:conversations";
const ACTIVE_ID_KEY = "research-assistant:active-conversation-id";
const TITLE_MAX_LENGTH = 48;

let nextId = 0;
const newId = (prefix: string) => `${prefix}_${Date.now()}_${nextId++}`;

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(CONVERSATIONS_KEY);
    return raw ? (JSON.parse(raw) as Conversation[]) : [];
  } catch {
    return [];
  }
}

function makeTitle(query: string): string {
  const trimmed = query.trim();
  return trimmed.length > TITLE_MAX_LENGTH ? `${trimmed.slice(0, TITLE_MAX_LENGTH)}…` : trimmed;
}

function makeConversation(title: string): Conversation {
  const now = Date.now();
  return { id: newId("conv"), title, messages: [], createdAt: now, updatedAt: now };
}

/**
 * Persiste múltiples conversaciones en localStorage (a diferencia de
 * useChatSession, que solo mantenía una en memoria y la perdía al
 * recargar). El backend es stateless por diseño — cada /chat es
 * independiente — así que toda la persistencia vive acá, en el cliente.
 */
export function useConversations(): UseConversationsResult {
  const [conversations, setConversations] = useState<Conversation[]>(() => loadConversations());
  const [activeConversationId, setActiveConversationId] = useState<string | null>(() => {
    const stored = localStorage.getItem(ACTIVE_ID_KEY);
    const loaded = loadConversations();
    if (stored && loaded.some((c) => c.id === stored)) return stored;
    return loaded[0]?.id ?? null;
  });
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations));
  }, [conversations]);

  useEffect(() => {
    if (activeConversationId) localStorage.setItem(ACTIVE_ID_KEY, activeConversationId);
  }, [activeConversationId]);

  const updateConversation = useCallback(
    (conversationId: string, updater: (conversation: Conversation) => Conversation) => {
      setConversations((prev) =>
        prev.map((c) => (c.id === conversationId ? updater(c) : c))
      );
    },
    []
  );

  const updateAssistantMessage = useCallback(
    (conversationId: string, messageId: string, patch: Partial<Message>) => {
      updateConversation(conversationId, (c) => ({
        ...c,
        messages: c.messages.map((m) => (m.id === messageId ? { ...m, ...patch } : m)),
        updatedAt: Date.now(),
      }));
    },
    [updateConversation]
  );

  const createConversation = useCallback(() => {
    const conversation = makeConversation("Nueva conversación");
    setConversations((prev) => [conversation, ...prev]);
    setActiveConversationId(conversation.id);
  }, []);

  const switchConversation = useCallback((id: string) => {
    abortRef.current?.abort();
    setActiveConversationId(id);
  }, []);

  const ask = useCallback(
    async (request: ChatRequest) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      // Si todavía no hay conversación activa (primer uso del todo), se
      // crea sobre la marcha en vez de obligar a apretar "+ Nueva" antes
      // de poder preguntar algo.
      let conversationId = activeConversationId;
      if (conversationId === null) {
        const conversation = makeConversation(makeTitle(request.query));
        conversationId = conversation.id;
        setConversations((prev) => [conversation, ...prev]);
        setActiveConversationId(conversationId);
      }

      const userMessage: Message = { id: newId("msg"), role: "user", content: request.query };
      const assistantId = newId("msg");
      const assistantMessage: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        status: "streaming",
      };

      updateConversation(conversationId, (c) => ({
        ...c,
        title: c.messages.length === 0 ? makeTitle(request.query) : c.title,
        messages: [...c.messages, userMessage, assistantMessage],
        updatedAt: Date.now(),
      }));
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

        await parseSSEStream<ChatEvent>(res.body, (event) => {
          if (event.type === "chunk") {
            updateConversation(conversationId, (c) => ({
              ...c,
              messages: c.messages.map((m) =>
                m.id === assistantId ? { ...m, content: m.content + event.text } : m
              ),
              updatedAt: Date.now(),
            }));
          } else if (event.type === "done") {
            updateAssistantMessage(conversationId, assistantId, {
              citations: event.citations,
              status: "done",
            });
          } else if (event.type === "no_results") {
            updateAssistantMessage(conversationId, assistantId, { status: "no_results" });
          } else if (event.type === "error") {
            updateAssistantMessage(conversationId, assistantId, {
              status: "error",
              error: event.error,
            });
          }
        });
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        updateAssistantMessage(conversationId, assistantId, {
          status: "error",
          error: err instanceof Error ? err.message : String(err),
        });
      } finally {
        setIsStreaming(false);
      }
    },
    [activeConversationId, updateConversation, updateAssistantMessage]
  );

  const activeConversation = conversations.find((c) => c.id === activeConversationId) ?? null;

  return {
    conversations,
    activeConversationId,
    messages: activeConversation?.messages ?? [],
    isStreaming,
    createConversation,
    switchConversation,
    ask,
  };
}
