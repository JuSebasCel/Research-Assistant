import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ArrowUp, Check, Copy, Sparkles } from "lucide-react";
import type { Message } from "../hooks/useConversations";
import { Markdown } from "./Markdown";
import { MessageSources } from "./MessageSources";

interface ChatViewProps {
  scopeLabel: string | null;
  messages: Message[];
  isStreaming: boolean;
  onAsk: (query: string) => void;
}

export function ChatView({ scopeLabel, messages, isStreaming, onAsk }: ChatViewProps) {
  const [query, setQuery] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || isStreaming) return;
    onAsk(trimmed);
    setQuery("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event as unknown as FormEvent);
    }
  };

  const handleInput = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuery(event.target.value);
    const el = event.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  return (
    <main className="flex min-w-0 flex-1 flex-col">
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-8 py-8">
        <div className="mx-auto flex max-w-2xl flex-col gap-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center gap-3 py-24 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/10">
                <Sparkles size={22} className="text-accent" />
              </div>
              <h2 className="text-[22px] font-semibold text-ink dark:text-ink-dark">
                Pregunta sobre tus papers
              </h2>
              <p className="max-w-sm text-[14px] text-ink-secondary dark:text-ink-secondary-dark">
                {scopeLabel
                  ? `Buscando solo en ${scopeLabel}.`
                  : "Buscando en todos los documentos indexados."}
              </p>
            </div>
          )}

          {messages.map((message) =>
            message.role === "user" ? (
              <UserBubble key={message.id} text={message.content} />
            ) : (
              <AssistantMessage key={message.id} message={message} />
            )
          )}
        </div>
      </div>

      <div className="border-t border-black/5 px-8 py-5 dark:border-white/10">
        <form onSubmit={handleSubmit} className="mx-auto flex max-w-2xl items-end gap-2">
          <textarea
            ref={textareaRef}
            value={query}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            rows={1}
            placeholder="Pregunta lo que sea sobre tus papers..."
            disabled={isStreaming}
            className="max-h-[200px] min-h-12 flex-1 resize-none rounded-3xl border border-black/5 bg-surface px-5 py-3 text-[15px] text-ink placeholder:text-ink-secondary focus:border-accent/40 focus:ring-4 focus:ring-accent/10 focus:outline-none disabled:opacity-60 dark:border-white/10 dark:bg-surface-dark dark:text-ink-dark dark:placeholder:text-ink-secondary-dark"
          />
          <button
            type="submit"
            disabled={isStreaming || !query.trim()}
            className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-accent text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            <ArrowUp size={18} />
          </button>
        </form>
      </div>
    </main>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-bubble bg-user-bubble px-4 py-2.5 text-[15px] leading-relaxed text-ink dark:bg-user-bubble-dark dark:text-ink-dark">
        {text}
      </div>
    </div>
  );
}

function AssistantMessage({ message }: { message: Message }) {
  const [copied, setCopied] = useState(false);
  const waitingForFirstToken = message.status === "streaming" && message.content.length === 0;

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="group">
      {message.status === "no_results" ? (
        <p className="text-[15px] leading-relaxed text-ink-secondary italic dark:text-ink-secondary-dark">
          No encontré contenido suficientemente relevante en los documentos disponibles para esta
          pregunta.
        </p>
      ) : message.status === "error" ? (
        <p className="text-[15px] leading-relaxed text-red-500">
          {message.error ?? "Ocurrió un error inesperado."}
        </p>
      ) : waitingForFirstToken ? (
        <ThinkingIndicator />
      ) : (
        <>
          <Markdown>{message.content}</Markdown>
          {message.status === "streaming" && (
            <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-accent align-middle" />
          )}
        </>
      )}

      {message.status === "done" && message.content && (
        <button
          onClick={handleCopy}
          className="mt-2 flex items-center gap-1.5 text-[12px] text-ink-secondary opacity-0 transition-opacity group-hover:opacity-100 hover:text-ink dark:text-ink-secondary-dark dark:hover:text-ink-dark"
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? "Copiado" : "Copiar"}
        </button>
      )}

      {message.status === "done" && message.citations && (
        <MessageSources citations={message.citations} />
      )}
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-1.5 py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-secondary/50 dark:bg-ink-secondary-dark/50"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
}
