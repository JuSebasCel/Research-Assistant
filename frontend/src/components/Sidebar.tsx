import { FileText, Library, Loader2, Plus, Upload } from "lucide-react";
import { useRef, type ChangeEvent } from "react";
import type { Conversation } from "../hooks/useConversations";
import { useIngest } from "../hooks/useIngest";
import { formatRelativeTime } from "../lib/format";

interface SidebarProps {
  documents: string[];
  loading: boolean;
  error: string | null;
  selected: string | null;
  onSelect: (document: string | null) => void;
  onUploaded: () => void;
  conversations: Conversation[];
  activeConversationId: string | null;
  onNewConversation: () => void;
  onSwitchConversation: (id: string) => void;
}

export function Sidebar({
  documents,
  loading,
  error,
  selected,
  onSelect,
  onUploaded,
  conversations,
  activeConversationId,
  onNewConversation,
  onSwitchConversation,
}: SidebarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { status: uploadStatus, stageMessage, error: uploadError, upload } = useIngest(onUploaded);
  const isUploading = uploadStatus === "uploading";

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    // Resetea el input para poder volver a subir el mismo archivo si falló.
    event.target.value = "";
    if (file) upload(file);
  };

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-black/5 bg-canvas/80 backdrop-blur-xl dark:border-white/10 dark:bg-canvas-dark/80">
      <div className="flex items-center gap-2 px-5 pt-6 pb-4">
        <Library size={18} className="text-ink-secondary dark:text-ink-secondary-dark" />
        <h1 className="text-[15px] font-semibold text-ink dark:text-ink-dark">
          Research Assistant
        </h1>
      </div>

      <div className="flex-1 space-y-5 overflow-y-auto px-3 pb-4">
        <section>
          <div className="flex items-center justify-between px-2 pb-1.5">
            <h2 className="text-[11px] font-semibold tracking-wide text-ink-secondary uppercase dark:text-ink-secondary-dark">
              Conversaciones
            </h2>
            <button
              onClick={onNewConversation}
              className="flex items-center gap-1 rounded-lg px-1.5 py-1 text-[12px] font-medium text-accent transition-colors hover:bg-accent/10"
            >
              <Plus size={13} />
              Nueva
            </button>
          </div>

          <div className="space-y-0.5">
            {conversations.length === 0 && (
              <p className="px-3 py-2 text-[13px] text-ink-secondary dark:text-ink-secondary-dark">
                Sin conversaciones todavía.
              </p>
            )}
            {conversations.map((conversation) => (
              <ConversationItem
                key={conversation.id}
                conversation={conversation}
                active={conversation.id === activeConversationId}
                onClick={() => onSwitchConversation(conversation.id)}
              />
            ))}
          </div>
        </section>

        <section>
          <h2 className="px-2 pb-1.5 text-[11px] font-semibold tracking-wide text-ink-secondary uppercase dark:text-ink-secondary-dark">
            Documentos
          </h2>

          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={handleFileChange}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="mb-1.5 flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-black/15 px-3 py-2 text-[13px] font-medium text-ink-secondary transition-colors hover:bg-black/[0.04] disabled:cursor-not-allowed disabled:opacity-60 dark:border-white/15 dark:text-ink-secondary-dark dark:hover:bg-white/5"
          >
            {isUploading ? (
              <Loader2 size={14} className="shrink-0 animate-spin" />
            ) : (
              <Upload size={14} className="shrink-0" />
            )}
            <span className="truncate">
              {isUploading ? (stageMessage ?? "Subiendo...") : "Subir paper"}
            </span>
          </button>
          {uploadError && <p className="mb-1.5 px-1 text-[12px] text-red-500">{uploadError}</p>}

          <div className="space-y-0.5">
            <SidebarItem
              label="Todos los documentos"
              active={selected === null}
              onClick={() => onSelect(null)}
            />

            {loading && (
              <p className="px-3 py-2 text-[13px] text-ink-secondary dark:text-ink-secondary-dark">
                Cargando...
              </p>
            )}

            {error && <p className="px-3 py-2 text-[13px] text-red-500">{error}</p>}

            {documents.map((doc) => (
              <SidebarItem
                key={doc}
                label={doc}
                active={selected === doc}
                onClick={() => onSelect(doc)}
              />
            ))}
          </div>
        </section>
      </div>
    </aside>
  );
}

function SidebarItem({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-left text-[14px] transition-colors ${
        active
          ? "bg-accent/10 font-medium text-accent"
          : "text-ink-secondary hover:bg-black/[0.04] dark:text-ink-secondary-dark dark:hover:bg-white/5"
      }`}
    >
      <FileText size={15} className="shrink-0" />
      <span className="truncate">{label}</span>
    </button>
  );
}

function ConversationItem({
  conversation,
  active,
  onClick,
}: {
  conversation: Conversation;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex w-full flex-col items-start gap-0.5 rounded-xl px-3 py-2 text-left transition-colors ${
        active ? "bg-accent/10" : "hover:bg-black/[0.04] dark:hover:bg-white/5"
      }`}
    >
      <span
        className={`w-full truncate text-[14px] ${
          active ? "font-medium text-accent" : "text-ink dark:text-ink-dark"
        }`}
      >
        {conversation.title || "Nueva conversación"}
      </span>
      <span className="text-[11px] text-ink-secondary dark:text-ink-secondary-dark">
        {formatRelativeTime(conversation.updatedAt)}
      </span>
    </button>
  );
}
