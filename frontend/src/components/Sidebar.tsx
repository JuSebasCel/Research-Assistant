import {
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Folder,
  FolderPlus,
  Globe,
  Library,
  Loader2,
  Pencil,
  Pin,
  Plus,
  Settings,
  Trash2,
  Upload,
} from "lucide-react";
import { useRef, useState, type ChangeEvent, type DragEvent, type KeyboardEvent } from "react";
import type { Conversation } from "../hooks/useConversations";
import { useIngest } from "../hooks/useIngest";
import { type DocumentInfo, updateDocument } from "../lib/api";
import { formatRelativeTime } from "../lib/format";
import { SettingsPanel } from "./SettingsPanel";

// Clave custom para dataTransfer: evita colisionar con drags nativos del
// navegador (ej. arrastrar texto o links) que también usan "text/plain".
const DOCUMENT_DRAG_TYPE = "application/x-document-name";

interface SidebarProps {
  documents: DocumentInfo[];
  loading: boolean;
  error: string | null;
  selectedDocument: string | null;
  selectedFolder: string | null;
  onSelectDocument: (document: string | null) => void;
  onSelectFolder: (folder: string | null) => void;
  onUploaded: () => void;
  onDocumentUpdated: () => void;
  conversations: Conversation[];
  activeConversationId: string | null;
  onNewConversation: () => void;
  onSwitchConversation: (id: string) => void;
  onRenameConversation: (id: string, title: string) => void;
  onTogglePinConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
}

export function Sidebar({
  documents,
  loading,
  error,
  selectedDocument,
  selectedFolder,
  onSelectDocument,
  onSelectFolder,
  onUploaded,
  onDocumentUpdated,
  conversations,
  activeConversationId,
  onNewConversation,
  onSwitchConversation,
  onRenameConversation,
  onTogglePinConversation,
  onDeleteConversation,
}: SidebarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { status: uploadStatus, stageMessage, error: uploadError, upload } = useIngest(onUploaded);
  const isUploading = uploadStatus === "uploading";
  const [documentsOpen, setDocumentsOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [collapsedFolders, setCollapsedFolders] = useState<Set<string>>(new Set());
  const [dragOverFolder, setDragOverFolder] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    // Resetea el input para poder volver a subir el mismo archivo si falló.
    event.target.value = "";
    if (files.length > 0) upload(files);
  };

  const toggleFolderCollapsed = (folder: string) => {
    setCollapsedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(folder)) next.delete(folder);
      else next.add(folder);
      return next;
    });
  };

  const handleDropOnFolder = async (event: DragEvent<HTMLElement>, folder: string) => {
    event.preventDefault();
    setDragOverFolder(null);
    const documentName = event.dataTransfer.getData(DOCUMENT_DRAG_TYPE);
    const doc = documents.find((d) => d.document_name === documentName);
    if (!doc || doc.folder === folder) return;
    await updateDocument(documentName, { folder });
    onDocumentUpdated();
  };

  const folderNames = Array.from(
    new Set(documents.filter((d) => d.folder).map((d) => d.folder as string))
  ).sort();
  const unfoldered = documents.filter((d) => !d.folder);

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-black/5 bg-canvas/80 backdrop-blur-xl dark:border-white/10 dark:bg-canvas-dark/80">
      <div className="flex items-center gap-2 px-5 pt-6 pb-4">
        <Library size={18} className="text-ink-secondary dark:text-ink-secondary-dark" />
        <h1 className="flex-1 text-[15px] font-semibold text-ink dark:text-ink-dark">
          Research Assistant
        </h1>
        <button
          onClick={() => setSettingsOpen((v) => !v)}
          title="Configuración"
          className="rounded-lg p-1 text-ink-secondary transition-colors hover:bg-black/[0.04] hover:text-ink dark:text-ink-secondary-dark dark:hover:bg-white/5 dark:hover:text-ink-dark"
        >
          <Settings size={16} />
        </button>
      </div>

      {settingsOpen && <SettingsPanel />}

      <div className="flex-1 space-y-3 overflow-y-auto px-3 pb-4">
        <section className="rounded-2xl border border-black/5 bg-black/[0.015] p-2 dark:border-white/10 dark:bg-white/[0.02]">
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
                onRename={(title) => onRenameConversation(conversation.id, title)}
                onTogglePin={() => onTogglePinConversation(conversation.id)}
                onDelete={() => onDeleteConversation(conversation.id)}
              />
            ))}
          </div>
        </section>

        <section className="rounded-2xl border border-black/5 bg-black/[0.015] p-2 dark:border-white/10 dark:bg-white/[0.02]">
          <button
            onClick={() => setDocumentsOpen((v) => !v)}
            className="flex w-full items-center justify-between px-2 pb-1.5"
          >
            <h2 className="text-[11px] font-semibold tracking-wide text-ink-secondary uppercase dark:text-ink-secondary-dark">
              Documentos
            </h2>
            {documentsOpen ? (
              <ChevronUp size={14} className="text-ink-secondary dark:text-ink-secondary-dark" />
            ) : (
              <ChevronDown size={14} className="text-ink-secondary dark:text-ink-secondary-dark" />
            )}
          </button>

          {documentsOpen && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf"
                multiple
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
                <span className="min-w-0 truncate">
                  {isUploading ? (stageMessage ?? "Subiendo...") : "Subir papers"}
                </span>
              </button>

              <button
                disabled
                title="Próximamente: importar papers en lote desde APIs externas (arXiv, PubMed, etc.)"
                className="mb-1.5 flex w-full cursor-not-allowed items-center justify-center gap-2 rounded-xl border border-dashed border-black/10 px-3 py-2 text-[13px] font-medium text-ink-secondary/50 dark:border-white/10 dark:text-ink-secondary-dark/50"
              >
                <Globe size={14} className="shrink-0" />
                <span className="min-w-0 truncate">Importar desde arXiv / PubMed</span>
              </button>

              {uploadError && (
                <p className="mb-1.5 px-1 text-[12px] text-red-500">{uploadError}</p>
              )}

              <div className="space-y-0.5">
                <SidebarItem
                  label="Todos los documentos"
                  active={selectedDocument === null && selectedFolder === null}
                  onClick={() => onSelectDocument(null)}
                />

                {loading && (
                  <p className="px-3 py-2 text-[13px] text-ink-secondary dark:text-ink-secondary-dark">
                    Cargando...
                  </p>
                )}

                {error && <p className="px-3 py-2 text-[13px] text-red-500">{error}</p>}

                {folderNames.map((folder) => {
                  const collapsed = collapsedFolders.has(folder);
                  return (
                    <div key={folder}>
                      <div className="flex items-center gap-0.5">
                        <button
                          onClick={() => toggleFolderCollapsed(folder)}
                          title={collapsed ? "Expandir carpeta" : "Colapsar carpeta"}
                          className="shrink-0 rounded-lg p-1 text-ink-secondary hover:text-ink dark:text-ink-secondary-dark dark:hover:text-ink-dark"
                        >
                          {collapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
                        </button>
                        <button
                          onClick={() => onSelectFolder(folder)}
                          onDragOver={(e) => {
                            e.preventDefault();
                            setDragOverFolder(folder);
                          }}
                          onDragLeave={() =>
                            setDragOverFolder((current) => (current === folder ? null : current))
                          }
                          onDrop={(e) => handleDropOnFolder(e, folder)}
                          className={`flex min-w-0 flex-1 items-center gap-2.5 rounded-xl px-2 py-2 text-left text-[14px] transition-colors ${
                            selectedFolder === folder
                              ? "bg-accent/10 font-medium text-accent"
                              : "text-ink-secondary hover:bg-black/[0.04] dark:text-ink-secondary-dark dark:hover:bg-white/5"
                          } ${dragOverFolder === folder ? "ring-2 ring-accent/50" : ""}`}
                        >
                          <Folder size={15} className="shrink-0" />
                          <span className="min-w-0 truncate">{folder}</span>
                        </button>
                      </div>
                      {!collapsed && (
                        <div className="ml-3 space-y-0.5 border-l border-black/5 pl-1.5 dark:border-white/10">
                          {documents
                            .filter((d) => d.folder === folder)
                            .map((doc) => (
                              <DocumentItem
                                key={doc.document_name}
                                document={doc}
                                active={selectedDocument === doc.document_name}
                                onSelect={() => onSelectDocument(doc.document_name)}
                                onUpdated={onDocumentUpdated}
                              />
                            ))}
                        </div>
                      )}
                    </div>
                  );
                })}

                {unfoldered.map((doc) => (
                  <DocumentItem
                    key={doc.document_name}
                    document={doc}
                    active={selectedDocument === doc.document_name}
                    onSelect={() => onSelectDocument(doc.document_name)}
                    onUpdated={onDocumentUpdated}
                  />
                ))}
              </div>
            </>
          )}
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
      <span className="min-w-0 truncate">{label}</span>
    </button>
  );
}

function InlineTextInput({
  value,
  onSave,
  onCancel,
}: {
  value: string;
  onSave: (value: string) => void;
  onCancel: () => void;
}) {
  const [draft, setDraft] = useState(value);

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") onSave(draft);
    if (event.key === "Escape") onCancel();
  };

  return (
    <input
      autoFocus
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => onSave(draft)}
      onKeyDown={handleKeyDown}
      className="w-full rounded-xl border border-accent/40 bg-surface px-3 py-2 text-[14px] text-ink outline-none dark:bg-surface-dark dark:text-ink-dark"
    />
  );
}

function DocumentItem({
  document,
  active,
  onSelect,
  onUpdated,
}: {
  document: DocumentInfo;
  active: boolean;
  onSelect: () => void;
  onUpdated: () => void;
}) {
  const [editingName, setEditingName] = useState(false);
  const [editingFolder, setEditingFolder] = useState(false);

  const saveName = async (draft: string) => {
    setEditingName(false);
    const trimmed = draft.trim();
    if (!trimmed || trimmed === document.display_name) return;
    await updateDocument(document.document_name, { display_name: trimmed });
    onUpdated();
  };

  const saveFolder = async (draft: string) => {
    setEditingFolder(false);
    const newFolder = draft.trim() || null;
    if (newFolder === document.folder) return;
    await updateDocument(document.document_name, { folder: newFolder });
    onUpdated();
  };

  if (editingName) {
    return <InlineTextInput value={document.display_name} onSave={saveName} onCancel={() => setEditingName(false)} />;
  }

  if (editingFolder) {
    return (
      <InlineTextInput
        value={document.folder ?? ""}
        onSave={saveFolder}
        onCancel={() => setEditingFolder(false)}
      />
    );
  }

  return (
    <div
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(DOCUMENT_DRAG_TYPE, document.document_name);
        e.dataTransfer.effectAllowed = "move";
      }}
      title="Arrastrá para mover a una carpeta"
      className={`group flex cursor-grab items-center gap-0.5 rounded-xl pr-1 transition-colors active:cursor-grabbing ${
        active ? "bg-accent/10" : "hover:bg-black/[0.04] dark:hover:bg-white/5"
      }`}
    >
      <button
        onClick={onSelect}
        className={`flex min-w-0 flex-1 items-center gap-2 py-2 pl-3 text-left text-[14px] ${
          active ? "font-medium text-accent" : "text-ink-secondary dark:text-ink-secondary-dark"
        }`}
      >
        <span className="min-w-0 truncate">{document.display_name}</span>
      </button>
      <button
        onClick={() => setEditingName(true)}
        title="Renombrar"
        className="shrink-0 rounded-lg p-1 text-ink-secondary opacity-0 transition-opacity group-hover:opacity-100 hover:text-ink dark:text-ink-secondary-dark dark:hover:text-ink-dark"
      >
        <Pencil size={12} />
      </button>
      <button
        onClick={() => setEditingFolder(true)}
        title="Carpeta"
        className="shrink-0 rounded-lg p-1 text-ink-secondary opacity-0 transition-opacity group-hover:opacity-100 hover:text-ink dark:text-ink-secondary-dark dark:hover:text-ink-dark"
      >
        <FolderPlus size={12} />
      </button>
    </div>
  );
}

function ConversationItem({
  conversation,
  active,
  onClick,
  onRename,
  onTogglePin,
  onDelete,
}: {
  conversation: Conversation;
  active: boolean;
  onClick: () => void;
  onRename: (title: string) => void;
  onTogglePin: () => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);

  if (editing) {
    return (
      <InlineTextInput
        value={conversation.title}
        onSave={(title) => {
          setEditing(false);
          if (title.trim()) onRename(title);
        }}
        onCancel={() => setEditing(false)}
      />
    );
  }

  return (
    <div
      className={`group flex items-center gap-0.5 rounded-xl pr-1 transition-colors ${
        active ? "bg-accent/10" : "hover:bg-black/[0.04] dark:hover:bg-white/5"
      }`}
    >
      <button
        onClick={onClick}
        className="flex min-w-0 flex-1 flex-col items-start gap-0.5 py-2 pl-3 text-left"
      >
        <span className="flex w-full min-w-0 items-center gap-1 text-[14px]">
          {conversation.pinned && <Pin size={10} className="shrink-0 fill-current" />}
          <span
            className={`min-w-0 flex-1 truncate ${
              active ? "font-medium text-accent" : "text-ink dark:text-ink-dark"
            }`}
          >
            {conversation.title || "Nueva conversación"}
          </span>
        </span>
        <span className="text-[11px] text-ink-secondary dark:text-ink-secondary-dark">
          {formatRelativeTime(conversation.updatedAt)}
        </span>
      </button>
      <div className="flex shrink-0 items-center opacity-0 transition-opacity group-hover:opacity-100">
        <button
          onClick={onTogglePin}
          title={conversation.pinned ? "Quitar pin" : "Pinear"}
          className="rounded-lg p-1 text-ink-secondary hover:text-ink dark:text-ink-secondary-dark dark:hover:text-ink-dark"
        >
          <Pin size={12} />
        </button>
        <button
          onClick={() => setEditing(true)}
          title="Renombrar"
          className="rounded-lg p-1 text-ink-secondary hover:text-ink dark:text-ink-secondary-dark dark:hover:text-ink-dark"
        >
          <Pencil size={12} />
        </button>
        <button
          onClick={() => {
            if (window.confirm("¿Borrar esta conversación? No se puede deshacer.")) onDelete();
          }}
          title="Borrar"
          className="rounded-lg p-1 text-ink-secondary hover:text-red-500 dark:text-ink-secondary-dark"
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}
