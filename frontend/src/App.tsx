import { useEffect, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { ChatView } from "./components/ChatView";
import { useConversations } from "./hooks/useConversations";
import { fetchDocuments, type DocumentInfo } from "./lib/api";

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  // Mutuamente excluyentes: seleccionar un documento puntual limpia la
  // carpeta seleccionada y viceversa (ver Sidebar).
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);

  const {
    conversations,
    activeConversationId,
    messages,
    isStreaming,
    createConversation,
    switchConversation,
    renameConversation,
    togglePin,
    deleteConversation,
    ask,
  } = useConversations();

  const loadDocuments = () => {
    setDocumentsLoading(true);
    fetchDocuments()
      .then(setDocuments)
      .catch((err) => setDocumentsError(err instanceof Error ? err.message : String(err)))
      .finally(() => setDocumentsLoading(false));
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  const handleSelectDocument = (document: string | null) => {
    setSelectedDocument(document);
    setSelectedFolder(null);
  };

  const handleSelectFolder = (folder: string | null) => {
    setSelectedFolder(folder);
    setSelectedDocument(null);
  };

  const handleAsk = (query: string) => {
    if (selectedFolder) {
      const documentFilters = documents
        .filter((d) => d.folder === selectedFolder)
        .map((d) => d.document_name);
      ask({ query, document_filters: documentFilters, top_k: 5 });
    } else {
      ask({ query, document_filter: selectedDocument, top_k: 5 });
    }
  };

  const selectedDocumentLabel = selectedDocument
    ? (documents.find((d) => d.document_name === selectedDocument)?.display_name ??
      selectedDocument)
    : null;
  const scopeLabel = selectedFolder
    ? `la carpeta "${selectedFolder}"`
    : selectedDocumentLabel
      ? `"${selectedDocumentLabel}"`
      : null;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-canvas text-ink dark:bg-canvas-dark dark:text-ink-dark">
      <Sidebar
        documents={documents}
        loading={documentsLoading}
        error={documentsError}
        selectedDocument={selectedDocument}
        selectedFolder={selectedFolder}
        onSelectDocument={handleSelectDocument}
        onSelectFolder={handleSelectFolder}
        onUploaded={loadDocuments}
        onDocumentUpdated={loadDocuments}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onNewConversation={createConversation}
        onSwitchConversation={switchConversation}
        onRenameConversation={renameConversation}
        onTogglePinConversation={togglePin}
        onDeleteConversation={deleteConversation}
      />
      <ChatView scopeLabel={scopeLabel} messages={messages} isStreaming={isStreaming} onAsk={handleAsk} />
    </div>
  );
}
