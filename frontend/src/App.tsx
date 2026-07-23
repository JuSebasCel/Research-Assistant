import { useEffect, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { ChatView } from "./components/ChatView";
import { SourcesPanel } from "./components/SourcesPanel";
import { useChatSession } from "./hooks/useChatSession";
import { fetchDocuments } from "./lib/api";

export default function App() {
  const [documents, setDocuments] = useState<string[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);

  const { messages, isStreaming, ask } = useChatSession();

  useEffect(() => {
    fetchDocuments()
      .then(setDocuments)
      .catch((err) => setDocumentsError(err instanceof Error ? err.message : String(err)))
      .finally(() => setDocumentsLoading(false));
  }, []);

  const handleAsk = (query: string) => {
    ask({ query, document_filter: selectedDocument, top_k: 5 });
  };

  // El panel de fuentes siempre refleja las citas del mensaje de asistente
  // más reciente que ya las tenga (no se acumulan citas de toda la
  // conversación, solo la última pregunta respondida).
  const latestCitations =
    [...messages].reverse().find((m) => m.role === "assistant" && m.citations)?.citations ?? [];

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-canvas text-ink dark:bg-canvas-dark dark:text-ink-dark">
      <Sidebar
        documents={documents}
        loading={documentsLoading}
        error={documentsError}
        selected={selectedDocument}
        onSelect={setSelectedDocument}
      />
      <ChatView
        documentFilter={selectedDocument}
        messages={messages}
        isStreaming={isStreaming}
        onAsk={handleAsk}
      />
      <SourcesPanel citations={latestCitations} />
    </div>
  );
}
