import { useState } from "react";
import { BookOpen, ChevronDown, ChevronUp, FileText } from "lucide-react";
import { API_BASE, type Citation } from "../lib/api";
import { Card } from "./ui/Card";
import { Pill } from "./ui/Pill";

interface MessageSourcesProps {
  citations: Citation[];
}

export function MessageSources({ citations }: MessageSourcesProps) {
  const [expanded, setExpanded] = useState(false);

  if (citations.length === 0) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center gap-1.5 text-[12px] text-ink-secondary transition-colors hover:text-ink dark:text-ink-secondary-dark dark:hover:text-ink-dark"
      >
        <BookOpen size={13} />
        Fuentes ({citations.length})
        {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>

      {expanded && (
        <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
          {citations.map((citation) => (
            <Card key={citation.chunk_id} className="p-3">
              <div className="mb-2 flex flex-wrap items-center gap-1.5">
                <Pill active>{citation.document_name}</Pill>
                {citation.pages.length > 0 && (
                  <span className="text-[12px] text-ink-secondary dark:text-ink-secondary-dark">
                    p. {citation.pages.join(", ")}
                  </span>
                )}
              </div>

              {citation.image_urls.length > 0 && (
                <div className="mb-2 grid grid-cols-2 gap-2">
                  {citation.image_urls.map((url) => (
                    <img
                      key={url}
                      src={`${API_BASE}${url}`}
                      alt={`Figura de ${citation.document_name}`}
                      loading="lazy"
                      className="aspect-square rounded-lg border border-black/5 object-cover dark:border-white/10"
                    />
                  ))}
                </div>
              )}

              <a
                href={`${API_BASE}/static/uploads/${encodeURIComponent(citation.document_name)}.pdf#page=${citation.pages[0] ?? 1}`}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-1 text-[12px] font-medium text-accent hover:underline"
              >
                <FileText size={12} />
                Ver PDF
              </a>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
