import { BookOpen } from "lucide-react";
import { API_BASE, type Citation } from "../lib/api";
import { Card } from "./ui/Card";
import { Pill } from "./ui/Pill";

interface SourcesPanelProps {
  citations: Citation[];
}

export function SourcesPanel({ citations }: SourcesPanelProps) {
  if (citations.length === 0) {
    return (
      <aside className="hidden w-80 shrink-0 flex-col gap-3 border-l border-black/5 p-5 dark:border-white/10 lg:flex">
        <div className="flex items-center gap-2 text-ink-secondary dark:text-ink-secondary-dark">
          <BookOpen size={16} />
          <h2 className="text-[13px] font-semibold tracking-wide uppercase">Fuentes</h2>
        </div>
        <p className="text-[13px] text-ink-secondary dark:text-ink-secondary-dark">
          Las citas de tu próxima pregunta van a aparecer acá, con las figuras asociadas.
        </p>
      </aside>
    );
  }

  return (
    <aside className="hidden w-80 shrink-0 flex-col gap-3 overflow-y-auto border-l border-black/5 p-5 dark:border-white/10 lg:flex">
      <div className="flex items-center gap-2 text-ink-secondary dark:text-ink-secondary-dark">
        <BookOpen size={16} />
        <h2 className="text-[13px] font-semibold tracking-wide uppercase">Fuentes</h2>
      </div>

      {citations.map((citation) => (
        <Card key={citation.chunk_id} className="p-4">
          <div className="mb-2 flex flex-wrap items-center gap-1.5">
            <Pill active>{citation.document_name}</Pill>
            {citation.pages.length > 0 && (
              <span className="text-[12px] text-ink-secondary dark:text-ink-secondary-dark">
                p. {citation.pages.join(", ")}
              </span>
            )}
          </div>

          {citation.image_urls.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
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
        </Card>
      ))}
    </aside>
  );
}
