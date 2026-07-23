import { FileText, Library } from "lucide-react";

interface SidebarProps {
  documents: string[];
  loading: boolean;
  error: string | null;
  selected: string | null;
  onSelect: (document: string | null) => void;
}

export function Sidebar({ documents, loading, error, selected, onSelect }: SidebarProps) {
  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-black/5 bg-canvas/80 backdrop-blur-xl dark:border-white/10 dark:bg-canvas-dark/80">
      <div className="flex items-center gap-2 px-5 pt-6 pb-4">
        <Library size={18} className="text-ink-secondary dark:text-ink-secondary-dark" />
        <h1 className="text-[15px] font-semibold text-ink dark:text-ink-dark">Papers</h1>
      </div>

      <nav className="flex-1 space-y-0.5 overflow-y-auto px-3 pb-4">
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
      </nav>
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
