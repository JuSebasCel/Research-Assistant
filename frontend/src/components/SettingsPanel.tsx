import { useEffect, useState } from "react";
import {
  clearGeminiKey,
  fetchGeminiKeyStatus,
  saveGeminiKey,
  type GeminiKeyStatus,
} from "../lib/api";

export function SettingsPanel() {
  const [status, setStatus] = useState<GeminiKeyStatus | null>(null);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchGeminiKeyStatus()
      .then(setStatus)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const handleSave = async () => {
    const trimmed = draft.trim();
    if (!trimmed) return;
    setSaving(true);
    setError(null);
    try {
      setStatus(await saveGeminiKey(trimmed));
      setDraft("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const handleClear = async () => {
    setSaving(true);
    setError(null);
    try {
      setStatus(await clearGeminiKey());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mx-3 mb-3 rounded-2xl border border-black/5 bg-black/[0.015] p-3 dark:border-white/10 dark:bg-white/[0.02]">
      <h2 className="mb-1.5 text-[11px] font-semibold tracking-wide text-ink-secondary uppercase dark:text-ink-secondary-dark">
        API key de Gemini
      </h2>

      <p className="mb-2 text-[12px] text-ink-secondary dark:text-ink-secondary-dark">
        {status?.has_custom_key
          ? `Usando tu propia clave (termina en ${status.key_hint}).`
          : "Usando la clave configurada por el servidor."}
      </p>

      <div className="flex gap-1.5">
        <input
          type="password"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSave()}
          placeholder="Pega tu API key"
          disabled={saving}
          className="min-w-0 flex-1 rounded-xl border border-black/10 bg-surface px-3 py-1.5 text-[13px] text-ink outline-none focus:border-accent/40 dark:border-white/10 dark:bg-surface-dark dark:text-ink-dark"
        />
        <button
          onClick={handleSave}
          disabled={saving || !draft.trim()}
          className="shrink-0 rounded-xl bg-accent px-3 py-1.5 text-[13px] font-medium text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
        >
          Guardar
        </button>
      </div>

      {status?.has_custom_key && (
        <button
          onClick={handleClear}
          disabled={saving}
          className="mt-1.5 text-[12px] text-ink-secondary hover:text-ink dark:text-ink-secondary-dark dark:hover:text-ink-dark"
        >
          Volver a la clave del servidor
        </button>
      )}

      {error && <p className="mt-1.5 text-[12px] text-red-500">{error}</p>}

      <a
        href="https://aistudio.google.com/apikey"
        target="_blank"
        rel="noreferrer"
        className="mt-1.5 block text-[11px] text-accent hover:underline"
      >
        Conseguir una key gratis en Google AI Studio →
      </a>
    </div>
  );
}
