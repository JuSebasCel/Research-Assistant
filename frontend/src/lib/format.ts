const UNITS: [number, string][] = [
  [60, "s"],
  [60, "min"],
  [24, "h"],
  [7, "d"],
];

/** Formatea un timestamp como "hace 5 min", "hace 2 h", "hace 3 d", etc.
 * Más allá de una semana, cae a fecha corta (dd/mm). */
export function formatRelativeTime(timestamp: number): string {
  let diff = Math.max(0, (Date.now() - timestamp) / 1000);

  if (diff < 10) return "ahora";

  for (const [factor, label] of UNITS) {
    if (diff < factor) return `hace ${Math.floor(diff)} ${label}`;
    diff /= factor;
  }

  return new Date(timestamp).toLocaleDateString("es", { day: "2-digit", month: "2-digit" });
}
