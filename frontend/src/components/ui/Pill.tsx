import type { HTMLAttributes } from "react";

interface PillProps extends HTMLAttributes<HTMLSpanElement> {
  active?: boolean;
}

export function Pill({ active = false, className = "", ...props }: PillProps) {
  const base =
    "inline-flex items-center gap-1 rounded-full px-3 py-1 text-[13px] font-medium transition-colors";
  const state = active
    ? "bg-accent text-white"
    : "bg-black/[0.04] text-ink-secondary hover:bg-black/[0.07] dark:bg-white/10 dark:text-ink-secondary-dark dark:hover:bg-white/15";

  return <span className={`${base} ${state} ${className}`} {...props} />;
}
