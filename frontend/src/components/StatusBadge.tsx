import { cn } from '../utils/textAnalysis';

type Status = 'ready' | 'dev' | 'soon' | 'error';

interface StatusBadgeProps {
  status: Status;
  label?: string;
  className?: string;
}

const config: Record<Status, { dot: string; bg: string; text: string; default: string }> = {
  ready:  { dot: 'status-dot ready', bg: 'bg-emerald-950/50 border-emerald-800/60', text: 'text-emerald-300', default: 'READY' },
  dev:    { dot: 'status-dot dev',   bg: 'bg-amber-950/50 border-amber-800/60',     text: 'text-amber-300',   default: 'IN DEVELOPMENT' },
  soon:   { dot: 'status-dot soon',  bg: 'bg-zinc-800/60 border-zinc-700',          text: 'text-zinc-300',    default: 'COMING SOON' },
  error:  { dot: 'status-dot error', bg: 'bg-rose-950/50 border-rose-800/60',       text: 'text-rose-300',    default: 'ERROR' },
};

export function StatusBadge({ status, label, className }: StatusBadgeProps) {
  const c = config[status];
  return (
    <span className={cn(
      'inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-[11px] font-medium border tracking-normal',
      c.bg, c.text, className
    )}>
      <span className={c.dot} />
      {label ?? c.default}
    </span>
  );
}
