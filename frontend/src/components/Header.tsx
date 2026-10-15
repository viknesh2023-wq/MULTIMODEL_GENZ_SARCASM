import { Brain } from 'lucide-react';

interface HeaderProps {
  backendOnline: boolean;
  modelLoaded: boolean;
  multimodalReady?: boolean;
}

export function Header({ backendOnline, modelLoaded, multimodalReady = false }: HeaderProps) {
  return (
    <header className="sticky top-0 z-10 border-b border-zinc-800 bg-black px-6 py-3">
      <div className="flex items-center justify-between">
        {/* Left — Title */}
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center">
            <Brain className="w-3.5 h-3.5 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white tracking-tight">Multimodal Sarcasm Intelligence</h1>
            <p className="text-[10px] text-zinc-500 hidden sm:block">Gen-Z Sarcasm Detection Research Dashboard</p>
          </div>
        </div>

        {/* Right — status pills */}
        <div className="flex items-center gap-2 text-xs">
          <StatusPill label="Backend" online={backendOnline} />
          <StatusPill label="XLM-R" online={modelLoaded} />
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-[11px] font-medium
            ${multimodalReady
              ? 'bg-zinc-900 border-zinc-800 text-zinc-300'
              : 'bg-amber-950/30 border-amber-800/50 text-amber-300'
            }`}>
            <span className={`status-dot ${multimodalReady ? 'ready' : 'dev'}`} />
            <span className={`font-medium text-[11px] ${multimodalReady ? 'text-emerald-400' : 'text-amber-400'}`}>
              Multimodal: {multimodalReady ? 'Ready' : 'In Dev'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}

function StatusPill({ label, online }: { label: string; online: boolean }) {
  return (
    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-[11px] font-medium
      ${online
        ? 'bg-zinc-900 border-zinc-800 text-zinc-300'
        : 'bg-rose-950/30 border-rose-900/50 text-rose-300'
      }`}>
      <span className={`status-dot ${online ? 'ready' : 'error'}`} />
      <span>{label}: {online ? 'Ready' : 'Offline'}</span>
    </div>
  );
}
