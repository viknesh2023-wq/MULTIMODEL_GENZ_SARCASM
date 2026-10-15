import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard, Zap, FileText, Layers, GitBranch,
  BarChart2, Cpu, Database, BookOpen, Info,
  ChevronLeft, ChevronRight, Brain
} from 'lucide-react';
import type { NavPage } from '../types';

interface SidebarProps {
  current: NavPage;
  onChange: (page: NavPage) => void;
  backendOnline: boolean;
  modelLoaded: boolean;
  multimodalReady?: boolean;
}

const navItems: { id: NavPage; label: string; icon: React.ElementType }[] = [
  { id: 'overview',      label: 'Overview',           icon: LayoutDashboard },
  { id: 'detection',     label: 'Sarcasm Detection',  icon: Zap },
  { id: 'text-analysis', label: 'Text Analysis',      icon: FileText },
  { id: 'multimodal',   label: 'Multimodal Analysis', icon: Layers },
  { id: 'architecture', label: 'Model Architecture',  icon: GitBranch },
  { id: 'performance',  label: 'Model Performance',   icon: BarChart2 },
  { id: 'features',     label: 'Feature Extraction',  icon: Cpu },
  { id: 'dataset',      label: 'Dataset',             icon: Database },
  { id: 'research',     label: 'Research & Novelty',  icon: BookOpen },
  { id: 'about',        label: 'About Project',       icon: Info },
];

export function Sidebar({ current, onChange, backendOnline, modelLoaded, multimodalReady = false }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <motion.aside
      animate={{ width: collapsed ? 64 : 240 }}
      transition={{ duration: 0.2, ease: 'easeInOut' }}
      className="flex-shrink-0 h-screen sticky top-0 flex flex-col border-r border-zinc-800 bg-black z-20 overflow-hidden"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-zinc-800">
        <div className="w-8 h-8 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center flex-shrink-0">
          <Brain className="w-4 h-4 text-white" />
        </div>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.15 }}
              className="overflow-hidden"
            >
              <div className="text-xs font-bold text-white leading-tight whitespace-nowrap">Multimodal Sarcasm</div>
              <div className="text-[10px] text-zinc-500 whitespace-nowrap font-mono">Intelligence v1.0</div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Nav items */}
      <nav className="flex-1 overflow-y-auto py-3 space-y-1 px-2">
        {navItems.map(item => {
          const Icon = item.icon;
          const active = current === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onChange(item.id)}
              title={collapsed ? item.label : undefined}
              className={`
                w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors group
                ${active
                  ? 'bg-zinc-800 text-white font-semibold border border-zinc-700'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-900 border border-transparent'
                }
              `}
            >
              <Icon className={`w-4 h-4 flex-shrink-0 ${active ? 'text-white' : 'text-zinc-400 group-hover:text-zinc-200'}`} />
              <AnimatePresence>
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    className="text-xs font-medium whitespace-nowrap overflow-hidden"
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>
            </button>
          );
        })}
      </nav>

      {/* System Status */}
      <div className="border-t border-zinc-800 p-3 space-y-2">
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-2"
            >
              <div className="text-[10px] text-zinc-500 uppercase tracking-widest px-1 font-semibold">System Status</div>
              <div className="space-y-1.5">
                <div className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800">
                  <span className="text-xs text-zinc-400">TEXT MODEL</span>
                  <div className="flex items-center gap-1.5">
                    <span className={`status-dot ${modelLoaded ? 'ready' : 'error'}`} />
                    <span className={`text-[10px] font-semibold ${modelLoaded ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {modelLoaded ? 'ONLINE' : 'OFFLINE'}
                    </span>
                  </div>
                </div>
                <div className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800">
                  <span className="text-xs text-zinc-400">MULTIMODAL</span>
                  <div className="flex items-center gap-1.5">
                    <span className={`status-dot ${multimodalReady ? 'ready' : 'dev'}`} />
                    <span className={`text-[10px] font-semibold ${multimodalReady ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {multimodalReady ? 'ONLINE' : 'IN DEV'}
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center p-2 rounded-lg text-zinc-500 hover:text-zinc-200 hover:bg-zinc-900 transition-colors"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>
    </motion.aside>
  );
}
