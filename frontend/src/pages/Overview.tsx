import { motion } from 'framer-motion';
import { StatusBadge } from '../components/StatusBadge';
import { Brain, Image, Cpu, MessageSquare, Layers, Zap, AlertCircle, CheckCircle2 } from 'lucide-react';

const cards = [
  {
    title: 'Text Sarcasm Model',
    icon: Brain,
    status: 'ready' as const,
    metrics: [{ label: 'Accuracy', value: '77.19%' }, { label: 'Macro F1', value: '77.19%' }],
    note: 'XLM-R Base • Production Ready',
  },
  {
    title: 'Multimodal Fusion Model',
    icon: Layers,
    status: 'ready' as const,
    metrics: [{ label: 'Accuracy', value: '79.62%' }, { label: 'Macro F1', value: '78.64%' }],
    note: '2,847-D Mish MLP • Live on GPU',
  },
  {
    title: 'CLIP Vision Features',
    icon: Image,
    status: 'ready' as const,
    metrics: [{ label: 'Features', value: '33,077' }, { label: 'Dimension', value: '512-D' }],
    note: 'ViT-B/32 Image Embeddings',
  },
  {
    title: 'InternVL3-2B Features',
    icon: Cpu,
    status: 'ready' as const,
    metrics: [{ label: 'Features', value: '33,077' }, { label: 'Dimension', value: '1536-D' }],
    note: 'Vision-Language Representation',
  },
  {
    title: 'Sentiment Model',
    icon: MessageSquare,
    status: 'ready' as const,
    metrics: [{ label: 'Accuracy', value: '73.32%' }, { label: 'Macro F1', value: '72.95%' }],
    note: 'Positive / Neutral / Negative',
  },
  {
    title: 'Emotion Model',
    icon: Zap,
    status: 'ready' as const,
    metrics: [{ label: 'Micro F1', value: '49.26%' }, { label: 'Taxonomy', value: '28-D' }],
    note: 'GoEmotions Multi-Label',
  },
];

const PipelineNode = ({ label, status }: { label: string; status?: 'ready' | 'dev' }) => (
  <div className="px-4 py-2.5 rounded-lg border border-zinc-800 bg-zinc-900 text-center text-xs font-medium text-zinc-200">
    {label}
    {status && (
      <div className={`text-[10px] mt-0.5 font-semibold ${status === 'ready' ? 'text-emerald-400' : 'text-amber-400'}`}>
        {status === 'ready' ? '● Ready' : '● In Dev'}
      </div>
    )}
  </div>
);

const Arrow = () => (
  <div className="flex justify-center">
    <div className="w-px h-5 bg-zinc-700" />
  </div>
);

export function Overview() {
  return (
    <div className="space-y-8">
      {/* Hero */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="py-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-md bg-zinc-900 border border-zinc-800 text-zinc-300 text-xs font-medium mb-3">
          <span className="status-dot ready" />
          Research Prototype — Review 2
        </div>
        <h1 className="text-3xl font-bold text-white mb-2 tracking-tight">Multimodal Sarcasm Intelligence</h1>
        <p className="text-zinc-400 max-w-2xl text-sm leading-relaxed">
          AI-powered sarcasm detection using multilingual transformer representations and vision-language features.
          Text model is production-ready; multimodal fusion is under active development.
        </p>
      </motion.div>

      {/* Status Cards */}
      <div>
        <h2 className="text-xs text-zinc-500 uppercase tracking-widest mb-3 font-semibold">Component Status</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {cards.map((card, i) => {
            const Icon = card.icon;
            return (
              <motion.div
                key={card.title}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="bg-zinc-900/90 border border-zinc-800 p-5 rounded-xl card-hover flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-start justify-between mb-3">
                    <div className="p-2 rounded-lg bg-zinc-800 text-zinc-300">
                      <Icon className="w-4 h-4" />
                    </div>
                    <StatusBadge status={card.status} />
                  </div>
                  <h3 className="font-semibold text-zinc-100 text-sm mb-3">{card.title}</h3>
                  <div className="space-y-1.5">
                    {card.metrics.map(m => (
                      <div key={m.label} className="flex justify-between text-xs">
                        <span className="text-zinc-500">{m.label}</span>
                        <span className="font-mono text-zinc-200 font-semibold">{m.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="text-[11px] text-zinc-500 mt-4 border-t border-zinc-800 pt-2">{card.note}</div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* System Pipeline */}
      <div>
        <h2 className="text-xs text-zinc-500 uppercase tracking-widest mb-3 font-semibold">System Pipeline</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* Text-only pipeline */}
          <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6"
          >
            <div className="flex items-center gap-2 mb-4 pb-3 border-b border-zinc-800">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <h3 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Text-Only Pipeline — ACTIVE</h3>
            </div>
            <div className="space-y-1">
              <PipelineNode label="User Text Input" />
              <Arrow />
              <PipelineNode label="XLM-R Tokenizer" status="ready" />
              <Arrow />
              <PipelineNode label="XLM-R Encoder (768-D)" status="ready" />
              <Arrow />
              <PipelineNode label="Binary Classifier" status="ready" />
              <Arrow />
              <div className="px-4 py-2.5 rounded-lg border border-emerald-800 bg-emerald-950/40 text-center text-xs font-semibold text-emerald-300">
                Sarcasm Detection Output (Ready)
              </div>
            </div>
          </motion.div>

          {/* Multimodal pipeline */}
          <motion.div
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6"
          >
            <div className="flex items-center gap-2 mb-4 pb-3 border-b border-zinc-800">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <h3 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Multimodal Pipeline — ACTIVE & OPERATIONAL</h3>
            </div>
            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <PipelineNode label="Text Input" status="ready" />
                <PipelineNode label="Image Input" status="ready" />
              </div>
              <div className="grid grid-cols-3 gap-1 text-center">
                <PipelineNode label="XLM-R (768-D)" status="ready" />
                <PipelineNode label="CLIP (512-D)" status="ready" />
                <PipelineNode label="InternVL (1536-D)" status="ready" />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <PipelineNode label="Sentiment (3-D)" status="ready" />
                <PipelineNode label="Emotion (28-D)" status="ready" />
              </div>
              <Arrow />
              <PipelineNode label="Feature Fusion (2,847-D)" status="ready" />
              <Arrow />
              <div className="px-4 py-2.5 rounded-lg border border-emerald-800/80 bg-emerald-950/40 text-center text-xs font-semibold text-emerald-300">
                Multimodal Sarcasm Classifier (Trained & Live)
              </div>
            </div>
            <div className="mt-3 text-[11px] text-zinc-400 text-center">
              Trained 2,847-D Mish MLP • 79.62% Test Accuracy • 78.64% Macro F1 • Local XAI Grounding
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
