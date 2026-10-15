import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  GitBranch, X, Layers, Cpu, Brain, MessageSquare, Zap, Eye,
  Sparkles, CheckCircle2, ShieldCheck, ArrowDown, HelpCircle, Terminal
} from 'lucide-react';
import { StatusBadge } from '../components/StatusBadge';

interface ComponentInfo {
  id: string;
  name: string;
  checkpoint: string;
  purpose: string;
  input: string;
  output: string;
  dimensions: string;
  status: 'ready' | 'dev' | 'soon';
  hardware: string;
  icon: any;
}

const components: ComponentInfo[] = [
  {
    id: 'xlmr',
    name: 'XLM-R Contextual Text Encoder',
    checkpoint: 'D:\\NLP_PROJECT\\models\\xlmr_text_sarcasm_full',
    purpose: 'Multilingual transformer (FacebookAI/xlm-roberta-base) fine-tuned on 1,005,167 samples. Provides 768-D masked mean-pooled contextual text embeddings and standalone text sarcasm predictions.',
    input: 'Raw tokenized social media text (max_length = 128)',
    output: '768-D float32 contextual representation tensor',
    dimensions: '768 Dimensions',
    status: 'ready',
    hardware: 'CUDA GPU (~1,112 MB VRAM)',
    icon: Brain,
  },
  {
    id: 'clip',
    name: 'OpenAI CLIP ViT-B/32 Visual Encoder',
    checkpoint: 'OpenAI ViT-B/32 (Resident in memory)',
    purpose: 'Extracts global visual semantic representations from uploaded images, capturing object presence and literal scene aesthetics.',
    input: '224×224 normalized RGB image tensor',
    output: '512-D float32 normalized image embedding',
    dimensions: '512 Dimensions',
    status: 'ready',
    hardware: 'CUDA GPU (~360 MB VRAM)',
    icon: Eye,
  },
  {
    id: 'internvl',
    name: 'InternVL3-2B Semantic Vision-Language Model',
    checkpoint: 'OpenGVLab/InternVL3-2B (Resident in bfloat16)',
    purpose: 'Deep foundation vision-language model capturing complex semantic relationships, scene nuances, and physical visual features (e.g. scoreboard text, expressions).',
    input: 'Dynamic multi-tile image tensor (up to 12 tiles of 448×448)',
    output: '1,536-D float32 deep vision-language representation',
    dimensions: '1,536 Dimensions',
    status: 'ready',
    hardware: 'CUDA GPU (~3,850 MB VRAM, bfloat16)',
    icon: Cpu,
  },
  {
    id: 'sentiment',
    name: 'XLM-R 3-Class Sentiment Branch',
    checkpoint: 'D:\\NLP_PROJECT\\models\\xlmr_sentiment_model',
    purpose: 'Auxiliary classifier estimating valence polarity mismatch [Negative, Neutral, Positive]. Sarcasm often hinges on positive literal text clashing with negative reality.',
    input: 'Tokenized text sequence',
    output: '3-D softmax probability vector [Neg, Neu, Pos]',
    dimensions: '3 Dimensions',
    status: 'ready',
    hardware: 'Host CPU Offloaded (0 MB VRAM)',
    icon: MessageSquare,
  },
  {
    id: 'emotion',
    name: 'XLM-R 28-Class GoEmotions Affect Branch',
    checkpoint: 'D:\\NLP_PROJECT\\models\\xlmr_emotion_model',
    purpose: 'Auxiliary multi-label affect classifier outputting 28 independent continuous sigmoid probabilities to capture nuanced conflicting emotions (e.g. amusement vs annoyance).',
    input: 'Tokenized text sequence',
    output: '28-D independent sigmoid continuous probabilities',
    dimensions: '28 Dimensions',
    status: 'ready',
    hardware: 'Host CPU Offloaded (0 MB VRAM)',
    icon: Zap,
  },
  {
    id: 'fusion',
    name: 'Trained Multimodal Fusion MLP Classifier',
    checkpoint: 'D:\\NLP_PROJECT\\models\\multimodal_fusion\\fusion_model.pt',
    purpose: '4-layer Mish MLP with dropout trained on 25,384 samples. Performs non-linear cross-modal interaction synthesis and outputs authoritative multimodal sarcasm verdict.',
    input: '2,847-D normalized vector via StandardScaler',
    output: 'Calibrated binary probabilities (Non-Sarcastic vs Sarcastic)',
    dimensions: '2,847 Dimensions Input → 2 Output',
    status: 'ready',
    hardware: 'CUDA GPU (~14 MB VRAM, 0.104 ms latency)',
    icon: Sparkles,
  },
  {
    id: 'explainer',
    name: 'Local Explainable AI Reasoner (XAI)',
    checkpoint: 'Ollama llama3.2:1b (CPU) + Deterministic Fallback',
    purpose: 'Post-hoc interpretability engine generating grounded 5-part natural language explanations without modifying classifier probabilities (Delta p = 0).',
    input: 'Frozen prediction, text evidence, visual observations, affect vectors',
    output: 'Structured JSON: summary, text evidence, visual evidence, signals, limitations',
    dimensions: 'Structured Explainability Payload',
    status: 'ready',
    hardware: 'Host CPU (num_gpu: 0, 0 MB GPU Delta)',
    icon: Terminal,
  },
];

export function ModelArchitecture() {
  const [selected, setSelected] = useState<ComponentInfo | null>(null);

  const open = (id: string) => {
    const found = components.find((c) => c.id === id);
    if (found) setSelected(found);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2 mb-2">
          <div className="p-2 rounded-lg bg-zinc-800 border border-zinc-700">
            <GitBranch className="w-5 h-5 text-zinc-200" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">System Architecture & Model Routing Paths</h1>
        </div>
        <p className="text-zinc-400 text-sm">
          Comprehensive five-branch multimodal architecture detailing model checkpoints, tensor dimensions, hardware residency, and explanation isolation.
        </p>

        <div className="mt-3 flex items-center gap-2 px-3.5 py-2.5 rounded-xl bg-zinc-900 border border-zinc-800 text-xs text-zinc-300">
          <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>
            <strong>Architectural Invariant:</strong> The explanation reasoner is strictly post-hoc. Classifier predictions and softmax probabilities are mathematically invariant to the explanation engine (Δp = 0.000000).
          </span>
        </div>
      </motion.div>

      {/* 5 Core Inference Paths */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Path A */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Path A • Text Only</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">&lt; 25 ms</span>
          </div>
          <h3 className="text-sm font-bold text-white">Standalone Text Sarcasm</h3>
          <p className="text-xs text-zinc-400">
            Raw Text → Fine-Tuned XLM-R Classifier → 2-Class Softmax → Predicted Label (77.19% Test F1).
          </p>
          <div className="pt-2 border-t border-zinc-800 text-[11px] font-mono text-zinc-500">
            Checkpoint: <code>models/xlmr_text_sarcasm_full</code>
          </div>
        </div>

        {/* Path B */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">Path B • Sentiment</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">3-D Vector</span>
          </div>
          <h3 className="text-sm font-bold text-white">Auxiliary Sentiment Modeling</h3>
          <p className="text-xs text-zinc-400">
            Raw Text → Fine-Tuned XLM-R Sentiment → 3-Class Softmax: [Negative, Neutral, Positive] (72.95% Macro F1).
          </p>
          <div className="pt-2 border-t border-zinc-800 text-[11px] font-mono text-zinc-500">
            Checkpoint: <code>models/xlmr_sentiment_model</code>
          </div>
        </div>

        {/* Path C */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-amber-400 uppercase tracking-wider">Path C • Affect</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">28-D Sigmoid</span>
          </div>
          <h3 className="text-sm font-bold text-white">Auxiliary Emotion Affect</h3>
          <p className="text-xs text-zinc-400">
            Raw Text → Fine-Tuned XLM-R GoEmotions → 28 Independent Sigmoids (Multi-label, no softmax sum constraint).
          </p>
          <div className="pt-2 border-t border-zinc-800 text-[11px] font-mono text-zinc-500">
            Checkpoint: <code>models/xlmr_emotion_model</code>
          </div>
        </div>

        {/* Path D */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-3 md:col-span-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Path D • Flagship Multimodal</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800 text-emerald-300 font-mono">79.62% Acc • 78.64% F1</span>
          </div>
          <h3 className="text-sm font-bold text-white">Five-Source Heterogeneous Feature Fusion (2,847-D)</h3>
          <p className="text-xs text-zinc-400">
            User Text + Uploaded Image → 5 Parallel Feature Extractors (768-D + 512-D + 1536-D + 3-D + 28-D) → StandardScaler → MultimodalFusionMLP (Mish activation, 4 layers) → Final Sarcasm Classification.
          </p>
          <div className="pt-2 border-t border-zinc-800 text-[11px] font-mono text-zinc-500">
            Checkpoint: <code>models/multimodal_fusion/fusion_model.pt</code>
          </div>
        </div>

        {/* Path E */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-purple-400 uppercase tracking-wider">Path E • Explainability</span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">0 MB GPU Delta</span>
          </div>
          <h3 className="text-sm font-bold text-white">Local XAI Reasoner Engine</h3>
          <p className="text-xs text-zinc-400">
            Frozen Prediction + InternVL Scene Understanding + Affect Signals → Local Ollama Llama 3.2 1B (CPU) or Cognitive Reasoner fallback → 5-part structured explanation.
          </p>
          <div className="pt-2 border-t border-zinc-800 text-[11px] font-mono text-zinc-500">
            Service: <code>backend/app/services/explanation_service.py</code>
          </div>
        </div>
      </div>

      {/* Interactive Visual Schematic */}
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 sm:p-8 space-y-6">
        <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
          <span className="text-xs font-semibold text-zinc-400 uppercase tracking-widest">
            Interactive Modular Pipeline Explorer (Click Any Module)
          </span>
          <span className="text-xs text-emerald-400 font-mono">All 7 Components Operational</span>
        </div>

        {/* Dual Input Tier */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="p-4 bg-zinc-950 border border-zinc-800 rounded-xl text-center">
            <div className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Input Modality A</div>
            <div className="text-sm font-bold text-white mt-1">Raw User Text Sequence</div>
            <div className="text-xs text-zinc-400 font-mono mt-0.5">Max 128 Tokenized Subwords</div>
          </div>
          <div className="p-4 bg-zinc-950 border border-zinc-800 rounded-xl text-center">
            <div className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Input Modality B</div>
            <div className="text-sm font-bold text-white mt-1">Uploaded Social Media Image</div>
            <div className="text-xs text-zinc-400 font-mono mt-0.5">JPEG / PNG / WebP / Meme</div>
          </div>
        </div>

        {/* 5 Parallel Branches */}
        <div>
          <div className="text-center text-[11px] text-zinc-500 uppercase tracking-wider font-semibold mb-3">
            Parallel Feature Extraction Tier (2,847 Cumulative Dimensions)
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {components.slice(0, 5).map((comp) => {
              const Icon = comp.icon;
              return (
                <button
                  key={comp.id}
                  onClick={() => setSelected(comp)}
                  className="bg-zinc-950 p-3.5 border border-zinc-800 hover:border-zinc-600 rounded-xl text-left transition-all group flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="p-1.5 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-200 group-hover:text-white">
                        <Icon className="w-4 h-4" />
                      </div>
                      <StatusBadge status="ready" />
                    </div>
                    <div className="text-xs font-bold text-white line-clamp-1">{comp.name}</div>
                    <div className="text-[11px] font-mono text-emerald-400 font-bold mt-1">{comp.dimensions}</div>
                  </div>
                  <div className="mt-3 pt-2 border-t border-zinc-800/80 text-[10px] text-zinc-500">
                    Click to inspect details
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Fusion Convergence Section */}
        <div className="pt-2 flex flex-col items-center">
          <div className="w-px h-6 bg-zinc-700" />

          <button
            onClick={() => open('fusion')}
            className="w-full max-w-2xl bg-zinc-950 p-4 border border-zinc-700 hover:border-emerald-500/80 rounded-xl text-center transition-all group"
          >
            <div className="flex items-center justify-center gap-2 mb-1.5">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              <span className="text-sm font-bold text-white">Concatenated Feature Fusion (2,847 Dimensions)</span>
              <StatusBadge status="ready" label="OPERATIONAL" />
            </div>
            <div className="text-xs text-zinc-400 font-mono">
              [ XLM-R: 768-D ] + [ CLIP: 512-D ] + [ InternVL3-2B: 1536-D ] + [ Sentiment: 3-D ] + [ Emotion: 28-D ]
            </div>
            <div className="mt-2 text-[11px] text-emerald-400 font-semibold">
              4-Layer Mish MLP (2847 → 1024 → 512 → 128 → 2) • 79.62% Test Accuracy
            </div>
          </button>

          <div className="w-px h-6 bg-zinc-700" />

          {/* Final Output + XAI Split */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-2xl">
            <div className="px-5 py-3 border border-emerald-800/80 rounded-xl text-center bg-emerald-950/30">
              <div className="text-[10px] text-emerald-400 uppercase tracking-widest font-semibold">Primary Output</div>
              <div className="text-xs font-bold text-white mt-0.5">Calibrated Sarcasm Probabilities</div>
              <div className="text-[11px] text-zinc-400 font-mono mt-0.5">Authoritative Classifier Verdict</div>
            </div>
            <button
              onClick={() => open('explainer')}
              className="px-5 py-3 border border-zinc-800 hover:border-zinc-600 rounded-xl text-center bg-zinc-950 transition-colors"
            >
              <div className="text-[10px] text-purple-400 uppercase tracking-widest font-semibold">XAI Output</div>
              <div className="text-xs font-bold text-white mt-0.5">Structured Natural Language Reasoner</div>
              <div className="text-[11px] text-zinc-400 font-mono mt-0.5">Ollama Llama 3.2 1B (CPU Isolated)</div>
            </button>
          </div>
        </div>
      </div>

      {/* Feature Composition Table */}
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6">
        <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest mb-4">
          Dimensional Composition & Parameter Budget (Total: 2,847 Dimensions)
        </h3>
        <div className="space-y-3 font-mono">
          {[
            { name: 'XLM-R Contextual Text Embedding', dim: 768, pct: (768/2847)*100, role: 'Masked mean-pooled sequence semantics' },
            { name: 'OpenAI CLIP Visual Representation', dim: 512, pct: (512/2847)*100, role: 'Global object presence & visual concepts' },
            { name: 'InternVL3-2B Semantic Vision-Language', dim: 1536, pct: (1536/2847)*100, role: 'Fine-grained dynamic visual details & scene text' },
            { name: 'XLM-R Sentiment Probability Vector', dim: 3, pct: (3/2847)*100, role: 'Negative / Neutral / Positive valence' },
            { name: 'XLM-R GoEmotions Affect Vector', dim: 28, pct: (28/2847)*100, role: '28 independent multi-label affect probabilities' },
          ].map((item, idx) => (
            <div key={idx} className="space-y-1">
              <div className="flex justify-between items-center text-xs">
                <div className="flex items-center gap-2">
                  <span className="font-sans font-medium text-zinc-200">{item.name}</span>
                  <span className="text-[10px] text-zinc-500 font-sans">({item.role})</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-zinc-200 font-bold">{item.dim} D</span>
                  <span className="text-[10px] text-zinc-500 w-12 text-right">{item.pct.toFixed(1)}%</span>
                </div>
              </div>
              <div className="h-1.5 rounded-full bg-zinc-800 overflow-hidden">
                <div style={{ width: `${item.pct}%` }} className="h-full rounded-full bg-emerald-400" />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Modal Details Inspection */}
      <AnimatePresence>
        {selected && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-zinc-900 border border-zinc-700 rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl relative"
            >
              <button
                onClick={() => setSelected(null)}
                className="absolute top-4 right-4 p-1.5 text-zinc-400 hover:text-white rounded-lg hover:bg-zinc-800 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>

              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-zinc-800 border border-zinc-700 text-white">
                  <selected.icon className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-white text-base">{selected.name}</h3>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-xs font-mono text-emerald-400 font-bold">{selected.dimensions}</span>
                    <span className="text-zinc-600">•</span>
                    <span className="text-[11px] text-zinc-400 font-mono">{selected.hardware}</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3 text-xs pt-2 border-t border-zinc-800">
                <div>
                  <span className="text-zinc-500 uppercase text-[10px] font-semibold block mb-0.5">Architectural Purpose</span>
                  <p className="text-zinc-300 leading-relaxed">{selected.purpose}</p>
                </div>
                <div className="bg-zinc-950 p-2.5 rounded-lg border border-zinc-800 font-mono">
                  <span className="text-zinc-500 block text-[10px] uppercase font-sans font-semibold">Checkpoint Artifact</span>
                  <span className="text-zinc-200 text-[11px] break-all">{selected.checkpoint}</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-zinc-950 p-2.5 rounded-lg border border-zinc-800">
                    <span className="text-zinc-500 block text-[10px] uppercase font-semibold">Input Specification</span>
                    <span className="text-zinc-300 text-[11px]">{selected.input}</span>
                  </div>
                  <div className="bg-zinc-950 p-2.5 rounded-lg border border-zinc-800">
                    <span className="text-zinc-500 block text-[10px] uppercase font-semibold">Output Tensor</span>
                    <span className="text-zinc-300 text-[11px]">{selected.output}</span>
                  </div>
                </div>
              </div>

              <div className="pt-2 flex justify-end">
                <button
                  onClick={() => setSelected(null)}
                  className="px-4 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-white text-xs font-semibold transition-colors"
                >
                  Close Inspection
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
