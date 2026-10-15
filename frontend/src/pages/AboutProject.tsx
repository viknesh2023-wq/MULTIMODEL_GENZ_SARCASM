import { motion } from 'framer-motion';
import { Info, Cpu, Code2, Award, ShieldCheck, Terminal } from 'lucide-react';

export function AboutProject() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2 mb-2">
          <div className="p-2 rounded-lg bg-zinc-800 border border-zinc-700">
            <Info className="w-5 h-5 text-zinc-200" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">About The Research Project</h1>
        </div>
        <p className="text-zinc-400 text-sm">
          Technical specifications, system environment, and project metadata for Review 2 & Thesis presentation.
        </p>
      </motion.div>

      {/* Main Metadata Banner */}
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 border-l-4 border-l-zinc-400 space-y-4">
        <div className="text-[10px] font-semibold text-zinc-400 uppercase tracking-widest">
          Project Title & Domain
        </div>
        <h2 className="text-lg font-bold text-white">
          "A Multimodal Vision-Language Framework for Gen-Z Sarcasm Detection using XLM-R, CLIP and Vision-Language Models"
        </h2>
        <div className="flex flex-wrap gap-2 pt-1">
          {['Natural Language Processing', 'Multimodal AI', 'Vision-Language Models', 'Affective Computing', 'Sarcasm Detection'].map((tag, i) => (
            <span key={i} className="px-2.5 py-1 rounded-md bg-zinc-950 border border-zinc-800 text-xs text-zinc-300">
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* Hardware, Environment & Stack */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Hardware */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-xs font-bold text-white uppercase tracking-wider">
            <Cpu className="w-4 h-4 text-zinc-400" />
            <span>Hardware Environment</span>
          </div>
          <div className="space-y-2 text-xs text-zinc-300 font-mono">
            <div className="flex justify-between border-b border-zinc-800 pb-1.5">
              <span className="text-zinc-500">GPU:</span>
              <span className="text-white">NVIDIA RTX 4050 (6GB VRAM)</span>
            </div>
            <div className="flex justify-between border-b border-zinc-800 pb-1.5">
              <span className="text-zinc-500">CUDA:</span>
              <span>CUDA 12.8 / PyTorch cu128</span>
            </div>
            <div className="flex justify-between border-b border-zinc-800 pb-1.5">
              <span className="text-zinc-500">Python:</span>
              <span>Python 3.12.10</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500">OS:</span>
              <span>Windows 11 x64</span>
            </div>
          </div>
        </div>

        {/* Frameworks & Libraries */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-xs font-bold text-white uppercase tracking-wider">
            <Code2 className="w-4 h-4 text-zinc-400" />
            <span>Frameworks & Libraries</span>
          </div>
          <div className="space-y-2 text-xs text-zinc-300 font-mono">
            <div className="flex justify-between border-b border-zinc-800 pb-1.5">
              <span className="text-zinc-500">ML Backend:</span>
              <span>PyTorch 2.11, Transformers 4.52</span>
            </div>
            <div className="flex justify-between border-b border-zinc-800 pb-1.5">
              <span className="text-zinc-500">API Server:</span>
              <span>FastAPI 0.141, Uvicorn 0.53</span>
            </div>
            <div className="flex justify-between border-b border-zinc-800 pb-1.5">
              <span className="text-zinc-500">Frontend:</span>
              <span>React 18 + Vite + TypeScript</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500">Styling/Charts:</span>
              <span>Tailwind CSS + Recharts</span>
            </div>
          </div>
        </div>

        {/* Current State vs Planned */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-xs font-bold text-white uppercase tracking-wider">
            <Award className="w-4 h-4 text-zinc-400" />
            <span>All 6 Milestones Completed</span>
          </div>
          <div className="space-y-1.5 text-xs">
            <div className="flex items-center gap-2 text-emerald-400">
              <ShieldCheck className="w-3.5 h-3.5 flex-shrink-0" />
              <span>M1: Data Pipeline (1.005M text + 33k multimodal)</span>
            </div>
            <div className="flex items-center gap-2 text-emerald-400">
              <ShieldCheck className="w-3.5 h-3.5 flex-shrink-0" />
              <span>M2: Text Sarcasm (77.19% Acc, 96,519 test)</span>
            </div>
            <div className="flex items-center gap-2 text-emerald-400">
              <ShieldCheck className="w-3.5 h-3.5 flex-shrink-0" />
              <span>M3: Sentiment Model (73.32% Acc, 16,730 test)</span>
            </div>
            <div className="flex items-center gap-2 text-emerald-400">
              <ShieldCheck className="w-3.5 h-3.5 flex-shrink-0" />
              <span>M4: Emotion Model (49.26% Micro F1, 28 classes)</span>
            </div>
            <div className="flex items-center gap-2 text-emerald-400">
              <ShieldCheck className="w-3.5 h-3.5 flex-shrink-0" />
              <span>M5: Multimodal Fusion (79.62% Acc, +16.66% net gain)</span>
            </div>
            <div className="flex items-center gap-2 text-emerald-400">
              <ShieldCheck className="w-3.5 h-3.5 flex-shrink-0" />
              <span>M6: Local XAI Reasoner (Llama 3.2 1B CPU)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Directory & Artifact Mapping */}
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 space-y-3">
        <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
          Local Model Checkpoints & Artifact Directory Mapping
        </h3>
        <div className="space-y-2 text-xs font-mono text-zinc-300">
          <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 flex justify-between items-center">
            <span className="text-zinc-500">Text Sarcasm Model:</span>
            <span className="text-zinc-200">D:\NLP_PROJECT\models\xlmr_text_sarcasm_full</span>
          </div>
          <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 flex justify-between items-center">
            <span className="text-zinc-500">Sentiment Model:</span>
            <span className="text-zinc-200">D:\NLP_PROJECT\models\xlmr_sentiment_model</span>
          </div>
          <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 flex justify-between items-center">
            <span className="text-zinc-500">Emotion Model:</span>
            <span className="text-zinc-200">D:\NLP_PROJECT\models\xlmr_emotion_model</span>
          </div>
          <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 flex justify-between items-center">
            <span className="text-zinc-500">Multimodal Fusion & Ablations:</span>
            <span className="text-zinc-200">D:\NLP_PROJECT\models\multimodal_fusion</span>
          </div>
          <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 flex justify-between items-center">
            <span className="text-zinc-500">Backend API Service:</span>
            <span className="text-zinc-200">D:\NLP_PROJECT\backend\app\main.py</span>
          </div>
          <div className="p-2.5 rounded-lg bg-zinc-950 border border-zinc-800 flex justify-between items-center">
            <span className="text-zinc-500">Audit Documentation:</span>
            <span className="text-zinc-200">D:\NLP_PROJECT\PROJECT_AUDIT</span>
          </div>
        </div>
      </div>
    </div>
  );
}
