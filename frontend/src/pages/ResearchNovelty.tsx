import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { BookOpen, Sparkles, AlertTriangle, GitCompare, CheckCircle2, ArrowRight } from 'lucide-react';

export function ResearchNovelty() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2 mb-2">
          <div className="p-2 rounded-lg bg-zinc-800 border border-zinc-700">
            <BookOpen className="w-5 h-5 text-zinc-200" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Research Contribution & Novelty</h1>
        </div>
        <p className="text-zinc-400 text-sm">
          Theoretical motivation, existing paradigm limitations, and the proposed modular heterogeneous framework.
        </p>
      </motion.div>

      {/* Research Problem Statement */}
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 border-l-4 border-l-zinc-400 space-y-3">
        <div className="text-[10px] font-semibold text-zinc-400 uppercase tracking-widest">Research Problem</div>
        <h2 className="text-base font-bold text-white">
          Cross-Modal Incongruity in Gen-Z & Multimodal Sarcasm
        </h2>
        <p className="text-xs text-zinc-300 leading-relaxed">
          Sarcasm is an affective phenomenon characterized by a disparity between literal expression and intended pragmatic meaning. In modern digital communication (memes, social feeds, short-form posts), this incongruity manifests across textual polarity, fine-grained emotional tones, visual semantics, and high-level vision-language contextual cues.
        </p>
      </div>

      {/* Prior Works vs Proposed Contribution Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Existing Approaches */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2 text-xs font-bold text-zinc-300 uppercase tracking-wider">
            <GitCompare className="w-4 h-4 text-zinc-400" />
            <span>Existing Methodologies in Literature</span>
          </div>

          <div className="space-y-3 text-xs text-zinc-400">
            <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1">
              <strong className="text-zinc-200">Text-Only Sarcasm Classification:</strong>
              <p>BERT / RoBERTa models fail when the text is literally positive but visually contradicted.</p>
            </div>

            <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1">
              <strong className="text-zinc-200">Standard Cross-Modal Attention:</strong>
              <p>Direct token-to-patch attention suffers from high visual noise and over-parameterization.</p>
            </div>

            <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1">
              <strong className="text-zinc-200">Homogeneous CLIP-only Embeddings:</strong>
              <p>CLIP produces coarse alignment but lacks nuanced 28-class emotional granularity.</p>
            </div>
          </div>
        </div>

        {/* Proposed Contribution */}
        <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2 text-xs font-bold text-white uppercase tracking-wider">
            <Sparkles className="w-4 h-4 text-zinc-300" />
            <span>Proposed Heterogeneous Framework</span>
          </div>

          <div className="space-y-3 text-xs text-zinc-300">
            <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1">
              <strong className="text-white">Modular Heterogeneous Multimodal Representation:</strong>
              <p className="text-zinc-400">Combines 5 specialized representation vectors into a unified 2,847-dimensional feature space.</p>
            </div>

            <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1">
              <strong className="text-white">Affective & Contextual Incongruity Modeling:</strong>
              <p className="text-zinc-400">Bridges 3-D sentiment + 28-D GoEmotions with 768-D text and 1536-D InternVL3-2B VLM features.</p>
            </div>

            <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1">
              <strong className="text-white">Scalable Decoupled Extraction:</strong>
              <p className="text-zinc-400">Allows independent pre-extraction (CLIP & InternVL3 extracted for 33k pairs) without training bottlenecks.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Scientific Clarification Note */}
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-2">
        <div className="flex items-center gap-2 text-xs font-bold text-zinc-300 uppercase tracking-wider">
          <AlertTriangle className="w-4 h-4 text-amber-400" />
          <span>Scientific Scope & Novelty Definition</span>
        </div>
        <p className="text-xs text-zinc-400 leading-relaxed">
          The core novelty lies in the <strong>heterogeneous multi-aspect representation and empirical evaluation</strong> of diverse semantic, visual, and affective modalities for sarcasm detection—rather than claiming individual novelty for pretrained backbones (XLM-R, CLIP, InternVL).
        </p>
      </div>

      {/* Heterogeneous Fusion Formulation */}
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 space-y-4">
        <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
          Multimodal Representation Vector Formulation
        </h3>
        <div className="p-4 rounded-lg bg-zinc-950 border border-zinc-800 font-mono text-center text-xs text-zinc-300 overflow-x-auto">
          <span className="text-white font-bold">V_fusion</span> = [
          <span> e_XLM-R (768) </span> ⊕
          <span> e_CLIP (512) </span> ⊕
          <span> e_InternVL3 (1536) </span> ⊕
          <span> p_Sentiment (3) </span> ⊕
          <span> p_Emotion (28) </span>
          ] ∈ ℝ^<span className="text-white font-bold">2847</span>
        </div>
      </div>

      {/* 8-Way Empirical Validation Evidence */}
      <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-bold text-white uppercase tracking-wider">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>Empirical Ablation Benchmark Evidence</span>
          </div>
          <Link
            to="/performance"
            className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center gap-1 transition-colors"
          >
            Inspect full benchmark table <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
        <p className="text-xs text-zinc-400 leading-relaxed">
          The research claims are validated through an exhaustive <strong>8-way ablation benchmark</strong> evaluated on the identical 3,802 multimodal test split. As modalities are added to the text baseline (61.98% Macro F1), performance strictly increases up to the complete 5-source heterogeneous fusion (78.64% Macro F1, a net gain of <strong>+16.66% macro F1</strong>).
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 text-center font-mono text-xs">
          <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
            <div className="text-[10px] text-zinc-500 uppercase">Text Baseline</div>
            <div className="text-sm font-bold text-zinc-300 mt-1">61.98% F1</div>
            <div className="text-[10px] text-zinc-500 mt-0.5">768 Dimensions</div>
          </div>
          <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
            <div className="text-[10px] text-zinc-500 uppercase">+ Sentiment & Emotion</div>
            <div className="text-sm font-bold text-zinc-300 mt-1">65.82% F1</div>
            <div className="text-[10px] text-emerald-400 mt-0.5">+3.84% gain</div>
          </div>
          <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
            <div className="text-[10px] text-zinc-500 uppercase">+ Visual (CLIP + VLM)</div>
            <div className="text-sm font-bold text-zinc-300 mt-1">78.06% F1</div>
            <div className="text-[10px] text-emerald-400 mt-0.5">+16.08% gain</div>
          </div>
          <div className="bg-zinc-950 p-3 rounded-lg border border-emerald-950/60 bg-emerald-950/10">
            <div className="text-[10px] text-emerald-400 uppercase">Full 5-Source Fusion</div>
            <div className="text-sm font-bold text-emerald-300 mt-1">78.64% F1</div>
            <div className="text-[10px] text-emerald-400 mt-0.5">+16.66% net gain</div>
          </div>
        </div>
      </div>
    </div>
  );
}
