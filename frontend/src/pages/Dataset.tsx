import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Database, Layers, MessageSquare, Zap, ShieldCheck, CheckCircle2,
  PieChart as PieIcon, BarChart3, Globe, HardDrive, FileSpreadsheet
} from 'lucide-react';
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid
} from 'recharts';
import { getDatasetStats } from '../services/api';
import type { DatasetStatsResponse } from '../types';

type DatasetTab = 'text' | 'multimodal' | 'sentiment' | 'emotion';

export function Dataset() {
  const [activeTab, setActiveTab] = useState<DatasetTab>('text');
  const [stats, setStats] = useState<DatasetStatsResponse | null>(null);

  useEffect(() => {
    getDatasetStats()
      .then((data) => setStats(data))
      .catch((err) => console.error('Failed to fetch dataset stats:', err));
  }, []);

  const tabs: { id: DatasetTab; label: string; icon: React.ElementType; count: string }[] = [
    { id: 'text', label: 'Text Sarcasm', icon: Database, count: '1,005,167' },
    { id: 'multimodal', label: 'Multimodal Sarcasm', icon: Layers, count: '33,077' },
    { id: 'sentiment', label: 'Sentiment Corpus', icon: MessageSquare, count: '132,324' },
    { id: 'emotion', label: 'GoEmotions Corpus', icon: Zap, count: '54,263' },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2 mb-2">
          <div className="p-2 rounded-lg bg-zinc-800 border border-zinc-700">
            <Database className="w-5 h-5 text-zinc-200" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Research Datasets & Distribution Diagnostics</h1>
        </div>
        <p className="text-zinc-400 text-sm">
          Audited corpus specifications, data splits, and balanced annotations across 1,190,000+ verified records.
        </p>

        {/* Global Dataset Verification Banner */}
        <div className="mt-3 flex items-center justify-between px-3.5 py-2.5 rounded-xl bg-zinc-900 border border-zinc-800 text-xs text-zinc-300">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>
              <strong>Zero-Leakage Assurance:</strong> All 4 master datasets under <code>FINAL_DATASETS</code> have 0 missing values, verified splits, and 0 train/test leakage.
            </span>
          </div>
          <span className="font-mono font-bold text-emerald-400 text-xs hidden sm:block">Total Records: 1,224,831</span>
        </div>
      </motion.div>

      {/* Tabs */}
      <div className="flex flex-wrap gap-2 border-b border-zinc-800 pb-3">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all ${
                isActive
                  ? 'bg-zinc-100 text-zinc-950 shadow-md'
                  : 'bg-zinc-900/80 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-850 border border-zinc-800'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
              <span className={`text-[10px] px-1.5 py-0.2 rounded-md font-mono ${
                isActive ? 'bg-zinc-300 text-zinc-900' : 'bg-zinc-800 text-zinc-400'
              }`}>
                {tab.count}
              </span>
            </button>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        {/* ── TAB 1: TEXT SARCASM DATASET ── */}
        {activeTab === 'text' && (
          <motion.div
            key="text"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="space-y-6"
          >
            {/* Overview Card */}
            <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-zinc-800 pb-4 mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <span>Text Sarcasm Master Corpus</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">108.59 MB</span>
                  </h2>
                  <p className="text-zinc-400 text-xs mt-0.5">
                    <code>FINAL_DATASETS\TEXT_SARCASM\text_sarcasm_combined.csv</code> • Primary corpus for standalone XLM-R sequence classifier.
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-mono font-extrabold text-white">1,005,167</div>
                  <div className="text-[10px] text-zinc-500 uppercase tracking-widest">Total Verified Rows</div>
                </div>
              </div>

              {/* Splits Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-center">
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Train Set (80.8%)</span>
                  <div className="text-xl font-mono font-bold text-white mt-1">812,130</div>
                  <span className="text-[11px] text-zinc-400">403,476 Non-Sarc / 408,654 Sarc</span>
                </div>
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Validation Set (9.6%)</span>
                  <div className="text-xl font-mono font-bold text-white mt-1">96,518</div>
                  <span className="text-[11px] text-zinc-400">47,936 Non-Sarc / 48,582 Sarc</span>
                </div>
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Holdout Test Set (9.6%)</span>
                  <div className="text-xl font-mono font-bold text-emerald-400 mt-1">96,519</div>
                  <span className="text-[11px] text-zinc-400">47,936 Non-Sarc / 48,583 Sarc</span>
                </div>
              </div>
            </div>

            {/* Class Balance & Sources Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Class Balance */}
              <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6">
                <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-4">
                  Class Balance (Near 50:50 Symmetry)
                </h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-zinc-300 font-medium">Non-Sarcastic (Label 0)</span>
                      <span className="font-mono text-zinc-200">499,348 (49.68%)</span>
                    </div>
                    <div className="h-3 w-full bg-zinc-800 rounded-full overflow-hidden">
                      <div className="h-full bg-zinc-500 rounded-full" style={{ width: '49.68%' }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-zinc-300 font-medium">Sarcastic (Label 1)</span>
                      <span className="font-mono text-emerald-400">505,819 (50.32%)</span>
                    </div>
                    <div className="h-3 w-full bg-zinc-800 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-500 rounded-full" style={{ width: '50.32%' }} />
                    </div>
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-zinc-800 text-xs text-zinc-400">
                  <p>
                    <strong>Balanced Representation:</strong> The 49.68% to 50.32% balance ensures classifier decision thresholds remain unskewed by class imbalance priors.
                  </p>
                </div>
              </div>

              {/* Source Distribution */}
              <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6">
                <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-4">
                  Corpus Source & Linguistic Breakdown
                </h3>
                <div className="space-y-2.5 font-mono text-xs">
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-300">SARC (Self-Annotated Reddit Corpus)</span>
                    <span className="text-white font-bold">956,097 (95.1%)</span>
                  </div>
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-300">GenZ-SarC (Slang & Internet Corpus)</span>
                    <span className="text-amber-300 font-bold">19,984 (2.0%)</span>
                  </div>
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-300">Multilingual Synthetic Code-Mix</span>
                    <span className="text-cyan-300 font-bold">19,999 (2.0%)</span>
                  </div>
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-300">Sarcasm Corpus V2</span>
                    <span className="text-zinc-400 font-bold">9,087 (0.9%)</span>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-zinc-800 flex justify-between text-[11px] text-zinc-500">
                  <span>English: 985,168 (98.0%)</span>
                  <span>Tamil-Eng: 6,680</span>
                  <span>Kannada-Eng: 6,660</span>
                  <span>Malayalam-Eng: 6,659</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* ── TAB 2: MULTIMODAL SARCASM DATASET ── */}
        {activeTab === 'multimodal' && (
          <motion.div
            key="multimodal"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="space-y-6"
          >
            <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-zinc-800 pb-4 mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <span>Multimodal Image-Text Benchmark Dataset</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800 text-emerald-300 font-mono">
                      33,077 Images Verified
                    </span>
                  </h2>
                  <p className="text-zinc-400 text-xs mt-0.5">
                    <code>FINAL_DATASETS\MULTIMODAL_SARCASM\multimodal_sarcasm_combined.csv</code> • Paired text and physical images from MMSD 2.0 & MEMOTION 3.0.
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-mono font-extrabold text-white">33,077</div>
                  <div className="text-[10px] text-zinc-500 uppercase tracking-widest">Image-Text Pairs</div>
                </div>
              </div>

              {/* Splits Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-center">
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Train Set (76.7%)</span>
                  <div className="text-xl font-mono font-bold text-white mt-1">25,384</div>
                  <span className="text-[11px] text-zinc-400">11,189 Non-Sarc / 14,195 Sarc</span>
                </div>
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Validation Set (11.8%)</span>
                  <div className="text-xl font-mono font-bold text-white mt-1">3,891</div>
                  <span className="text-[11px] text-zinc-400">1,713 Non-Sarc / 2,178 Sarc</span>
                </div>
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Untouched Test Split (11.5%)</span>
                  <div className="text-xl font-mono font-bold text-emerald-400 mt-1">3,802</div>
                  <span className="text-[11px] text-zinc-400">1,666 Non-Sarc / 2,136 Sarc</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Feature Stores Status */}
              <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 space-y-3">
                <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                  Pre-Extracted Feature Stores (33,077 × Each Branch)
                </h3>
                <div className="space-y-2 text-xs font-mono">
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-200">1. XLM-R Contextual Text (768-D)</span>
                    <span className="text-emerald-400 font-bold">33,077 .npy files (100%)</span>
                  </div>
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-200">2. OpenAI CLIP Visual (512-D)</span>
                    <span className="text-emerald-400 font-bold">33,077 .npy files (100%)</span>
                  </div>
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-200">3. InternVL3-2B Semantic (1,536-D)</span>
                    <span className="text-emerald-400 font-bold">33,077 .npy files (100%)</span>
                  </div>
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-200">4. Sentiment Polarity Vectors (3-D)</span>
                    <span className="text-emerald-400 font-bold">33,077 .npy files (100%)</span>
                  </div>
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-200">5. GoEmotions Affect Vectors (28-D)</span>
                    <span className="text-emerald-400 font-bold">33,077 .npy files (100%)</span>
                  </div>
                </div>
              </div>

              {/* Source Distribution */}
              <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6">
                <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-4">
                  Source Dataset Attribution
                </h3>
                <div className="space-y-3 font-mono text-xs">
                  <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-sans font-bold text-white">MMSD 2.0 (Multi-Modal Sarcasm Detection)</span>
                      <span className="text-emerald-400 font-bold">24,635 (74.5%)</span>
                    </div>
                    <p className="text-[11px] text-zinc-500 font-sans">
                      Cleaned and re-annotated social media multimodal sarcasm benchmark resolving original MMSD 1.0 spurious label leaks.
                    </p>
                  </div>
                  <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-sans font-bold text-white">MEMOTION 3.0 (Meme Emotion & Sarcasm)</span>
                      <span className="text-amber-300 font-bold">8,442 (25.5%)</span>
                    </div>
                    <p className="text-[11px] text-zinc-500 font-sans">
                      Humorous internet memes exhibiting high visual-text dissonance, sarcastic irony, and visual punchlines.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* ── TAB 3: SENTIMENT DATASET ── */}
        {activeTab === 'sentiment' && (
          <motion.div
            key="sentiment"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="space-y-6"
          >
            <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-zinc-800 pb-4 mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <span>3-Class Sentiment Master Dataset</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">132,324 Rows</span>
                  </h2>
                  <p className="text-zinc-400 text-xs mt-0.5">
                    <code>FINAL_DATASETS\SENTIMENT\sentiment_combined.csv</code> • Used for training the auxiliary 3-D sentiment vector model.
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-mono font-extrabold text-white">132,324</div>
                  <div className="text-[10px] text-zinc-500 uppercase tracking-widest">Total Samples</div>
                </div>
              </div>

              {/* Splits */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-center">
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Train Set</span>
                  <div className="text-xl font-mono font-bold text-white mt-1">107,678</div>
                  <span className="text-[11px] text-zinc-400">81.4% of Corpus</span>
                </div>
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Validation Set</span>
                  <div className="text-xl font-mono font-bold text-white mt-1">7,916</div>
                  <span className="text-[11px] text-zinc-400">6.0% of Corpus</span>
                </div>
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Holdout Test Set</span>
                  <div className="text-xl font-mono font-bold text-emerald-400 mt-1">16,730</div>
                  <span className="text-[11px] text-zinc-400">12.6% of Corpus</span>
                </div>
              </div>
            </div>

            {/* Class Breakdown & Sources */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 space-y-3">
                <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
                  Sentiment Distribution Across 3 Classes
                </h3>
                <div className="space-y-3 font-mono text-xs">
                  <div className="flex justify-between items-center p-3 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-emerald-400 font-bold">Positive Polarity</span>
                    <span className="text-white font-bold">61,810 (46.71%)</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-300 font-bold">Neutral Polarity</span>
                    <span className="text-white font-bold">45,405 (34.31%)</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-rose-400 font-bold">Negative Polarity</span>
                    <span className="text-white font-bold">25,109 (18.98%)</span>
                  </div>
                </div>
              </div>

              <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 space-y-2">
                <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
                  Underlying Sources
                </h3>
                <div className="space-y-2 font-mono text-xs">
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-300">TweetEval Sentiment</span>
                    <span className="text-white font-bold">59,305 (44.8%)</span>
                  </div>
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-300">SentiMix (Hinglish / Spanglish)</span>
                    <span className="text-white font-bold">38,019 (28.7%)</span>
                  </div>
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-300">DravidianCodeMix</span>
                    <span className="text-white font-bold">23,000 (17.4%)</span>
                  </div>
                  <div className="flex justify-between items-center p-2.5 bg-zinc-950 rounded-lg border border-zinc-800">
                    <span className="font-sans text-zinc-300">Synthetic Social Slang</span>
                    <span className="text-zinc-400 font-bold">12,000 (9.1%)</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* ── TAB 4: EMOTION DATASET ── */}
        {activeTab === 'emotion' && (
          <motion.div
            key="emotion"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="space-y-6"
          >
            <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-zinc-800 pb-4 mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <span>Google GoEmotions Master Dataset</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 font-mono">28 Fine-Grained Classes</span>
                  </h2>
                  <p className="text-zinc-400 text-xs mt-0.5">
                    <code>FINAL_DATASETS\EMOTION\emotion_combined.csv</code> • Multi-label fine-grained human affect corpus.
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-mono font-extrabold text-white">54,263</div>
                  <div className="text-[10px] text-zinc-500 uppercase tracking-widest">Total Utterances</div>
                </div>
              </div>

              {/* Splits */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-center">
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Train Set</span>
                  <div className="text-xl font-mono font-bold text-white mt-1">43,410</div>
                  <span className="text-[11px] text-zinc-400">80.0% of Corpus</span>
                </div>
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Validation Set</span>
                  <div className="text-xl font-mono font-bold text-white mt-1">5,426</div>
                  <span className="text-[11px] text-zinc-400">10.0% of Corpus</span>
                </div>
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-800">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Holdout Test Set</span>
                  <div className="text-xl font-mono font-bold text-emerald-400 mt-1">5,427</div>
                  <span className="text-[11px] text-zinc-400">10.0% of Corpus</span>
                </div>
              </div>
            </div>

            {/* Multi-Label Properties */}
            <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 space-y-4">
              <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                Multi-Label Annotation Properties & Taxonomy
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-center text-xs">
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <span className="text-zinc-500 text-[10px] uppercase">Unique Classes</span>
                  <div className="text-lg font-bold text-white font-mono mt-0.5">28</div>
                  <span className="text-[10px] text-zinc-400">27 emotions + neutral</span>
                </div>
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <span className="text-zinc-500 text-[10px] uppercase">Multi-Label Utterances</span>
                  <div className="text-lg font-bold text-amber-400 font-mono mt-0.5">8,817</div>
                  <span className="text-[10px] text-zinc-400">16.2% have &gt; 1 label</span>
                </div>
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <span className="text-zinc-500 text-[10px] uppercase">Duplicate Comment Strings</span>
                  <div className="text-lg font-bold text-zinc-300 font-mono mt-0.5">270</div>
                  <span className="text-[10px] text-zinc-400">Annotator variance (No leak)</span>
                </div>
              </div>

              {/* Taxonomy Cloud */}
              <div className="pt-2">
                <span className="text-[11px] text-zinc-400 block mb-2 font-semibold">28-Class GoEmotions Label Vocabulary:</span>
                <div className="flex flex-wrap gap-1.5">
                  {[
                    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
                  ].map((label) => (
                    <span
                      key={label}
                      className="px-2 py-1 rounded bg-zinc-950 border border-zinc-800 text-[11px] text-zinc-300 font-mono capitalize"
                    >
                      {label}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
