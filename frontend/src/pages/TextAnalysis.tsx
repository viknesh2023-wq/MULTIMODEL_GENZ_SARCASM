import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Hash, AtSign, Link, Smile, AlignLeft, AlertCircle } from 'lucide-react';
import { analyzeText } from '../utils/textAnalysis';
import type { TextStats } from '../types';

export function TextAnalysis() {
  const [text, setText] = useState('');
  const [stats, setStats] = useState<TextStats | null>(null);

  const handleAnalyze = () => {
    if (text.trim()) setStats(analyzeText(text));
  };

  const statRows = stats ? [
    { icon: AlignLeft, label: 'Characters', value: stats.charCount },
    { icon: FileText, label: 'Words', value: stats.wordCount },
    { icon: FileText, label: 'Sentences', value: stats.sentenceCount },
    { icon: Hash, label: 'Hashtags', value: stats.hashtagCount },
    { icon: AtSign, label: 'Mentions', value: stats.mentionCount },
    { icon: Link, label: 'URLs', value: stats.urlCount },
    { icon: Smile, label: 'Emojis', value: stats.emojiCount },
    { icon: AlertCircle, label: 'Exclamations', value: stats.exclamationCount },
    { icon: AlertCircle, label: 'Question Marks', value: stats.questionCount },
    { icon: FileText, label: 'CAPS Words', value: stats.capsWordCount },
    { icon: AlertCircle, label: 'Repeated Punctuation', value: stats.repeatedPunctCount },
  ] : [];

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2 mb-2">
          <div className="p-2 rounded-lg bg-zinc-800 border border-zinc-700">
            <FileText className="w-5 h-5 text-zinc-200" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Text Analysis & Surface Statistics</h1>
        </div>
        <p className="text-zinc-400 text-sm">Deterministic linguistic statistics — these are NOT machine learning predictions.</p>
        <div className="mt-2 flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-900 border border-zinc-800 text-xs text-zinc-400">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 text-zinc-500" />
          All statistics below are computed using rule-based parsing algorithms, not model inferences.
        </div>
      </motion.div>

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }} className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6">
        <textarea
          value={text}
          onChange={e => { setText(e.target.value); setStats(null); }}
          placeholder="Paste your text here to analyze its linguistic properties..."
          rows={6}
          className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 transition-colors resize-none font-sans"
        />
        <div className="flex justify-end mt-3">
          <button
            onClick={handleAnalyze}
            disabled={!text.trim()}
            className="px-5 py-2 rounded-lg bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-bold transition-all disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Compute Statistics
          </button>
        </div>
      </motion.div>

      <AnimatePresence>
        {stats && (
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
            {/* Stats grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {statRows.map((row, i) => {
                const Icon = row.icon;
                return (
                  <motion.div
                    key={row.label}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.03 }}
                    className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 card-hover"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Icon className="w-3.5 h-3.5 text-zinc-500" />
                      <span className="text-xs text-zinc-400 font-medium">{row.label}</span>
                    </div>
                    <div className="text-2xl font-bold font-mono text-white">{row.value}</div>
                  </motion.div>
                );
              })}
            </div>

            {/* Possible slang */}
            {stats.possibleSlang.length > 0 && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
                <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">Detected Internet / Slang Keywords</h3>
                <div className="flex flex-wrap gap-2">
                  {stats.possibleSlang.map(s => (
                    <span key={s} className="px-2.5 py-1 rounded-md bg-zinc-800 border border-zinc-700 text-zinc-200 text-xs font-mono">
                      {s}
                    </span>
                  ))}
                </div>
                <p className="text-[11px] text-zinc-500 mt-3">Note: Lexicon match only; does not signify semantic intent.</p>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
