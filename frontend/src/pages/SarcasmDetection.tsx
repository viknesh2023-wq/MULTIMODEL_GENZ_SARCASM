import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, X, RotateCcw, ChevronDown } from 'lucide-react';
import { ConfidenceGauge } from '../components/ConfidenceGauge';
import { ProbabilityBars } from '../components/ProbabilityBars';
import { predictText } from '../services/api';
import type { TextPredictResponse } from '../types';

const EXAMPLES = [
  "Oh great, another Monday. Exactly what I needed.",
  "Wow, amazing job breaking the one thing that was working.",
  "I absolutely love waiting three hours for a five minute task.",
  "Yeah, because waiting two hours for a reply is exactly how I wanted to spend my day.",
  "I absolutely love when my WiFi disconnects during an important submission.",
  "Great, my code crashed right before the deadline. Perfect.",
  "Thanks for explaining something I already understood.",
];

export function SarcasmDetection() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TextPredictResponse | null>(null);
  const [measuredLatency, setMeasuredLatency] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showExamples, setShowExamples] = useState(false);
  const resultRef = useRef<HTMLDivElement>(null);

  const handleAnalyze = async () => {
    const trimmed = text.trim();
    if (!trimmed) {
      setError('Please enter text before analyzing.');
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    setMeasuredLatency(null);
    const startTime = performance.now();
    try {
      const res = await predictText(trimmed);
      const elapsed = Math.round(performance.now() - startTime);
      setMeasuredLatency(elapsed);
      setResult(res);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        ?? (err as { message?: string })?.message
        ?? 'Unknown error';
      if (msg.includes('Network') || msg.includes('ECONNREFUSED')) {
        setError('AI backend unavailable. Start the FastAPI server on port 8000.');
      } else {
        setError(`Analysis failed: ${msg}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => { setText(''); setResult(null); setError(null); setMeasuredLatency(null); };
  const handleExample = (ex: string) => { setText(ex); setResult(null); setError(null); setMeasuredLatency(null); setShowExamples(false); };

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2 mb-2">
          <div className="p-2 rounded-lg bg-zinc-800 border border-zinc-700">
            <Zap className="w-5 h-5 text-zinc-200" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Text Sarcasm Detection</h1>
          <span className="ml-2 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            Standalone Text Model
          </span>
        </div>
        <p className="text-zinc-400 text-sm">
          Binary sarcasm classification evaluated directly by the fine-tuned XLM-RoBERTa text model (77.19% Test Accuracy across 96,519 samples).
        </p>
      </motion.div>

      {/* Input panel */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-3">
          <label className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Input Utterance / Post</label>
          <div className="flex items-center gap-2">
            <div className="relative">
              <button
                onClick={() => setShowExamples(!showExamples)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-white hover:bg-zinc-800 border border-zinc-700 transition-colors"
              >
                Load Example <ChevronDown className="w-3.5 h-3.5" />
              </button>
              <AnimatePresence>
                {showExamples && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    className="absolute right-0 top-8 w-80 bg-zinc-900 border border-zinc-700 rounded-xl p-1.5 z-30 shadow-2xl space-y-1"
                  >
                    {EXAMPLES.map((ex, i) => (
                      <button
                        key={i}
                        onClick={() => handleExample(ex)}
                        className="w-full text-left px-3 py-2 text-xs text-zinc-300 hover:text-white hover:bg-zinc-800 rounded-lg transition-colors leading-relaxed"
                      >
                        "{ex}"
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>

        <textarea
          value={text}
          onChange={e => { setText(e.target.value); setError(null); }}
          placeholder="Enter a social media post, comment, message, or sentence..."
          rows={5}
          className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-zinc-500 transition-colors resize-none font-sans leading-relaxed"
          onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleAnalyze(); }}
        />

        <div className="flex items-center justify-between mt-3">
          <div className="text-xs text-zinc-500 font-mono">{text.length} chars • Press Ctrl+Enter to analyze</div>
          <div className="flex gap-2">
            <button
              onClick={handleClear}
              disabled={!text && !result}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs text-zinc-400 hover:text-white hover:bg-zinc-800 border border-zinc-700 transition-colors disabled:opacity-30"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Clear
            </button>
            <button
              onClick={handleAnalyze}
              disabled={loading || !text.trim()}
              className="flex items-center gap-2 px-5 py-2 rounded-lg bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-bold transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? (
                <><span className="w-3.5 h-3.5 border-2 border-zinc-950/30 border-t-zinc-950 rounded-full animate-spin" /> Analyzing...</>
              ) : (
                <><Zap className="w-3.5 h-3.5" /> Analyze Text</>
              )}
            </button>
          </div>
        </div>
      </motion.div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="flex items-center gap-3 p-4 rounded-xl bg-rose-950/40 border border-rose-900/60 text-rose-300 text-xs">
            <X className="w-4 h-4 flex-shrink-0" /> {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div ref={resultRef} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-4">

            {/* Main result */}
            <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-6 sm:p-8">
              <div className="flex flex-col lg:flex-row items-center gap-8">
                {/* Gauge */}
                <div className="flex-shrink-0">
                  <ConfidenceGauge
                    confidence={result.confidence}
                    label={result.label_name.toUpperCase()}
                    isSarcastic={result.label === 1}
                  />
                </div>

                {/* Probability bars + details */}
                <div className="flex-1 w-full space-y-6">
                  <ProbabilityBars
                    nonSarcastic={result.probabilities.non_sarcastic}
                    sarcastic={result.probabilities.sarcastic}
                  />

                  {/* Model info */}
                  <div className="grid grid-cols-2 gap-3 pt-4 border-t border-zinc-800">
                    {[
                      { label: 'Model Backbone', value: result.model },
                      { label: 'Compute Device', value: result.device },
                      { label: 'Task Domain', value: 'Binary Text Sarcasm' },
                      { label: 'Measured Latency', value: measuredLatency ? `${measuredLatency} ms` : 'Sub-50ms' },
                      { label: 'Test Accuracy', value: '77.1941%' },
                      { label: 'Test Macro F1', value: '77.1939%' },
                      { label: 'Training Corpus', value: '1,005,167 samples' },
                      { label: 'Sequence Limit', value: '128 tokens' },
                    ].map(item => (
                      <div key={item.label} className="flex justify-between text-xs">
                        <span className="text-zinc-500">{item.label}</span>
                        <span className="font-mono text-zinc-300 font-medium">{item.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Research Methodology Note: Sample Confidence vs Model Accuracy */}
              <div className="mt-6 pt-4 border-t border-zinc-800 text-[11px] text-zinc-400 bg-zinc-950/60 p-3.5 rounded-lg border border-zinc-800/80 leading-relaxed space-y-1">
                <div className="font-semibold text-zinc-300 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                  Statistical Interpretation Notice
                </div>
                <p>
                  The <strong className="text-zinc-200">sample prediction confidence ({(result.confidence * 100).toFixed(2)}%)</strong> reflects the posterior softmax probability over the final classification logits for this specific utterance. This value is mathematically distinct from the overall <strong className="text-zinc-200">test set accuracy (77.1941%)</strong> evaluated across the 96,519 held-out test split.
                </p>
                <p className="text-zinc-500">
                  Note: Auxiliary sentiment and emotion classifiers are decoupled into dedicated CPU services and are not invoked on the standalone text sarcasm endpoint to preserve single-pass inference latency.
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
