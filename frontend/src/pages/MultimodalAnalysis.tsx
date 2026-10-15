import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Layers, Upload, X, CheckCircle2, Zap, AlertCircle, ChevronDown, ChevronUp,
  Activity, Cpu, ShieldCheck, HelpCircle, BarChart3, Heart, Smile,
  Sparkles, FileText, Eye, Info, MessageSquare
} from 'lucide-react';
import { StatusBadge } from '../components/StatusBadge';
import { ProbabilityBars } from '../components/ProbabilityBars';
import { predictMultimodal, explainPrediction } from '../services/api';
import type {
  MultimodalPredictResponse,
  ExplanationContent,
  ExplanationReasonerMeta
} from '../types';

const PipelineStage = ({
  label, sublabel, status, dimension
}: {
  label: string; sublabel: string; status: 'ready' | 'dev' | 'soon'; dimension?: string;
}) => {
  return (
    <div className="bg-zinc-900 border border-zinc-800 p-3.5 rounded-xl text-center space-y-1">
      <div className="font-semibold text-xs text-zinc-200">{label}</div>
      <div className="text-[11px] text-zinc-500">{sublabel}</div>
      {dimension && <div className="text-xs font-mono text-zinc-300 font-bold">{dimension}</div>}
      <div className="pt-1.5 flex justify-center">
        <StatusBadge status={status} />
      </div>
    </div>
  );
};

const EXAMPLES = [
  "Oh great, another delay. Exactly what I needed today!",
  "Totally love when the printer stops working right before the presentation.",
  "What a wonderful morning for a flat tire on the highway.",
  "Super excited to work on weekends while everyone else relaxes.",
];

export function MultimodalAnalysis() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MultimodalPredictResponse | null>(null);
  const [showAllEmotions, setShowAllEmotions] = useState(false);
  const [showExamples, setShowExamples] = useState(false);
  
  // Explanation State
  const [autoExplain, setAutoExplain] = useState(true);
  const [explaining, setExplaining] = useState(false);
  const [explanation, setExplanation] = useState<ExplanationContent | null>(null);
  const [explanationReasoner, setExplanationReasoner] = useState<ExplanationReasonerMeta | null>(null);
  const [explanationError, setExplanationError] = useState<string | null>(null);

  const resultsRef = useRef<HTMLDivElement>(null);
  const explanationRef = useRef<HTMLDivElement>(null);

  const handleFile = (file: File) => {
    if (!['image/png', 'image/jpeg', 'image/jpg', 'image/webp'].includes(file.type)) {
      setError('Unsupported file type. Please upload a PNG, JPG, JPEG, or WEBP image.');
      return;
    }
    setError(null);
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = e => setImagePreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, []);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const clearImage = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setExplanation(null);
  };

  const handleClearAll = () => {
    setText('');
    setImageFile(null);
    setImagePreview(null);
    setError(null);
    setResult(null);
    setExplanation(null);
    setExplanationError(null);
  };

  const handleLoadExample = (ex: string) => {
    setText(ex);
    setShowExamples(false);
    setError(null);
  };

  const handleAnalyze = async () => {
    const trimmed = text.trim();
    if (!trimmed) {
      setError('Please provide text (post, caption, or comment) before analyzing.');
      return;
    }
    if (!imageFile) {
      setError('Please upload an image or meme to perform multimodal analysis.');
      return;
    }

    setError(null);
    setLoading(true);
    setResult(null);
    setExplanation(null);
    setExplanationError(null);

    try {
      const response = await predictMultimodal(trimmed, imageFile, autoExplain);
      setResult(response);
      if (response.explanation) {
        setExplanation(response.explanation);
        setExplanationReasoner(response.reasoner ?? null);
      }
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 150);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        ?? (err as { message?: string })?.message
        ?? 'Inference error';
      if (msg.includes('Network') || msg.includes('ECONNREFUSED')) {
        setError('FastAPI backend is unavailable. Ensure the backend server is running on port 8000.');
      } else {
        setError(`Multimodal inference failed: ${msg}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSynthesizeExplanation = async () => {
    if (!result) return;
    setExplaining(true);
    setExplanationError(null);

    try {
      const resp = await explainPrediction({
        text: text.trim(),
        prediction: result.prediction,
        sentiment: result.sentiment,
        emotions: result.emotions,
      });

      if (resp.status === 'success' && resp.explanation) {
        setExplanation(resp.explanation);
        setExplanationReasoner(resp.reasoner);
        setTimeout(() => {
          explanationRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
      } else {
        setExplanationError(resp.error ?? 'Failed to synthesize explanation.');
      }
    } catch (err: unknown) {
      const msg = (err as { message?: string })?.message ?? 'Explanation service error';
      setExplanationError(`Reasoner error: ${msg}`);
    } finally {
      setExplaining(false);
    }
  };

  const isFormValid = text.trim().length > 0 && imageFile !== null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2 mb-2">
          <div className="p-2 rounded-lg bg-zinc-800 border border-zinc-700">
            <Layers className="w-5 h-5 text-indigo-400" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Multimodal Sarcasm Analysis</h1>
          <span className="ml-2 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            Production Ready
          </span>
          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-purple-500/10 text-purple-400 border border-purple-500/30">
            Local XAI Reasoner
          </span>
        </div>
        <p className="text-zinc-400 text-sm">
          Real request-time multimodal inference: extracts five distinct feature representations (2,847-D),
          classifies sarcasm via the trained Multimodal Fusion MLP, and provides evidence-grounded local explanations.
        </p>
      </motion.div>

      {/* Error alert banner */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            className="flex items-start gap-3 p-4 rounded-xl bg-red-950/40 border border-red-800/60 text-red-200 text-xs leading-relaxed"
          >
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1 font-medium">{error}</div>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-200">
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Input panel: 2 columns for Text & Image */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Text input */}
        <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                1. Post / Caption Text
              </label>
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setShowExamples(!showExamples)}
                  className="flex items-center gap-1 px-2.5 py-1 rounded text-xs text-zinc-400 hover:text-white bg-zinc-800/80 hover:bg-zinc-800 border border-zinc-700 transition-colors"
                >
                  Examples <ChevronDown className="w-3 h-3" />
                </button>
                <AnimatePresence>
                  {showExamples && (
                    <motion.div
                      initial={{ opacity: 0, y: -6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -6 }}
                      className="absolute right-0 top-7 w-72 bg-zinc-900 border border-zinc-700 rounded-xl p-1 z-30 shadow-2xl space-y-1"
                    >
                      {EXAMPLES.map((ex, i) => (
                        <button
                          key={i}
                          type="button"
                          onClick={() => handleLoadExample(ex)}
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
            <textarea
              value={text}
              onChange={e => { setText(e.target.value); setError(null); }}
              placeholder="Enter text utterance, caption, or social post to evaluate alongside the image..."
              rows={6}
              disabled={loading}
              className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-indigo-500 transition-colors resize-none font-sans leading-relaxed disabled:opacity-50"
            />
          </div>
          <div className="flex justify-between items-center text-xs text-zinc-500 pt-2 font-mono">
            <span>{text.length} characters</span>
            <span>Max sequence: 128 tokens</span>
          </div>
        </motion.div>

        {/* Image input */}
        <motion.div initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.15 }} className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 flex flex-col justify-between">
          <div>
            <label className="text-xs font-semibold text-zinc-300 uppercase tracking-wider mb-2 block">
              2. Visual Meme / Image
            </label>
            {imagePreview ? (
              <div className="relative group">
                <img
                  src={imagePreview}
                  alt="Uploaded preview"
                  className="w-full h-44 object-contain rounded-xl border border-zinc-800 bg-zinc-950 p-1"
                />
                {!loading && (
                  <button
                    type="button"
                    onClick={clearImage}
                    className="absolute top-2 right-2 p-1.5 rounded-full bg-zinc-900/90 text-zinc-400 hover:text-white border border-zinc-700 shadow-md transition-colors"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
                <div className="mt-2 text-xs text-zinc-400 font-mono flex items-center justify-between">
                  <span>{imageFile?.name}</span>
                  <span>{((imageFile?.size ?? 0) / 1024).toFixed(0)} KB</span>
                </div>
              </div>
            ) : (
              <label
                className={`flex flex-col items-center justify-center h-44 rounded-xl border border-dashed cursor-pointer transition-all
                  ${isDragOver ? 'border-indigo-400 bg-indigo-950/20' : 'border-zinc-700 hover:border-zinc-500 bg-zinc-950/80 hover:bg-zinc-950'}`}
                onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={onDrop}
              >
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg,image/webp"
                  onChange={onInputChange}
                  disabled={loading}
                  className="hidden"
                />
                <Upload className="w-7 h-7 text-zinc-500 mb-2" />
                <div className="text-xs text-zinc-200 font-semibold">Drop image here or click to browse</div>
                <div className="text-[11px] text-zinc-500 mt-1">PNG · JPG · JPEG · WEBP (Max 25MB)</div>
              </label>
            )}
          </div>
          <div className="flex items-center justify-between text-xs text-zinc-400 font-mono pt-2">
            <span>CLIP + InternVL3-2B</span>
            <label className="flex items-center gap-1.5 cursor-pointer text-zinc-300 hover:text-white select-none">
              <input
                type="checkbox"
                checked={autoExplain}
                onChange={e => setAutoExplain(e.target.checked)}
                className="rounded border-zinc-700 bg-zinc-800 text-indigo-500 focus:ring-0 focus:ring-offset-0"
              />
              <span className="text-[11px]">Generate XAI explanation</span>
            </label>
          </div>
        </motion.div>
      </div>

      {/* Submission Actions */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="flex gap-3">
        <button
          type="button"
          onClick={handleClearAll}
          disabled={loading || (!text && !imageFile && !result)}
          className="px-4 py-3 rounded-xl border border-zinc-800 bg-zinc-900 text-zinc-400 hover:text-white hover:bg-zinc-800 text-xs font-semibold transition-all disabled:opacity-30 disabled:cursor-not-allowed"
        >
          Reset
        </button>
        <button
          type="button"
          onClick={handleAnalyze}
          disabled={loading || !isFormValid}
          className="flex-1 py-3 px-6 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition-all shadow-lg shadow-indigo-950/50 flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-indigo-600"
        >
          {loading ? (
            <>
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Extracting 5 Feature Branches & Computing 2,847-D Fusion...</span>
            </>
          ) : (
            <>
              <Zap className="w-4 h-4 text-amber-300" />
              <span>Run Multimodal Inference (2,847-D Fusion)</span>
            </>
          )}
        </button>
      </motion.div>

      {/* RESULTS DISPLAY */}
      <AnimatePresence>
        {result && (
          <motion.div
            ref={resultsRef}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 15 }}
            className="space-y-6 pt-2"
          >
            {/* Primary Classification Result Card */}
            <div className={`p-6 rounded-2xl border ${result.prediction.label === 'sarcastic' ? 'bg-gradient-to-br from-purple-950/40 via-zinc-900 to-zinc-900 border-purple-800/50' : 'bg-gradient-to-br from-emerald-950/40 via-zinc-900 to-zinc-900 border-emerald-800/50'}`}>
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-zinc-800">
                <div>
                  <div className="text-[11px] font-mono uppercase tracking-wider text-zinc-400">
                    Multimodal Classification Decision
                  </div>
                  <div className="flex items-center gap-3 mt-1">
                    <span className={`text-2xl font-black uppercase tracking-tight ${result.prediction.label === 'sarcastic' ? 'text-purple-300' : 'text-emerald-300'}`}>
                      {result.prediction.label === 'sarcastic' ? '⚡ Sarcastic' : '✓ Non-Sarcastic'}
                    </span>
                    <span className="px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-zinc-800 border border-zinc-700 text-zinc-300">
                      Class ID: {result.prediction.class_id}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <div className="text-[11px] text-zinc-400 uppercase font-mono">Calibrated Confidence</div>
                    <div className="text-xl font-bold font-mono text-white">
                      {(result.prediction.confidence * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="px-3 py-1.5 rounded-lg bg-zinc-800/80 border border-zinc-700 text-[11px] font-mono text-zinc-300">
                    ⏱ {result.performance.total_latency_ms} ms
                  </div>
                </div>
              </div>

              {/* Class Probability Distribution */}
              <div className="mt-5 space-y-3">
                <div className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                  Calibrated Class Probabilities (Softmax over MLP Logits)
                </div>
                <ProbabilityBars
                  nonSarcastic={result.prediction.probabilities.non_sarcastic}
                  sarcastic={result.prediction.probabilities.sarcastic}
                />
              </div>
            </div>

            {/* Evidence-Grounded Explanation Section (XAI Reasoner) */}
            <div ref={explanationRef} className="bg-zinc-900/90 border border-zinc-800 rounded-2xl p-6 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800 pb-4">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-lg bg-purple-950/60 border border-purple-800/60 text-purple-300">
                    <Sparkles className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <h2 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
                      Evidence-Grounded Explanation
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-purple-900/40 text-purple-300 border border-purple-800/60 font-semibold">
                        XAI Reasoner
                      </span>
                    </h2>
                    <p className="text-[11px] text-zinc-400 mt-0.5">
                      Synthesized from empirical multimodal vectors, lexical contrast, and visual grounding.
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {explanationReasoner && (
                    <span className="px-2.5 py-1 rounded-lg bg-zinc-950 border border-zinc-800 text-[11px] font-mono text-zinc-400">
                      {explanationReasoner.model} {explanationReasoner.latency_ms && `(${explanationReasoner.latency_ms}ms)`}
                    </span>
                  )}
                  {!explanation && (
                    <button
                      type="button"
                      onClick={handleSynthesizeExplanation}
                      disabled={explaining}
                      className="px-3.5 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold transition-all flex items-center gap-1.5 shadow-md shadow-purple-950 disabled:opacity-50"
                    >
                      {explaining ? (
                        <>
                          <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          <span>Synthesizing...</span>
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-3.5 h-3.5 text-amber-300" />
                          <span>Generate Explanation</span>
                        </>
                      )}
                    </button>
                  )}
                </div>
              </div>

              {explanationError && (
                <div className="p-3 rounded-lg bg-red-950/30 border border-red-800/50 text-red-300 text-xs">
                  {explanationError}
                </div>
              )}

              {explaining && (
                <div className="py-8 flex flex-col items-center justify-center space-y-3 text-center">
                  <div className="w-8 h-8 border-2 border-purple-500/30 border-t-purple-500 rounded-full animate-spin" />
                  <div className="text-xs text-zinc-300 font-medium">Synthesizing evidence-grounded explanation on local CPU...</div>
                  <div className="text-[11px] text-zinc-500 font-mono">Running local LLM reasoner without GPU memory overhead</div>
                </div>
              )}

              {explanation && (
                <div className="space-y-5">
                  {/* Summary Callout */}
                  <div className="p-4 rounded-xl bg-gradient-to-r from-purple-950/30 via-zinc-950 to-zinc-950 border-l-4 border-l-purple-500 border border-zinc-800/80">
                    <div className="text-[10px] font-mono uppercase tracking-wider text-purple-400 font-semibold mb-1">
                      Reasoning Summary
                    </div>
                    <div className="text-xs text-zinc-200 leading-relaxed font-sans">
                      "{explanation.summary}"
                    </div>
                  </div>

                  {/* 4-Box Evidence Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Text Evidence */}
                    <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800/90 space-y-2">
                      <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300 uppercase tracking-wider border-b border-zinc-800/80 pb-2">
                        <FileText className="w-3.5 h-3.5 text-indigo-400" />
                        Textual Evidence
                      </div>
                      <ul className="space-y-1.5 text-xs text-zinc-400 leading-relaxed list-disc list-inside">
                        {explanation.text_evidence.map((ev, i) => (
                          <li key={i} className="text-zinc-300">{ev}</li>
                        ))}
                      </ul>
                    </div>

                    {/* Visual Evidence */}
                    <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800/90 space-y-2">
                      <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300 uppercase tracking-wider border-b border-zinc-800/80 pb-2">
                        <Eye className="w-3.5 h-3.5 text-cyan-400" />
                        Visual Observations
                      </div>
                      <ul className="space-y-1.5 text-xs text-zinc-400 leading-relaxed list-disc list-inside">
                        {explanation.visual_evidence.map((ev, i) => (
                          <li key={i} className="text-zinc-300">{ev}</li>
                        ))}
                      </ul>
                    </div>

                    {/* Supporting Affective Signals */}
                    <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800/90 space-y-2">
                      <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300 uppercase tracking-wider border-b border-zinc-800/80 pb-2">
                        <BarChart3 className="w-3.5 h-3.5 text-emerald-400" />
                        Affective & Valence Signals
                      </div>
                      <ul className="space-y-1.5 text-xs text-zinc-400 leading-relaxed list-disc list-inside">
                        {explanation.supporting_signals.map((sig, i) => (
                          <li key={i} className="text-zinc-300">{sig}</li>
                        ))}
                      </ul>
                    </div>

                    {/* Epistemological Limitations */}
                    <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800/90 space-y-2">
                      <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300 uppercase tracking-wider border-b border-zinc-800/80 pb-2">
                        <ShieldCheck className="w-3.5 h-3.5 text-amber-400" />
                        Epistemological Limitations
                      </div>
                      <ul className="space-y-1.5 text-xs text-zinc-400 leading-relaxed list-disc list-inside">
                        {explanation.limitations.map((lim, i) => (
                          <li key={i} className="text-zinc-400">{lim}</li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* Grounding Guarantee Notice */}
                  <div className="flex items-center gap-2 p-3 rounded-lg bg-zinc-950/60 border border-zinc-800 text-[11px] text-zinc-500">
                    <Info className="w-4 h-4 text-zinc-400 flex-shrink-0" />
                    <span>
                      <strong className="text-zinc-400">Architectural Separation:</strong> The sarcasm verdict was classified authoritatively by the 2,847-D MLP classifier. The explanation reasoner articulates supporting evidence without modifying model outputs.
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Two Column Grid: Sentiment Analysis & Emotion Spectrum */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Sentiment Card */}
              <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-800/80 pb-3">
                  <div className="flex items-center gap-2">
                    <Smile className="w-4 h-4 text-amber-400" />
                    <span className="text-xs font-bold text-zinc-200 uppercase tracking-wider">
                      Sentiment Polarity (3-D)
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-zinc-500">XLM-R Classifier</span>
                </div>

                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="bg-zinc-950 p-3 rounded-lg border border-red-950/50">
                    <div className="text-[11px] text-red-400 font-semibold uppercase">Negative</div>
                    <div className="text-base font-bold font-mono text-zinc-100 mt-1">
                      {(result.sentiment.negative * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                    <div className="text-[11px] text-zinc-400 font-semibold uppercase">Neutral</div>
                    <div className="text-base font-bold font-mono text-zinc-100 mt-1">
                      {(result.sentiment.neutral * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-zinc-950 p-3 rounded-lg border border-emerald-950/50">
                    <div className="text-[11px] text-emerald-400 font-semibold uppercase">Positive</div>
                    <div className="text-base font-bold font-mono text-zinc-100 mt-1">
                      {(result.sentiment.positive * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
                <p className="text-[11px] text-zinc-500 leading-relaxed">
                  Canonical indices 2816–2818 in the fusion matrix: [P(Negative), P(Neutral), P(Positive)].
                </p>
              </div>

              {/* Emotion Spectrum Card */}
              <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-zinc-800/80 pb-3">
                  <div className="flex items-center gap-2">
                    <Heart className="w-4 h-4 text-rose-400" />
                    <span className="text-xs font-bold text-zinc-200 uppercase tracking-wider">
                      GoEmotions Spectrum (28-D)
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-zinc-500">Sigmoid Probabilities</span>
                </div>

                <div className="space-y-2">
                  <div className="text-[11px] text-zinc-400 font-medium">Top Detected Emotions:</div>
                  <div className="space-y-1.5">
                    {result.emotions.top_emotions?.map(item => (
                      <div key={item.emotion} className="flex items-center justify-between text-xs bg-zinc-950 px-3 py-2 rounded-lg border border-zinc-800/80">
                        <span className="font-semibold text-zinc-200 capitalize">{item.emotion}</span>
                        <span className="font-mono text-zinc-400">{(item.probability * 100).toFixed(2)}%</span>
                      </div>
                    ))}
                  </div>

                  <button
                    type="button"
                    onClick={() => setShowAllEmotions(!showAllEmotions)}
                    className="w-full mt-2 py-1.5 text-center text-[11px] text-zinc-400 hover:text-white flex items-center justify-center gap-1 transition-colors"
                  >
                    {showAllEmotions ? <>Hide full taxonomy <ChevronUp className="w-3 h-3" /></> : <>View all 28 GoEmotions <ChevronDown className="w-3 h-3" /></>}
                  </button>

                  <AnimatePresence>
                    {showAllEmotions && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="grid grid-cols-2 gap-1.5 pt-2 text-[10px] font-mono max-h-48 overflow-y-auto pr-1"
                      >
                        {Object.entries(result.emotions.label_probabilities).map(([label, prob]) => (
                          <div key={label} className="flex justify-between bg-zinc-950 px-2 py-1 rounded border border-zinc-800/60">
                            <span className="text-zinc-400 truncate">{label}</span>
                            <span className="text-zinc-200">{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </div>

            {/* Architecture Diagnostics & Feature Concatenation Details */}
            <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs font-bold text-zinc-200 uppercase tracking-wider">
                    Pipeline Execution & Representation Breakdown
                  </span>
                </div>
                <span className="text-xs font-mono text-cyan-400 font-bold">2,847 Total Dimensions</span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <div className="text-[10px] text-zinc-500 uppercase font-mono">XLM-R Text</div>
                  <div className="text-xs font-bold text-zinc-200 mt-0.5">768-D</div>
                  <div className="text-[10px] text-zinc-500 font-mono mt-1">Indices 0–767</div>
                </div>
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <div className="text-[10px] text-zinc-500 uppercase font-mono">CLIP ViT-B/32</div>
                  <div className="text-xs font-bold text-zinc-200 mt-0.5">512-D</div>
                  <div className="text-[10px] text-zinc-500 font-mono mt-1">Indices 768–1279</div>
                </div>
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <div className="text-[10px] text-zinc-500 uppercase font-mono">InternVL3-2B</div>
                  <div className="text-xs font-bold text-zinc-200 mt-0.5">1,536-D</div>
                  <div className="text-[10px] text-zinc-500 font-mono mt-1">Indices 1280–2815</div>
                </div>
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <div className="text-[10px] text-zinc-500 uppercase font-mono">Sentiment</div>
                  <div className="text-xs font-bold text-zinc-200 mt-0.5">3-D</div>
                  <div className="text-[10px] text-zinc-500 font-mono mt-1">Indices 2816–2818</div>
                </div>
                <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800">
                  <div className="text-[10px] text-zinc-500 uppercase font-mono">Emotion</div>
                  <div className="text-xs font-bold text-zinc-200 mt-0.5">28-D</div>
                  <div className="text-[10px] text-zinc-500 font-mono mt-1">Indices 2819–2846</div>
                </div>
              </div>

              {result.performance.latency_breakdown_ms && (
                <div className="pt-2 text-[11px] text-zinc-400 font-mono flex flex-wrap gap-x-4 gap-y-1">
                  <span>Latency Breakdown:</span>
                  <span>Text: {result.performance.latency_breakdown_ms.text_ms}ms</span>
                  <span>CLIP: {result.performance.latency_breakdown_ms.clip_ms}ms</span>
                  <span>InternVL: {result.performance.latency_breakdown_ms.vlm_ms}ms</span>
                  <span>Sentiment: {result.performance.latency_breakdown_ms.sentiment_ms}ms</span>
                  <span>Emotion: {result.performance.latency_breakdown_ms.emotion_ms}ms</span>
                  <span>Fusion: {result.performance.latency_breakdown_ms.fusion_ms}ms</span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Feature Pipeline Architecture Overview */}
      <div className="space-y-4 pt-4 border-t border-zinc-800/80">
        <h2 className="text-xs text-zinc-500 uppercase tracking-widest font-semibold">Multimodal Feature Pipeline Architecture</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <PipelineStage label="XLM-R" sublabel="Text Context" status="ready" dimension="768-D" />
          <PipelineStage label="CLIP" sublabel="ViT-B/32 Vision" status="ready" dimension="512-D" />
          <PipelineStage label="InternVL3-2B" sublabel="Dynamic Tiling" status="ready" dimension="1536-D" />
          <PipelineStage label="Sentiment" sublabel="Polarity Vector" status="ready" dimension="3-D" />
          <PipelineStage label="Emotion" sublabel="GoEmotions" status="ready" dimension="28-D" />
          <PipelineStage label="Fusion MLP" sublabel="Concatenated" status="ready" dimension="2847-D" />
        </div>
      </div>
    </div>
  );
}
