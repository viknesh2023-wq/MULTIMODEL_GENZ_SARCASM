// ── API Response Types ────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version: string;
  model_loaded: boolean;
  multimodal_ready?: boolean;
  explanation_ready?: boolean;
}

export interface ModelStatusResponse {
  model: string;
  status: 'ready' | 'error' | 'loading';
  device: string;
  model_path: string;
  loaded: boolean;
  multimodal_ready?: boolean;
  explanation_ready?: boolean;
}

export interface ProbabilitySet {
  non_sarcastic: number;
  sarcastic: number;
}

export interface TextPredictResponse {
  label: 0 | 1;
  label_name: string;
  confidence: number;
  probabilities: ProbabilitySet;
  model: string;
  device: string;
}

export interface TextSarcasmMetrics {
  model: string;
  dataset: string;
  train_samples: number;
  validation_samples: number;
  test_samples: number;
  max_length: number;
  epochs: number;
  learning_rate: number;
  train_batch_size: number;
  eval_batch_size: number;
  gradient_accumulation_steps: number;
  effective_batch_size: number;
  test_accuracy: number;
  test_macro_precision: number;
  test_macro_recall: number;
  test_macro_f1: number;
  confusion_matrix: number[][];
  labels?: Record<string, string>;
}

export interface SentimentClassReport {
  precision: number;
  recall: number;
  'f1-score': number;
  support: number;
}

export interface SentimentMetrics {
  model: string;
  task: string;
  train_samples: number;
  validation_samples: number;
  test_samples: number;
  epochs: number;
  effective_batch_size: number;
  learning_rate: number;
  test_accuracy: number;
  test_macro_precision: number;
  test_macro_recall: number;
  test_macro_f1: number;
  confusion_matrix: number[][];
  classification_report?: {
    Negative: SentimentClassReport;
    Neutral: SentimentClassReport;
    Positive: SentimentClassReport;
    accuracy: number;
    'macro avg': SentimentClassReport;
    'weighted avg': SentimentClassReport;
  };
  labels?: Record<string, string>;
}

export interface SingleEmotionMetric {
  emotion_id: number;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface EmotionMetrics {
  model: string;
  task: string;
  train_samples: number;
  validation_samples: number;
  test_samples: number;
  classification_threshold: number;
  test_macro_f1: number;
  test_micro_f1: number;
  test_macro_precision: number;
  test_macro_recall: number;
  test_micro_precision: number;
  test_micro_recall: number;
  hamming_loss: number;
  exact_match_accuracy: number;
  per_emotion_metrics: Record<string, SingleEmotionMetric>;
}

export interface FusionPerClassMetric {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface MultimodalFusionMetrics {
  configuration: string;
  description: string;
  input_dimension: number;
  best_epoch: number;
  val_macro_f1: number;
  test_accuracy: number;
  test_macro_f1: number;
  test_macro_precision: number;
  test_macro_recall: number;
  test_weighted_f1: number;
  per_class_metrics?: {
    '0_non_sarcastic': FusionPerClassMetric;
    '1_sarcastic': FusionPerClassMetric;
  };
  confusion_matrix: number[][];
  training_time_seconds: number;
  test_inference_time_seconds: number;
  latency_per_sample_ms: number;
  peak_gpu_vram_mb: number;
}

export interface AblationBenchmarkRow {
  configuration: string;
  description: string;
  dimensions: number;
  best_epoch: number;
  val_macro_f1: number;
  test_accuracy: number;
  test_macro_f1: number;
  test_macro_precision: number;
  test_macro_recall: number;
  test_weighted_f1: number;
  class_0_f1: number;
  class_1_f1: number;
  train_time_seconds: number;
  latency_ms: number;
  peak_vram_mb: number;
}

export interface MetricsResponse {
  test_accuracy: number;
  test_macro_precision: number;
  test_macro_recall: number;
  test_macro_f1: number;
  confusion_matrix: number[][];
  train_samples: number;
  validation_samples: number;
  test_samples: number;
  epochs: number;
  learning_rate: number;
  model_name: string;
  text_sarcasm?: TextSarcasmMetrics;
  sentiment?: SentimentMetrics;
  emotion?: EmotionMetrics;
  multimodal_fusion?: MultimodalFusionMetrics;
  ablation_benchmark?: AblationBenchmarkRow[];
}

export interface DatasetStats {
  total_samples: number;
  train_samples: number;
  validation_samples: number;
  test_samples: number;
  non_sarcastic?: number;
  sarcastic?: number;
  positive?: number;
  neutral?: number;
  negative?: number;
  classes?: number;
  total_classes?: number;
  taxonomy?: string;
  multi_label_samples?: number;
  null_count?: number;
  duplicate_count?: number;
  verified_images_on_disk?: number;
  languages?: Record<string, number> | string[];
  sources?: Record<string, number> | string[];
  clip_features_extracted?: number;
  internvl_features_extracted?: number;
  xlmr_features_extracted?: number;
  sentiment_features_extracted?: number;
  emotion_features_extracted?: number;
  file_path?: string;
}

export interface DatasetStatsResponse {
  text_dataset: DatasetStats;
  multimodal_dataset: DatasetStats;
  sentiment_dataset?: DatasetStats;
  emotion_dataset?: DatasetStats;
}

export interface FeatureItem {
  name: string;
  dimension: number;
  samples: number | null;
  status: 'ready' | 'text_model_ready' | 'pending' | 'in_development';
  status_label: string;
  color: 'green' | 'amber' | 'blue';
  description: string;
}

export interface FeatureStatusResponse {
  features: FeatureItem[];
}

export interface MultimodalNotReadyResponse {
  status: 'not_ready';
  message: string;
  feature_status: Record<string, string>;
}

export interface MultimodalPrediction {
  label: 'sarcastic' | 'non-sarcastic';
  class_id: 0 | 1;
  probabilities: ProbabilitySet;
  confidence: number;
}

export interface SentimentProbabilities {
  negative: number;
  neutral: number;
  positive: number;
}

export interface TopEmotion {
  emotion: string;
  probability: number;
}

export interface EmotionSpectrum {
  label_probabilities: Record<string, number>;
  top_emotions?: TopEmotion[];
}

export interface MultimodalModelMeta {
  fusion_model: string;
  feature_dimensions: number;
  branches?: Record<string, number>;
}

export interface MultimodalPerformance {
  total_latency_ms: number;
  latency_breakdown_ms?: Record<string, number>;
}

export interface ExplanationContent {
  summary: string;
  text_evidence: string[];
  visual_evidence: string[];
  supporting_signals: string[];
  limitations: string[];
}

export interface ExplanationReasonerMeta {
  model: string;
  runtime: string;
  latency_ms?: number;
}

export interface ExplanationResponse {
  status: 'success' | 'unavailable' | 'partial';
  explanation?: ExplanationContent;
  prediction_reference: MultimodalPrediction;
  reasoner: ExplanationReasonerMeta;
  error?: string;
}

export interface MultimodalPredictResponse {
  status: 'success';
  prediction: MultimodalPrediction;
  sentiment: SentimentProbabilities;
  emotions: EmotionSpectrum;
  model: MultimodalModelMeta;
  performance: MultimodalPerformance;
  explanation?: ExplanationContent;
  reasoner?: ExplanationReasonerMeta;
}

// ── UI State Types ────────────────────────────────────────────────────────────

export type ModelStatus = 'online' | 'offline' | 'loading';

export interface BackendStatus {
  connected: boolean;
  modelLoaded: boolean;
  device: string;
}

export type NavPage =
  | 'overview'
  | 'detection'
  | 'text-analysis'
  | 'multimodal'
  | 'architecture'
  | 'performance'
  | 'features'
  | 'dataset'
  | 'research'
  | 'about';

export interface AnalysisResult {
  prediction: TextPredictResponse;
  textStats: TextStats;
  inputText: string;
}

export interface TextStats {
  charCount: number;
  wordCount: number;
  sentenceCount: number;
  hashtagCount: number;
  mentionCount: number;
  urlCount: number;
  emojiCount: number;
  exclamationCount: number;
  questionCount: number;
  capsWordCount: number;
  repeatedPunctCount: number;
  possibleSlang: string[];
}
