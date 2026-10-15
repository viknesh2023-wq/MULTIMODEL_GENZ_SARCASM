import axios from 'axios';
import type {
  HealthResponse,
  ModelStatusResponse,
  TextPredictResponse,
  MetricsResponse,
  DatasetStatsResponse,
  FeatureStatusResponse,
  MultimodalNotReadyResponse,
  MultimodalPredictResponse,
  ExplanationResponse,
} from '../types';

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Health & Status ───────────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthResponse> {
  const res = await api.get<HealthResponse>('/health');
  return res.data;
}

export async function getModelStatus(): Promise<ModelStatusResponse> {
  const res = await api.get<ModelStatusResponse>('/model/status');
  return res.data;
}

// ── Prediction ────────────────────────────────────────────────────────────────

export async function predictText(text: string): Promise<TextPredictResponse> {
  const res = await api.post<TextPredictResponse>('/predict/text', { text });
  return res.data;
}

export async function predictMultimodal(
  text: string,
  imageFile: File,
  explain: boolean = false
): Promise<MultimodalPredictResponse> {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('image', imageFile);
  if (explain) {
    formData.append('explain', 'true');
  }

  const res = await api.post<MultimodalPredictResponse>('/predict/multimodal', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
  return res.data;
}

export async function explainPrediction(payload: {
  text: string;
  prediction: any;
  sentiment: any;
  emotions: any;
  visual_evidence?: string;
}): Promise<ExplanationResponse> {
  const res = await api.post<ExplanationResponse>('/predict/explain', payload, {
    timeout: 60000,
  });
  return res.data;
}


// ── Metrics & Data ────────────────────────────────────────────────────────────

export async function getMetrics(): Promise<MetricsResponse> {
  const res = await api.get<MetricsResponse>('/model/metrics');
  return res.data;
}

export async function getDatasetStats(): Promise<DatasetStatsResponse> {
  const res = await api.get<DatasetStatsResponse>('/dataset/stats');
  return res.data;
}

export async function getFeatureStatus(): Promise<FeatureStatusResponse> {
  const res = await api.get<FeatureStatusResponse>('/features/status');
  return res.data;
}
