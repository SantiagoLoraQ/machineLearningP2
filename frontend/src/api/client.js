const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    let detail = `Error HTTP ${response.status}`;
    try {
      const errData = await response.json();
      if (errData?.detail) detail = errData.detail;
    } catch {
      // Ignorar si no es JSON
    }
    throw new Error(detail);
  }
  return response.json();
}

export function getHealth() {
  return requestJson('/api/health');
}

export function getDatasetSummary() {
  return requestJson('/api/dataset/summary');
}

export function getDatasetSamples({
  species = '',
  quality = '',
  split = '',
  search = '',
  limit = 80,
  offset = 0,
} = {}) {
  const params = new URLSearchParams();
  if (species) params.set('species', species);
  if (quality) params.set('quality', quality);
  if (split) params.set('split', split);
  if (search) params.set('search', search);
  params.set('limit', String(limit));
  params.set('offset', String(offset));
  return requestJson(`/api/dataset/samples?${params.toString()}`);
}

export function getAudioUrl(sampleId) {
  return `${API_BASE}/api/audio/${encodeURIComponent(sampleId)}`;
}

export function predictAudioFile(file, theme = 'light') {
  const formData = new FormData();
  formData.append('file', file);
  return requestJson(`/api/predict?theme=${encodeURIComponent(theme)}`, {
    method: 'POST',
    body: formData,
  });
}

export function predictDatasetSample(sampleId, theme = 'light') {
  return requestJson(
    `/api/predict/sample/${encodeURIComponent(sampleId)}?theme=${encodeURIComponent(theme)}`,
    {
      method: 'POST',
    }
  );
}

export function getMetricsOverview() {
  return requestJson('/api/metrics/overview');
}

export function getUnsupervisedStatus() {
  return requestJson('/api/unsupervised/status');
}
