import React, { useEffect, useState } from 'react';
import { getDatasetSummary, getHealth, getMetricsOverview } from './api/client';
import AudioDetector from './components/AudioDetector';
import DatasetExplorer from './components/DatasetExplorer';
import Header from './components/Header';
import ModelMetrics from './components/ModelMetrics';
import UnsupervisedPlaceholder from './components/UnsupervisedPlaceholder';

function getInitialTheme() {
  try {
    const saved = localStorage.getItem('aves-bolivia-theme');
    if (saved === 'dark' || saved === 'light') return saved;
  } catch {
    // Ignorar si localStorage no está disponible
  }
  return 'light';
}

export default function App() {
  const [activeTab, setActiveTab] = useState('detector');
  const [theme, setTheme] = useState(getInitialTheme);

  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState('');

  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [summaryError, setSummaryError] = useState('');

  const [metricsData, setMetricsData] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [metricsError, setMetricsError] = useState('');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('aves-bolivia-theme', theme);
    } catch {
      // Ignorar
    }
  }, [theme]);

  useEffect(() => {
    getHealth()
      .then((data) => {
        setHealth(data);
        setHealthError('');
      })
      .catch((err) => setHealthError(err.message));

    getDatasetSummary()
      .then((data) => {
        setSummary(data);
        setSummaryError('');
      })
      .catch((err) => setSummaryError(err.message))
      .finally(() => setSummaryLoading(false));

    getMetricsOverview()
      .then((data) => {
        setMetricsData(data);
        setMetricsError('');
      })
      .catch((err) => setMetricsError(err.message))
      .finally(() => setMetricsLoading(false));
  }, []);

  function toggleTheme() {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  }

  return (
    <div className="app-shell">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        health={health}
        healthError={healthError}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <main className="main-content">
        {activeTab === 'detector' && <AudioDetector theme={theme} />}

        {activeTab === 'dataset' && (
          <DatasetExplorer
            summary={summary}
            loading={summaryLoading}
            error={summaryError}
            theme={theme}
          />
        )}

        {activeTab === 'metrics' && (
          <ModelMetrics
            metricsData={metricsData}
            loading={metricsLoading}
            error={metricsError}
            theme={theme}
          />
        )}

        {activeTab === 'unsupervised' && <UnsupervisedPlaceholder />}
      </main>

      <footer className="app-footer">
        Práctica del Segundo Parcial · Reconocimiento de Vocalizaciones de Aves de Bolivia · Primera Entrega (CNN 2D PyTorch + FastAPI + React)
      </footer>
    </div>
  );
}
