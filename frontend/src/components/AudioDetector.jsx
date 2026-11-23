import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  AudioLines,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FolderOpen,
  Mic,
  Shuffle,
  Square,
  Upload,
  X,
  XCircle,
} from 'lucide-react';
import {
  getAudioUrl,
  getDatasetSamples,
  predictAudioFile,
  predictDatasetSample,
} from '../api/client';
import { createMicrophoneRecorder } from '../utils/audioRecorder';

function WaveformSvg({ points, theme }) {
  if (!points || points.length === 0) return null;
  const width = 600;
  const height = 70;
  const mid = height / 2;
  const barWidth = width / points.length;
  const barColor = theme === 'dark' ? '#64a883' : '#2d6a4f';
  const lineColor = theme === 'dark' ? '#2f4236' : '#cddbce';

  return (
    <div className="waveform-box">
      <div
        style={{
          fontSize: '0.76rem',
          color: 'var(--text-secondary)',
          marginBottom: 4,
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>Forma de onda estandarizada</span>
        <span>22,050 Hz · Mono</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: '100%', height: 60, display: 'block' }}
      >
        <line x1="0" y1={mid} x2={width} y2={mid} stroke={lineColor} strokeWidth="1" />
        {points.map((val, idx) => {
          const amp = Math.max(0.03, Math.min(1, Math.abs(val)));
          const barH = amp * (height - 8);
          return (
            <rect
              key={idx}
              x={idx * barWidth + 0.5}
              y={mid - barH / 2}
              width={Math.max(1.5, barWidth - 1.5)}
              height={barH}
              rx="1"
              fill={barColor}
              opacity={0.45 + 0.55 * amp}
            />
          );
        })}
      </svg>
    </div>
  );
}

function formatSeconds(sec) {
  const s = Math.floor(sec);
  const mm = String(Math.floor(s / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

export default function AudioDetector({ theme = 'light' }) {
  const fileInputRef = useRef(null);
  const recorderRef = useRef(null);

  const [isDragging, setIsDragging] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordElapsed, setRecordElapsed] = useState(0);
  const [micLevel, setMicLevel] = useState(0);

  const [audioSourceLabel, setAudioSourceLabel] = useState('');
  const [localAudioUrl, setLocalAudioUrl] = useState('');
  const [selectedSample, setSelectedSample] = useState(null);

  const [showDatasetDrawer, setShowDatasetDrawer] = useState(false);
  const [splitFilter, setSplitFilter] = useState('test');
  const [speciesFilter, setSpeciesFilter] = useState('');
  const [qualityFilter, setQualityFilter] = useState('');
  const [searchText, setSearchText] = useState('');
  const [samples, setSamples] = useState([]);
  const [speciesList, setSpeciesList] = useState([]);
  const [totalFiltered, setTotalFiltered] = useState(0);

  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    getDatasetSamples({
      split: splitFilter,
      species: speciesFilter,
      quality: qualityFilter,
      search: searchText,
      limit: 100,
    })
      .then((data) => {
        if (!active) return;
        setSamples(data.items || []);
        setTotalFiltered(data.total || 0);
        if (data.species_list?.length) setSpeciesList(data.species_list);
      })
      .catch((err) => {
        if (active) setError(err.message);
      });
    return () => {
      active = false;
    };
  }, [splitFilter, speciesFilter, qualityFilter, searchText]);

  useEffect(() => {
    return () => {
      if (localAudioUrl) URL.revokeObjectURL(localAudioUrl);
    };
  }, [localAudioUrl]);

  useEffect(() => {
    return () => {
      if (recorderRef.current) {
        recorderRef.current.cancel();
      }
    };
  }, []);

  const activeAudioSrc = useMemo(() => {
    if (localAudioUrl) return localAudioUrl;
    if (selectedSample) return getAudioUrl(selectedSample.id);
    return '';
  }, [localAudioUrl, selectedSample]);

  async function runFilePrediction(file, objectUrl, sourceLabel) {
    setError('');
    setSelectedSample(null);
    if (localAudioUrl && localAudioUrl !== objectUrl) {
      URL.revokeObjectURL(localAudioUrl);
    }
    setLocalAudioUrl(objectUrl);
    setAudioSourceLabel(sourceLabel);
    setPredicting(true);
    try {
      const res = await predictAudioFile(file, theme);
      setPrediction(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  }

  function handleFileSelected(file) {
    if (!file) return;
    const objUrl = URL.createObjectURL(file);
    runFilePrediction(file, objUrl, `Archivo: ${file.name}`);
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    if (isRecording || predicting) return;
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFileSelected(file);
  }

  async function handleStopAndAnalyzeMic() {
    if (!recorderRef.current) return;
    const recorder = recorderRef.current;
    recorderRef.current = null;
    setIsRecording(false);
    setMicLevel(0);

    try {
      const { file, url, durationSec } = await recorder.stop();
      await runFilePrediction(
        file,
        url,
        `Grabación de micrófono (${durationSec} s)`
      );
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleToggleMicrophone() {
    if (isRecording) {
      await handleStopAndAnalyzeMic();
      return;
    }
    setError('');
    try {
      const recorder = await createMicrophoneRecorder({
        maxDurationSec: 15,
        onLevel: setMicLevel,
        onTick: setRecordElapsed,
        onAutoStop: () => {
          handleStopAndAnalyzeMic();
        },
      });
      recorderRef.current = recorder;
      setRecordElapsed(0);
      setIsRecording(true);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleCancelMicrophone() {
    if (recorderRef.current) {
      const recorder = recorderRef.current;
      recorderRef.current = null;
      await recorder.cancel();
    }
    setIsRecording(false);
    setRecordElapsed(0);
    setMicLevel(0);
  }

  async function handleAnalyzeSample(sample) {
    if (!sample) return;
    if (localAudioUrl) {
      URL.revokeObjectURL(localAudioUrl);
      setLocalAudioUrl('');
    }
    setSelectedSample(sample);
    setAudioSourceLabel(`Muestra del dataset: XC${sample.id} (${sample.especie})`);
    setError('');
    setPredicting(true);
    try {
      const res = await predictDatasetSample(sample.id, theme);
      setPrediction(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  }

  function handlePickRandomSample() {
    if (!samples.length) return;
    const randomIdx = Math.floor(Math.random() * samples.length);
    handleAnalyzeSample(samples[randomIdx]);
  }

  const isDark = theme === 'dark';
  const gridStroke = isDark ? '#28392e' : '#d5e2d7';
  const axisStroke = isDark ? '#a4b8aa' : '#4a5e50';
  const primaryBar = isDark ? '#3d8b63' : '#2d6a4f';
  const secondaryBar = isDark ? '#64a883' : '#6ea484';
  const tooltipStyle = {
    backgroundColor: isDark ? '#1a231d' : '#ffffff',
    borderColor: isDark ? '#2f4236' : '#cddbce',
    color: isDark ? '#e7efe9' : '#1c2a21',
    borderRadius: 4,
  };

  return (
    <div className="detector-center-wrap">
      <div className="detector-intro">
        <h2>Identificar Especie por Canto o Llamado</h2>
        <p>
          Sube una grabación de campo o escucha directamente desde el micrófono para
          determinar a qué especie de ave de Bolivia corresponde el sonido.
        </p>
      </div>

      {/* Ventana principal centrada con los dos botones en el medio */}
      <div
        className="window-card"
        onDragOver={(e) => {
          e.preventDefault();
          if (!isRecording) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <div className="window-bar">
          <div>
            <h3 className="window-title">
              <AudioLines size={16} color="var(--green-primary)" />
              Entrada de Sonido
            </h3>
            <p className="window-subtitle">
              Formatos aceptados: .wav, .mp3, .flac, .ogg o captura en vivo (máx. 15 s)
            </p>
          </div>
          <span className="badge badge-green">CNN 2D · 39 especies</span>
        </div>

        <div className="window-body">
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,.ogg"
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFileSelected(file);
              e.target.value = '';
            }}
          />

          <div className="primary-action-grid">
            <button
              type="button"
              className={`action-tile primary ${isDragging ? 'dragging' : ''}`}
              disabled={isRecording || predicting}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={26} />
              <span className="action-tile-title">Subir archivo de audio</span>
              <span className="action-tile-desc">
                Haz clic para seleccionar o arrastra aquí un archivo (.wav, .mp3)
              </span>
            </button>

            <button
              type="button"
              className={`action-tile ${isRecording ? 'recording' : ''}`}
              disabled={predicting}
              onClick={handleToggleMicrophone}
            >
              {isRecording ? <Square size={26} /> : <Mic size={26} color="var(--green-primary)" />}
              <span className="action-tile-title">
                {isRecording ? 'Detener y analizar' : 'Escuchar por micrófono'}
              </span>
              <span className="action-tile-desc">
                {isRecording
                  ? `Grabando audio en vivo (${formatSeconds(recordElapsed)} / 00:15)`
                  : 'Graba el canto desde el micrófono de tu equipo y clasifícalo'}
              </span>
            </button>
          </div>

          {isRecording && (
            <div className="mic-panel">
              <div className="mic-panel-row">
                <span style={{ fontWeight: 600, fontSize: '0.86rem' }}>
                  Escuchando micrófono... {formatSeconds(recordElapsed)} / 00:15
                </span>
                <div className="controls-row">
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={handleStopAndAnalyzeMic}
                  >
                    <Square size={14} />
                    Detener y analizar
                  </button>
                  <button
                    type="button"
                    className="btn btn-neutral"
                    onClick={handleCancelMicrophone}
                  >
                    <X size={14} />
                    Cancelar
                  </button>
                </div>
              </div>
              <div className="mic-meter-track" title="Nivel de entrada del micrófono">
                <div
                  className="mic-meter-fill"
                  style={{ width: `${Math.round(micLevel * 100)}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Opción secundaria desplegable para probar grabaciones del dataset */}
      <div className="window-card">
        <button
          type="button"
          className="dataset-drawer-toggle"
          onClick={() => setShowDatasetDrawer((prev) => !prev)}
        >
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
            <FolderOpen size={15} color="var(--green-primary)" />
            Probar con grabaciones verificadas del dataset ({totalFiltered} disponibles)
          </span>
          {showDatasetDrawer ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        {showDatasetDrawer && (
          <div className="window-body" style={{ borderTop: '1px solid var(--border-color)' }}>
            <div className="controls-row" style={{ marginBottom: 10 }}>
              <select
                className="select"
                value={splitFilter}
                onChange={(e) => setSplitFilter(e.target.value)}
              >
                <option value="test">Partición: Prueba (Test)</option>
                <option value="val">Partición: Validación</option>
                <option value="train">Partición: Entrenamiento</option>
                <option value="">Todas las particiones</option>
              </select>

              <select
                className="select"
                value={qualityFilter}
                onChange={(e) => setQualityFilter(e.target.value)}
              >
                <option value="">Calidad: Todas</option>
                <option value="A">Calidad A</option>
                <option value="B">Calidad B</option>
                <option value="C">Calidad C</option>
              </select>

              <select
                className="select"
                style={{ flex: 1, minWidth: 160 }}
                value={speciesFilter}
                onChange={(e) => setSpeciesFilter(e.target.value)}
              >
                <option value="">Todas las especies (39)</option>
                {speciesList.map((sp) => (
                  <option key={sp} value={sp}>
                    {sp}
                  </option>
                ))}
              </select>

              <input
                type="text"
                className="input"
                placeholder="Buscar ID o tipo..."
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
              />

              <button
                type="button"
                className="btn"
                onClick={handlePickRandomSample}
                disabled={!samples.length || predicting}
              >
                <Shuffle size={14} />
                Muestra aleatoria
              </button>
            </div>

            <div className="sample-list">
              {samples.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  className={`sample-item ${selectedSample?.id === s.id ? 'selected' : ''}`}
                  disabled={predicting}
                  onClick={() => handleAnalyzeSample(s)}
                >
                  <div>
                    <div className="italic-species" style={{ fontWeight: 600 }}>
                      {s.especie}
                    </div>
                    <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                      XC{s.id} · {s.duracion_seg} s · {s.tipo_vocalizacion_principal}
                    </div>
                  </div>
                  <div className="pill-list">
                    <span className="badge badge-neutral">{s.split}</span>
                    <span
                      className={`badge ${
                        s.calidad === 'A'
                          ? 'badge-green'
                          : s.calidad === 'B'
                          ? 'badge-neutral'
                          : 'badge-amber'
                      }`}
                    >
                      Calidad {s.calidad}
                    </span>
                  </div>
                </button>
              ))}
              {samples.length === 0 && (
                <div className="empty-state">No hay muestras para estos filtros.</div>
              )}
            </div>
          </div>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {predicting && (
        <div className="window-card">
          <div className="window-body empty-state" style={{ border: 'none' }}>
            Procesando señal de audio y ejecutando modelo CNN 2D...
          </div>
        </div>
      )}

      {!predicting && (activeAudioSrc || prediction) && (
        <div className="section-stack">
          {prediction && (
            <div className="window-card">
              <div className="window-bar">
                <h3 className="window-title">Resultado de Identificación</h3>
                {prediction.audio_info && (
                  <div className="pill-list">
                    <span className="badge badge-neutral">
                      Duración: {prediction.audio_info.duration_sec} s
                    </span>
                    <span className="badge badge-neutral">
                      Ventanas 4 s: {prediction.audio_info.windows_analyzed}
                    </span>
                  </div>
                )}
              </div>

              <div className="window-body">
                <div className="prediction-banner">
                  <div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                      Especie probable
                    </div>
                    <div
                      className="italic-species"
                      style={{
                        fontSize: '1.5rem',
                        fontWeight: 700,
                        color: 'var(--green-primary)',
                      }}
                    >
                      {prediction.predicted_species}
                    </div>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '1.45rem', fontWeight: 700 }}>
                      {(prediction.confidence * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      Confianza Softmax
                    </div>
                  </div>
                </div>

                <div className="confidence-bar-bg" style={{ marginBottom: 12 }}>
                  <div
                    className="confidence-bar-fill"
                    style={{ width: `${Math.min(100, prediction.confidence * 100)}%` }}
                  />
                </div>

                {prediction.mode === 'dataset_sample' && (
                  <div
                    className="pill-list"
                    style={{
                      paddingTop: 10,
                      borderTop: '1px solid var(--border-color)',
                    }}
                  >
                    {prediction.is_correct ? (
                      <span className="badge badge-green">
                        <CheckCircle2 size={14} />
                        Predicción correcta
                      </span>
                    ) : (
                      <span className="badge badge-danger">
                        <XCircle size={14} />
                        Confusión detectada
                      </span>
                    )}
                    <span style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
                      Especie real:{' '}
                      <strong className="italic-species" style={{ color: 'var(--text-primary)' }}>
                        {prediction.true_species}
                      </strong>
                    </span>
                    <span className="badge badge-neutral">Split: {prediction.split}</span>
                    <span className="badge badge-neutral">Calidad: {prediction.calidad}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeAudioSrc && (
            <div className="window-card">
              <div className="window-bar">
                <h3 className="window-title">Reproductor y Forma de Onda</h3>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {audioSourceLabel}
                </span>
              </div>
              <div className="window-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <audio
                  key={activeAudioSrc}
                  controls
                  src={activeAudioSrc}
                  style={{ width: '100%' }}
                />
                {prediction?.waveform_preview && (
                  <WaveformSvg points={prediction.waveform_preview} theme={theme} />
                )}
              </div>
            </div>
          )}

          {prediction && (
            <div className="grid-2">
              <div className="window-card">
                <div className="window-bar">
                  <div>
                    <h3 className="window-title">Espectrograma de Mel (dB)</h3>
                    <p className="window-subtitle">
                      64 bandas Mel entre 300 Hz y 11,025 Hz
                    </p>
                  </div>
                </div>
                <div className="window-body">
                  {prediction.spectrogram_base64 && (
                    <img
                      className="spec-image"
                      src={`data:image/png;base64,${prediction.spectrogram_base64}`}
                      alt="Espectrograma de Mel"
                    />
                  )}
                </div>
              </div>

              <div className="window-card">
                <div className="window-bar">
                  <div>
                    <h3 className="window-title">Top-5 Especies Más Probables</h3>
                    <p className="window-subtitle">
                      Distribución de probabilidad Softmax
                    </p>
                  </div>
                </div>
                <div className="window-body">
                  <div style={{ width: '100%', height: 230 }}>
                    <ResponsiveContainer>
                      <BarChart
                        layout="vertical"
                        data={(prediction.top_k || []).map((item) => ({
                          species: item.species,
                          pct: Number((item.probability * 100).toFixed(2)),
                        }))}
                        margin={{ top: 4, right: 22, left: 8, bottom: 4 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                        <XAxis
                          type="number"
                          domain={[0, 100]}
                          unit="%"
                          stroke={axisStroke}
                          fontSize={11}
                        />
                        <YAxis
                          type="category"
                          dataKey="species"
                          width={142}
                          stroke={axisStroke}
                          fontSize={11}
                        />
                        <Tooltip
                          formatter={(val) => [`${val}%`, 'Probabilidad']}
                          contentStyle={tooltipStyle}
                        />
                        <Bar dataKey="pct" radius={[0, 4, 4, 0]}>
                          {(prediction.top_k || []).map((_, idx) => (
                            <Cell
                              key={idx}
                              fill={idx === 0 ? primaryBar : secondaryBar}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
