import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { AlertTriangle, Award, Cpu, Grid, TrendingUp } from 'lucide-react';

function cellBackground(val, maxVal, isDiag, isDark) {
  if (val === 0) return isDark ? '#161f19' : '#eef4ef';
  if (isDiag) {
    const alpha = Math.min(1, 0.35 + 0.65 * (val / Math.max(1, maxVal)));
    return isDark
      ? `rgba(61, 139, 99, ${alpha.toFixed(2)})`
      : `rgba(45, 106, 79, ${alpha.toFixed(2)})`;
  }
  const alpha = Math.min(1, 0.32 + 0.68 * (val / Math.max(1, maxVal)));
  return isDark
    ? `rgba(207, 102, 102, ${alpha.toFixed(2)})`
    : `rgba(181, 72, 72, ${alpha.toFixed(2)})`;
}

export default function ModelMetrics({ metricsData, loading, error, theme = 'light' }) {
  const [hoveredCell, setHoveredCell] = useState(null);
  const [highlightSpecies, setHighlightSpecies] = useState('');
  const [sortField, setSortField] = useState('f1_score');
  const [sortAsc, setSortAsc] = useState(true);

  const sortedReport = useMemo(() => {
    const rows = [...(metricsData?.classification_report || [])];
    rows.sort((a, b) => {
      const av = a[sortField];
      const bv = b[sortField];
      if (typeof av === 'string') {
        return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      return sortAsc ? av - bv : bv - av;
    });
    return rows;
  }, [metricsData, sortField, sortAsc]);

  if (loading) {
    return <div className="empty-state">Cargando métricas del modelo CNN 2D...</div>;
  }
  if (error) {
    return (
      <div className="error-banner">
        No se pudieron cargar las métricas: {error}
      </div>
    );
  }
  if (!metricsData?.test_metrics) {
    return <div className="empty-state">Aún no existen métricas entrenadas.</div>;
  }

  const isDark = theme === 'dark';
  const gridStroke = isDark ? '#28392e' : '#d5e2d7';
  const axisStroke = isDark ? '#a4b8aa' : '#4a5e50';
  const greenMain = isDark ? '#3d8b63' : '#2d6a4f';
  const greenSec = isDark ? '#89bfa1' : '#74a988';
  const amberColor = isDark ? '#c79a49' : '#9e7425';
  const tooltipStyle = {
    backgroundColor: isDark ? '#1a231d' : '#ffffff',
    borderColor: isDark ? '#2f4236' : '#cddbce',
    color: isDark ? '#e7efe9' : '#1c2a21',
    borderRadius: 4,
  };

  const tm = metricsData.test_metrics;
  const history = metricsData.training_history?.history || [];
  const cm = metricsData.confusion_matrix || { labels: [], matrix: [], top_confusions: [] };
  const qa = metricsData.quality_error_analysis || { by_quality: [], by_vocalization: [] };

  const labels = cm.labels || [];
  const matrix = cm.matrix || [];
  let maxVal = 1;
  matrix.forEach((row) =>
    row.forEach((v) => {
      if (v > maxVal) maxVal = v;
    })
  );

  function toggleSort(field) {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(field === 'especie');
    }
  }

  return (
    <div className="section-stack">
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">
            <span>Accuracy en Prueba (Test)</span>
            <Award size={15} color="var(--green-primary)" />
          </div>
          <div className="kpi-value">{(tm.accuracy * 100).toFixed(2)}%</div>
          <div className="kpi-sub">
            {tm.num_test_recordings} grabaciones independientes ({tm.num_classes} clases)
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">
            <span>Top-3 / Top-5 Accuracy</span>
            <TrendingUp size={15} color="var(--green-primary)" />
          </div>
          <div className="kpi-value">
            {(tm.top3_accuracy * 100).toFixed(1)}% / {(tm.top5_accuracy * 100).toFixed(1)}%
          </div>
          <div className="kpi-sub">
            Azar uniforme (39 clases): {(100 / tm.num_classes).toFixed(2)}%
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">
            <span>F1-Score Macro / Ponderado</span>
            <Award size={15} color="var(--green-primary)" />
          </div>
          <div className="kpi-value">
            {(tm.f1_macro * 100).toFixed(2)}% / {(tm.f1_weighted * 100).toFixed(2)}%
          </div>
          <div className="kpi-sub">
            Precision: {(tm.precision_macro * 100).toFixed(1)}% · Recall: {(tm.recall_macro * 100).toFixed(1)}%
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">
            <span>Test Loss y Arquitectura</span>
            <Cpu size={15} color="var(--green-primary)" />
          </div>
          <div className="kpi-value">{tm.test_loss.toFixed(4)}</div>
          <div className="kpi-sub">
            Mejor época: {tm.best_epoch} · {tm.model_params?.toLocaleString()} parámetros
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="window-card">
          <div className="window-bar">
            <div>
              <h3 className="window-title">Curva de Pérdida (Loss) por Época</h3>
              <p className="window-subtitle">
                Entrenamiento vs Validación ({metricsData.training_history?.epochs_completed} épocas)
              </p>
            </div>
          </div>
          <div className="window-body">
            <div style={{ width: '100%', height: 260 }}>
              <ResponsiveContainer>
                <LineChart data={history} margin={{ top: 10, right: 16, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                  <XAxis dataKey="epoch" stroke={axisStroke} fontSize={12} />
                  <YAxis stroke={axisStroke} fontSize={12} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="train_loss"
                    name="Train Loss"
                    stroke={greenSec}
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_loss"
                    name="Val Loss (Grabación)"
                    stroke={amberColor}
                    strokeWidth={2.5}
                    dot={{ r: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="window-card">
          <div className="window-bar">
            <div>
              <h3 className="window-title">Curva de Precisión (Accuracy) por Época</h3>
              <p className="window-subtitle">
                Mejor validación en época {tm.best_epoch}:{' '}
                {((metricsData.training_history?.best_val_accuracy || 0) * 100).toFixed(2)}%
              </p>
            </div>
          </div>
          <div className="window-body">
            <div style={{ width: '100%', height: 260 }}>
              <ResponsiveContainer>
                <LineChart data={history} margin={{ top: 10, right: 16, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                  <XAxis dataKey="epoch" stroke={axisStroke} fontSize={12} />
                  <YAxis domain={[0, 1]} stroke={axisStroke} fontSize={12} />
                  <Tooltip
                    formatter={(v) => `${(v * 100).toFixed(2)}%`}
                    contentStyle={tooltipStyle}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="train_accuracy"
                    name="Train Accuracy"
                    stroke={greenSec}
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_accuracy"
                    name="Val Accuracy (Grabación)"
                    stroke={greenMain}
                    strokeWidth={2.5}
                    dot={{ r: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="window-card">
          <div className="window-bar">
            <div>
              <h3 className="window-title">Desempeño por Calidad de Grabación (A / B / C)</h3>
              <p className="window-subtitle">
                Evaluación de robustez frente a ruido de campo en prueba
              </p>
            </div>
          </div>
          <div className="window-body">
            <div style={{ width: '100%', height: 245 }}>
              <ResponsiveContainer>
                <BarChart
                  data={(qa.by_quality || []).map((q) => ({
                    ...q,
                    name: `Calidad ${q.calidad} (n=${q.count})`,
                    acc_pct: Number((q.accuracy * 100).toFixed(1)),
                    conf_pct: Number((q.avg_confidence * 100).toFixed(1)),
                  }))}
                  margin={{ top: 10, right: 16, left: 0, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                  <XAxis dataKey="name" stroke={axisStroke} fontSize={12} />
                  <YAxis domain={[0, 100]} unit="%" stroke={axisStroke} fontSize={12} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend />
                  <Bar
                    dataKey="acc_pct"
                    name="Accuracy (%)"
                    fill={greenMain}
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="conf_pct"
                    name="Confianza Media (%)"
                    fill={greenSec}
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="window-card">
          <div className="window-bar">
            <div>
              <h3 className="window-title">Desempeño por Tipo de Vocalización</h3>
              <p className="window-subtitle">
                Exactitud segmentada entre cantos, llamados, dúos, etc.
              </p>
            </div>
          </div>
          <div className="window-body">
            <div style={{ width: '100%', height: 245 }}>
              <ResponsiveContainer>
                <BarChart
                  data={(qa.by_vocalization || []).map((v) => ({
                    ...v,
                    label: `${v.tipo_vocalizacion} (n=${v.count})`,
                    acc_pct: Number((v.accuracy * 100).toFixed(1)),
                  }))}
                  margin={{ top: 10, right: 16, left: 0, bottom: 16 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                  <XAxis dataKey="label" stroke={axisStroke} fontSize={11} />
                  <YAxis domain={[0, 100]} unit="%" stroke={axisStroke} fontSize={12} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar
                    dataKey="acc_pct"
                    name="Accuracy (%)"
                    fill={amberColor}
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      <div className="window-card">
        <div className="window-bar">
          <div>
            <h3 className="window-title">
              <Grid size={16} color="var(--green-primary)" />
              Matriz de Confusión Interactiva ({labels.length} × {labels.length})
            </h3>
            <p className="window-subtitle">
              Verde = aciertos en diagonal · Rojo = confusiones fuera de diagonal. Pasa el cursor sobre cualquier celda.
            </p>
          </div>
          <div className="controls-row">
            <select
              className="select"
              value={highlightSpecies}
              onChange={(e) => setHighlightSpecies(e.target.value)}
            >
              <option value="">Resaltar especie: Ninguna</option>
              {labels.map((sp) => (
                <option key={sp} value={sp}>
                  {sp}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="window-body">
          <div className="cm-container">
            <div
              className="cm-grid"
              style={{
                gridTemplateColumns: `repeat(${labels.length}, 17px)`,
              }}
            >
              {matrix.map((row, i) =>
                row.map((val, j) => {
                  const trueSp = labels[i];
                  const predSp = labels[j];
                  const isDiag = i === j;
                  const dimmed =
                    highlightSpecies &&
                    trueSp !== highlightSpecies &&
                    predSp !== highlightSpecies;
                  return (
                    <div
                      key={`${i}-${j}`}
                      className="cm-cell"
                      style={{
                        backgroundColor: cellBackground(val, maxVal, isDiag, isDark),
                        opacity: dimmed ? 0.22 : 1,
                      }}
                      onMouseEnter={() =>
                        setHoveredCell({ trueSp, predSp, count: val, isDiag })
                      }
                    />
                  );
                })
              )}
            </div>
          </div>

          <div className="cm-tooltip">
            {hoveredCell ? (
              <>
                <span
                  className={`badge ${
                    hoveredCell.isDiag
                      ? 'badge-green'
                      : hoveredCell.count > 0
                      ? 'badge-danger'
                      : 'badge-neutral'
                  }`}
                >
                  Conteo: {hoveredCell.count}
                </span>
                <span>
                  Real: <strong className="italic-species">{hoveredCell.trueSp}</strong> → Predicha:{' '}
                  <strong className="italic-species">{hoveredCell.predSp}</strong>
                </span>
              </>
            ) : (
              <span style={{ color: 'var(--text-muted)' }}>
                Inspecciona una celda de la matriz {labels.length}×{labels.length} para ver la especie real y predicha.
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="window-card">
          <div className="window-bar">
            <div>
              <h3 className="window-title">
                <AlertTriangle size={16} color="var(--amber)" />
                Pares de Especies con Mayor Confusión
              </h3>
              <p className="window-subtitle">
                Errores más frecuentes en el conjunto de prueba independiente
              </p>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Especie Real</th>
                  <th>Especie Predicha</th>
                  <th>Confusiones</th>
                </tr>
              </thead>
              <tbody>
                {(cm.top_confusions || []).map((item, idx) => (
                  <tr key={idx}>
                    <td className="italic-species">{item.true_species}</td>
                    <td className="italic-species">{item.predicted_species}</td>
                    <td>
                      <span className="badge badge-danger">{item.count}</span>
                    </td>
                  </tr>
                ))}
                {(cm.top_confusions || []).length === 0 && (
                  <tr>
                    <td colSpan={3} style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                      No se registraron confusiones.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="window-card">
          <div className="window-bar">
            <div>
              <h3 className="window-title">Reporte de Clasificación por Especie</h3>
              <p className="window-subtitle">
                Haz clic en los encabezados para ordenar
              </p>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th className="sortable" onClick={() => toggleSort('especie')}>
                    Especie
                  </th>
                  <th className="sortable" onClick={() => toggleSort('precision')}>
                    Precision
                  </th>
                  <th className="sortable" onClick={() => toggleSort('recall')}>
                    Recall
                  </th>
                  <th className="sortable" onClick={() => toggleSort('f1_score')}>
                    F1-Score
                  </th>
                  <th className="sortable" onClick={() => toggleSort('support')}>
                    Soporte
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedReport.map((row) => (
                  <tr key={row.especie}>
                    <td className="italic-species">{row.especie}</td>
                    <td>{(row.precision * 100).toFixed(1)}%</td>
                    <td>{(row.recall * 100).toFixed(1)}%</td>
                    <td>
                      <span
                        className={`badge ${
                          row.f1_score >= 0.75
                            ? 'badge-green'
                            : row.f1_score >= 0.45
                            ? 'badge-amber'
                            : 'badge-danger'
                        }`}
                      >
                        {(row.f1_score * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td>{row.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
