import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { CheckCircle2, Clock, FileAudio, Layers, Search } from 'lucide-react';

export default function DatasetExplorer({ summary, loading, error, theme = 'light' }) {
  const [speciesFilter, setSpeciesFilter] = useState('');

  const filteredSpecies = useMemo(() => {
    if (!summary?.species_breakdown) return [];
    const q = speciesFilter.trim().toLowerCase();
    if (!q) return summary.species_breakdown;
    return summary.species_breakdown.filter((row) =>
      row.especie.toLowerCase().includes(q)
    );
  }, [summary, speciesFilter]);

  if (loading) {
    return <div className="empty-state">Cargando estadísticas del dataset...</div>;
  }
  if (error) {
    return (
      <div className="error-banner">
        No se pudo cargar el resumen del dataset: {error}
      </div>
    );
  }
  if (!summary) {
    return <div className="empty-state">Sin datos disponibles.</div>;
  }

  const isDark = theme === 'dark';
  const gridStroke = isDark ? '#28392e' : '#d5e2d7';
  const axisStroke = isDark ? '#a4b8aa' : '#4a5e50';
  const tooltipStyle = {
    backgroundColor: isDark ? '#1a231d' : '#ffffff',
    borderColor: isDark ? '#2f4236' : '#cddbce',
    color: isDark ? '#e7efe9' : '#1c2a21',
    borderRadius: 4,
  };

  const qualityColors = isDark
    ? { A: '#3d8b63', B: '#64a883', C: '#c79a49' }
    : { A: '#2d6a4f', B: '#529471', C: '#b0893e' };

  const greenPalette = isDark
    ? ['#3d8b63', '#529c76', '#64a883', '#7ab896', '#94c7aa', '#7b9182']
    : ['#2d6a4f', '#3d7f5e', '#4b8b68', '#67a07f', '#87b59a', '#6c7f71'];

  const qualityData = Object.entries(summary.quality_distribution || {}).map(
    ([name, value]) => ({
      name: `Calidad ${name}`,
      code: name,
      value,
    })
  );

  const vocData = Object.entries(summary.vocalization_distribution || {}).map(
    ([name, value]) => ({
      name,
      count: value,
    })
  );

  const audit = summary.cleaning_audit || {};
  const exclusions = audit.exclusion_summary || {};
  const durStats = summary.duration_stats || {};

  return (
    <div className="section-stack">
      <div className="audit-banner">
        <div>
          <strong style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <CheckCircle2 size={17} color="var(--green-primary)" />
            Depuración Automatizada: {audit.raw_total_rows ?? 1000} registros crudos →{' '}
            {summary.total_clean_records} grabaciones válidas ({summary.num_species} especies)
          </strong>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: 4 }}>
            Archivo generado: <code>{audit.output_csv || 'data/processed/dataset_limpio.csv'}</code>
          </div>
        </div>
        <div className="pill-list">
          <span className="badge badge-amber">
            Mystery mystery excluidos: {exclusions.mystery_mystery || 0}
          </span>
          <span className="badge badge-danger">
            Faltantes/corruptos:{' '}
            {(exclusions.missing_file || 0) + (exclusions.corrupted_or_unreadable || 0)}
          </span>
          <span className="badge badge-green">
            MP3: {summary.audio_format_distribution?.mp3 || 0} · WAV:{' '}
            {summary.audio_format_distribution?.wav || 0}
          </span>
        </div>
      </div>

      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">
            <span>Grabaciones Limpias</span>
            <FileAudio size={15} color="var(--green-primary)" />
          </div>
          <div className="kpi-value">{summary.total_clean_records}</div>
          <div className="kpi-sub">
            De {audit.raw_total_rows ?? 1000} filas en manifest_verificado.csv
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">
            <span>Especies de Bolivia</span>
            <Layers size={15} color="var(--green-primary)" />
          </div>
          <div className="kpi-value">{summary.num_species}</div>
          <div className="kpi-sub">Clases supervisadas verificadas</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">
            <span>Calidad A / B / C</span>
            <CheckCircle2 size={15} color="var(--green-primary)" />
          </div>
          <div className="kpi-value">
            {summary.quality_distribution?.A || 0} / {summary.quality_distribution?.B || 0} /{' '}
            {summary.quality_distribution?.C || 0}
          </div>
          <div className="kpi-sub">Etiquetas Xeno-canto en disco</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">
            <span>Mediana de Duración</span>
            <Clock size={15} color="var(--green-primary)" />
          </div>
          <div className="kpi-value">{durStats.median_sec ?? 0} s</div>
          <div className="kpi-sub">
            Rango: {durStats.min_sec ?? 0}s – {durStats.max_sec ?? 0}s ({durStats.total_hours ?? 0} h)
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="window-card">
          <div className="window-bar">
            <div>
              <h3 className="window-title">Distribución por Calidad de Grabación</h3>
              <p className="window-subtitle">
                Calidad A (alta), B (intermedia) y C (mayor ruido ambiental)
              </p>
            </div>
          </div>
          <div className="window-body">
            <div style={{ width: '100%', height: 260 }}>
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={qualityData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={52}
                    outerRadius={88}
                    paddingAngle={2}
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {qualityData.map((entry) => (
                      <Cell
                        key={entry.code}
                        fill={qualityColors[entry.code] || '#6c7f71'}
                      />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="window-card">
          <div className="window-bar">
            <div>
              <h3 className="window-title">Tipos de Vocalización Principal</h3>
              <p className="window-subtitle">
                Agrupación determinística desde <code>tipo_vocalizacion</code>
              </p>
            </div>
          </div>
          <div className="window-body">
            <div style={{ width: '100%', height: 260 }}>
              <ResponsiveContainer>
                <BarChart data={vocData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                  <XAxis dataKey="name" stroke={axisStroke} fontSize={12} />
                  <YAxis stroke={axisStroke} fontSize={12} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="count" name="Grabaciones" radius={[4, 4, 0, 0]}>
                    {vocData.map((entry, idx) => (
                      <Cell
                        key={entry.name}
                        fill={greenPalette[idx % greenPalette.length]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="window-card">
          <div className="window-bar">
            <div>
              <h3 className="window-title">Distribución de Duraciones Reales</h3>
              <p className="window-subtitle">
                Medición verificada directamente con <code>soundfile</code>
              </p>
            </div>
          </div>
          <div className="window-body">
            <div style={{ width: '100%', height: 270 }}>
              <ResponsiveContainer>
                <BarChart
                  data={summary.duration_histogram || []}
                  margin={{ top: 10, right: 16, left: 0, bottom: 16 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                  <XAxis dataKey="range" stroke={axisStroke} fontSize={11} />
                  <YAxis stroke={axisStroke} fontSize={12} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar
                    dataKey="count"
                    name="Audios"
                    fill={greenPalette[1]}
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
              <h3 className="window-title">Muestras por Especie (Calidad Apilada)</h3>
              <p className="window-subtitle">
                Entre 21 y 27 grabaciones en las {summary.num_species} especies
              </p>
            </div>
            <div className="controls-row">
              <Search size={14} color="var(--text-muted)" />
              <input
                type="text"
                className="input"
                placeholder="Filtrar especie..."
                value={speciesFilter}
                onChange={(e) => setSpeciesFilter(e.target.value)}
              />
            </div>
          </div>
          <div className="window-body">
            <div style={{ width: '100%', height: 270 }}>
              <ResponsiveContainer>
                <BarChart
                  data={filteredSpecies}
                  margin={{ top: 8, right: 16, left: 0, bottom: 48 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                  <XAxis
                    dataKey="especie"
                    stroke={axisStroke}
                    fontSize={10}
                    angle={-35}
                    textAnchor="end"
                    interval={0}
                    height={62}
                  />
                  <YAxis stroke={axisStroke} fontSize={12} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend verticalAlign="top" height={28} />
                  <Bar dataKey="quality_a" stackId="q" name="Calidad A" fill={qualityColors.A} />
                  <Bar dataKey="quality_b" stackId="q" name="Calidad B" fill={qualityColors.B} />
                  <Bar dataKey="quality_c" stackId="q" name="Calidad C" fill={qualityColors.C} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      <div className="window-card">
        <div className="window-bar">
          <div>
            <h3 className="window-title">Catálogo Detallado de las {summary.num_species} Especies</h3>
            <p className="window-subtitle">
              Mostrando {filteredSpecies.length} especies del dataset depurado
            </p>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Especie (Nombre Científico)</th>
                <th>Total</th>
                <th>Calidad A</th>
                <th>Calidad B</th>
                <th>Calidad C</th>
                <th>Duración Media (s)</th>
                <th>Duración Mediana (s)</th>
              </tr>
            </thead>
            <tbody>
              {filteredSpecies.map((row, idx) => (
                <tr key={row.especie}>
                  <td>{idx + 1}</td>
                  <td className="italic-species">{row.especie}</td>
                  <td>
                    <strong>{row.total}</strong>
                  </td>
                  <td>{row.quality_a}</td>
                  <td>{row.quality_b}</td>
                  <td>{row.quality_c}</td>
                  <td>{row.avg_duration_sec}</td>
                  <td>{row.median_duration_sec}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
