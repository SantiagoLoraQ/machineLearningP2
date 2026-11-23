import React, { useEffect, useState } from 'react';
import { CalendarClock, FolderGit2, Layers, Lock } from 'lucide-react';
import { getUnsupervisedStatus } from '../api/client';

export default function UnsupervisedPlaceholder() {
  const [statusData, setStatusData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    getUnsupervisedStatus()
      .then(setStatusData)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <div className="section-stack">
      <div className="placeholder-box">
        <div className="pill-list" style={{ marginBottom: 12 }}>
          <span className="badge badge-amber">
            <Lock size={13} />
            Espacio Reservado · No Ejecutado en Primera Entrega
          </span>
          <span className="badge badge-neutral">
            <CalendarClock size={13} />
            {statusData?.delivery || 'Siguiente Entrega'}
          </span>
          <span className="badge badge-green">
            <FolderGit2 size={13} />
            {statusData?.planned_directory || 'src/unsupervised/'}
          </span>
        </div>

        <h2 style={{ margin: '0 0 8px 0', fontSize: '1.25rem' }}>
          {statusData?.title ||
            'Próximamente: Aprendizaje No Supervisado (Siguiente Entrega)'}
        </h2>
        <p style={{ margin: 0, color: 'var(--text-secondary)', maxWidth: 820 }}>
          {statusData?.description ||
            'De acuerdo con el alcance establecido para la Primera Entrega, el modelo de Aprendizaje No Supervisado no ha sido implementado ni entrenado todavía. Esta sección y la carpeta src/unsupervised/ quedan reservadas para la siguiente entrega del proyecto.'}
        </p>

        {error && (
          <p style={{ marginTop: 10, color: 'var(--danger)', fontSize: '0.84rem' }}>
            Nota de conexión API: {error}
          </p>
        )}
      </div>

      <div className="grid-2">
        <div className="window-card">
          <div className="window-bar">
            <h3 className="window-title">
              <Layers size={16} color="var(--amber)" />
              Componentes Planificados para la Siguiente Entrega
            </h3>
          </div>
          <div className="window-body">
            <ul style={{ margin: 0, paddingLeft: 20, color: 'var(--text-secondary)' }}>
              {(
                statusData?.planned_components || [
                  'Extracción de embeddings acústicos sin utilizar etiquetas de especie',
                  'Reducción dimensional exploratoria (PCA / UMAP)',
                  'Agrupamiento no supervisado (K-Means / HDBSCAN) y evaluación con Silhouette Score',
                  'Visualización interactiva de clústeres acústicos en esta pestaña',
                ]
              ).map((item, idx) => (
                <li key={idx} style={{ marginBottom: 8 }}>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="window-card">
          <div className="window-bar">
            <h3 className="window-title">
              <FolderGit2 size={16} color="var(--green-primary)" />
              Contrato de Arquitectura Reservado
            </h3>
          </div>
          <div className="window-body">
            <p style={{ margin: '0 0 12px 0', color: 'var(--text-secondary)' }}>
              El módulo supervisado actual (<code>CNN 2D en PyTorch</code>) y el futuro
              módulo no supervisado compartirán el mismo dataset depurado (
              <code>data/processed/dataset_limpio.csv</code>) y la misma configuración
              portable de rutas (<code>src/config.py</code>).
            </p>
            <div className="pill-list">
              <span className="badge badge-neutral">Endpoint: GET /api/unsupervised/status</span>
              <span className="badge badge-neutral">Paquete: src/unsupervised/</span>
              <span className="badge badge-neutral">implemented: false</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
