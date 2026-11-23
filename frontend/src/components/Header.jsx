import React from 'react';
import {
  AudioLines,
  BarChart3,
  Bird,
  Database,
  Lock,
  Moon,
  Sun,
} from 'lucide-react';

const TABS = [
  { id: 'detector', label: 'Identificar Ave', icon: AudioLines },
  { id: 'dataset', label: 'Dataset', icon: Database },
  { id: 'metrics', label: 'Métricas del Modelo', icon: BarChart3 },
  {
    id: 'unsupervised',
    label: 'No Supervisado',
    badge: 'Próximamente',
    icon: Lock,
  },
];

export default function Header({
  activeTab,
  onTabChange,
  health,
  healthError,
  theme,
  onToggleTheme,
}) {
  let statusDot = 'offline';
  let statusLabel = 'Backend desconectado';

  if (health) {
    if (health.model_loaded) {
      statusDot = 'online';
      statusLabel = `Modelo listo · ${health.num_classes} especies`;
    } else {
      statusDot = 'warning';
      statusLabel = 'API activa · Sin modelo';
    }
  } else if (healthError) {
    statusDot = 'offline';
    statusLabel = 'Sin conexión con API';
  }

  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="header-top">
          <div className="brand-block">
            <div className="brand-icon">
              <Bird size={20} />
            </div>
            <div>
              <h1 className="brand-title">Identificador de Aves de Bolivia</h1>
              <p className="brand-subtitle">
                Clasificación acústica de cantos y llamados (39 especies)
              </p>
            </div>
          </div>

          <div className="header-actions">
            <div className="status-pill" title="Estado del servidor local FastAPI">
              <span className={`status-dot ${statusDot}`} />
              <span>{statusLabel}</span>
            </div>

            <button
              type="button"
              className="theme-toggle-btn"
              onClick={onToggleTheme}
              title={
                theme === 'light' ? 'Cambiar a modo oscuro' : 'Cambiar a modo claro'
              }
            >
              {theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}
              <span>{theme === 'light' ? 'Modo oscuro' : 'Modo claro'}</span>
            </button>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="Navegación principal">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                className={`nav-tab ${isActive ? 'active' : ''}`}
                onClick={() => onTabChange(tab.id)}
              >
                <Icon size={15} />
                <span>{tab.label}</span>
                {tab.badge && <span className="tab-badge">{tab.badge}</span>}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
