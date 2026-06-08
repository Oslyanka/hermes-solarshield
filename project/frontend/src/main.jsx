import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

const navItems = [
  { id: 'overview', label: 'Inicio', icon: 'file' },
  { id: 'dashboard', label: 'Painel', icon: 'grid' },
  { id: 'solar', label: 'Analise Solar', icon: 'scan' },
  { id: 'assistant', label: 'Assistente', icon: 'spark' },
];

function Icon({ name }) {
  const common = { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8 };
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    scan: <><path d="M4 7V5a1 1 0 0 1 1-1h2" /><path d="M17 4h2a1 1 0 0 1 1 1v2" /><path d="M20 17v2a1 1 0 0 1-1 1h-2" /><path d="M7 20H5a1 1 0 0 1-1-1v-2" /><circle cx="12" cy="12" r="4" /></>,
    spark: <><path d="M12 2l1.8 5.7L19 10l-5.2 2.3L12 18l-1.8-5.7L5 10l5.2-2.3L12 2z" /><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15z" /></>,
    bot: <><rect x="5" y="8" width="14" height="11" rx="3" /><path d="M12 8V4" /><circle cx="9" cy="13" r="1" /><circle cx="15" cy="13" r="1" /><path d="M9 17h6" /></>,
    file: <><path d="M6 3h8l4 4v14H6z" /><path d="M14 3v5h5" /><path d="M9 13h6" /><path d="M9 17h6" /></>,
  };
  return <svg {...common}>{paths[name]}</svg>;
}

function riskClass(risk) {
  return String(risk || 'LOW').toLowerCase();
}

function fmtPercent(value) {
  if (value === undefined || value === null) return '0%';
  return `${Math.round(Number(value) * 100)}%`;
}

function normalizeAnalysis(item) {
  if (!item) return null;
  let metadata = {};
  if (typeof item.metadata === 'string' && item.metadata) {
    try {
      metadata = JSON.parse(item.metadata);
    } catch {
      metadata = {};
    }
  } else if (item.metadata && typeof item.metadata === 'object') {
    metadata = item.metadata;
  }
  return {
    ...item,
    features: item.features || metadata.features || {},
    class_probabilities: item.class_probabilities || metadata.class_probabilities,
    flare_probabilities: item.flare_probabilities || metadata.flare_probabilities,
  };
}

async function apiJson(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Erro HTTP ${response.status}`);
  }
  return response.json();
}

function App() {
  const [page, setPage] = useState('overview');
  const [health, setHealth] = useState(null);
  const [analyses, setAnalyses] = useState([]);
  const [reports, setReports] = useState([]);
  const [modelStatus, setModelStatus] = useState(null);
  const [spaceWeather, setSpaceWeather] = useState(null);
  const [latest, setLatest] = useState(null);
  const [toast, setToast] = useState('');

  const refresh = async () => {
    const [healthData, analysesData, reportsData, modelData] = await Promise.all([
      apiJson('/health').catch(() => null),
      apiJson('/analyses').catch(() => []),
      apiJson('/reports').catch(() => []),
      apiJson('/model/status').catch(() => null),
    ]);
    setHealth(healthData);
    const normalizedAnalyses = analysesData.map(normalizeAnalysis);
    setAnalyses(normalizedAnalyses);
    setReports(reportsData);
    setModelStatus(modelData);
    setLatest(normalizedAnalyses[0] || null);
  };

  useEffect(() => {
    refresh();
    const loadExternal = () => apiJson('/spaceweatherlive').then(setSpaceWeather).catch(() => null);
    if ('requestIdleCallback' in window) {
      window.requestIdleCallback(loadExternal, { timeout: 3500 });
    } else {
      window.setTimeout(loadExternal, 900);
    }
  }, []);

  const notify = (message) => {
    setToast(message);
    window.setTimeout(() => setToast(''), 3600);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <img src="/hermes-logo.png" alt="Hermes" />
          <div>
            <strong>Hermes</strong>
            <span>SolarShield</span>
          </div>
        </div>
        <nav>
          {navItems.map((item) => (
            <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => setPage(item.id)} title={item.label}>
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-status">
          <span className="pulse" />
          <div>
            <strong>{health?.status === 'ok' ? 'Monitoramento ativo' : 'API offline'}</strong>
            <small>Modo {health?.mode || 'demo'}</small>
          </div>
        </div>
      </aside>

      <main className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">Projeto Hermes / Space Weather Intelligence</p>
            <h1>{navItems.find((item) => item.id === page)?.label}</h1>
          </div>
          <div className={`risk-pill ${riskClass(latest?.risk_level)}`}>
            <span>Risco atual</span>
            <strong>{latest?.risk_level || 'SEM DADOS'}</strong>
          </div>
        </header>

        {page === 'overview' && <Overview modelStatus={modelStatus} spaceWeather={spaceWeather} goToSolar={() => setPage('solar')} goToDashboard={() => setPage('dashboard')} goToAssistant={() => setPage('assistant')} />}
        {page === 'dashboard' && <Dashboard latest={latest} analyses={analyses} health={health} reports={reports} modelStatus={modelStatus} spaceWeather={spaceWeather} goToSolar={() => setPage('solar')} />}
        {page === 'solar' && <SolarAnalysis latest={latest} setLatest={setLatest} refresh={refresh} notify={notify} modelStatus={modelStatus} spaceWeather={spaceWeather} />}
        {page === 'assistant' && <Assistant latest={latest} reports={reports} refresh={refresh} notify={notify} />}
      </main>
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

function Overview({ modelStatus, spaceWeather, goToSolar, goToDashboard, goToAssistant }) {
  const sections = [
    {
      title: 'Leitura visual',
      items: [
        'Upload ou templates solares validados no canal SDO AIA 131.',
        'Classificacao LOW/MEDIUM/HIGH com probabilidades diretas.',
        'Historico de analises persistido para auditoria.',
      ],
    },
    {
      title: 'Apoio tecnico',
      items: [
        'Explicacao tecnica baseada na ultima analise.',
        'Resumo simples e plano de mitigacao operacional.',
        'Geracao de relatorio em TXT e PDF.',
      ],
    },
  ];

  return (
    <section className="job-layout">
      <div className="job-header">
        <img src="/hermes-logo.png" alt="Hermes" />
        <p className="eyebrow">Projeto Hermes / Space Weather Intelligence</p>
        <h2>Hermes SolarShield</h2>
        <div className="job-meta">
          <span>Webapp academico</span>
          <span>Analise solar + assistente tecnico</span>
          <span>{modelStatus?.mode === 'trained' ? 'Modelo treinado' : 'Modo demo'}</span>
          <span>Fluxo auditavel</span>
        </div>
        <div className="job-actions">
          <button className="primary-button" onClick={goToSolar}>Analisar imagem solar</button>
          <button className="secondary-button" onClick={goToAssistant}>Abrir assistente</button>
          <button className="secondary-button" onClick={goToDashboard}>Ver painel</button>
        </div>
      </div>

      <article className="job-content">
        <p>
          Hermes SolarShield simula uma plataforma de clima espacial para analisar imagens solares,
          estimar risco operacional, explicar o resultado e gerar relatorios de apoio a decisao.
        </p>

        <section className="job-section">
          <h3>Status operacional</h3>
          <div className="status-grid">
            <div><span>Modelo</span><strong>{modelStatus?.mode === 'trained' ? 'Treinado' : 'Demo'}</strong></div>
            <div><span>Saida do modelo</span><strong>Sem ajuste externo</strong></div>
            <div><span>SWL C-class</span><strong>{spaceWeather?.probabilities?.C ?? '--'}%</strong></div>
          </div>
        </section>

        {sections.map((section) => (
          <section className="job-section" key={section.title}>
            <h3>{section.title}</h3>
            <ul>
              {section.items.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </section>
        ))}

        <section className="job-section">
          <h3>Fluxo da demonstracao</h3>
          <ol>
            <li>Selecionar uma imagem solar validada.</li>
            <li>Mostrar a classe LOW/MEDIUM/HIGH e as probabilidades diretas do modelo.</li>
            <li>Gerar explicacao, resumo e plano de mitigacao.</li>
            <li>Criar um relatorio operacional da ultima analise.</li>
          </ol>
        </section>
      </article>
    </section>
  );
}

function Dashboard({ latest, analyses, health, reports, modelStatus, spaceWeather, goToSolar }) {
  const cTarget = spaceWeather?.probabilities?.C;
  const cards = [
    { label: 'Ultima previsao', value: latest ? `#${latest.id}` : 'Aguardando', detail: latest?.created_at || 'Execute uma analise' },
    { label: 'Nivel de risco atual', value: latest?.risk_level || 'N/A', detail: latest ? `Score ${Number(latest.score).toFixed(3)}` : 'Sem previsao ativa', risk: latest?.risk_level },
    { label: 'Total de analises', value: analyses.length, detail: 'Registros no SQLite' },
    { label: 'Status do monitoramento', value: health?.status === 'ok' ? 'Online' : 'Offline', detail: `Backend ${health?.mode || 'demo'}` },
    { label: 'Modelo', value: modelStatus?.mode === 'trained' ? 'Treinado' : 'Demo', detail: 'Saida crua, sem calibracao externa' },
    { label: 'Relatorios', value: reports.length, detail: 'TXT e PDF simples' },
  ];
  return (
    <section className="page-grid">
      <div className="hero-panel">
        <div>
          <p className="eyebrow">Hermes SolarShield</p>
          <h2>Previsao e resposta a solar flares para infraestrutura orbital.</h2>
          <p>Analise imagens solares, consulte a trilha de resultados e gere explicacoes tecnicas em um fluxo unico.</p>
          <button className="primary-button hero-action" onClick={goToSolar}>Iniciar analise</button>
        </div>
        <img src="/hermes-logo-full.png" alt="Projeto Hermes" />
      </div>
      <div className="calibration-band">
        <div>
          <span>SpaceWeatherLive C-class</span>
          <strong>{cTarget ?? '--'}%</strong>
          <small>referencia externa</small>
        </div>
        <div>
          <span>Hermes modelo</span>
          <strong>{latest ? `${Math.round(Number(latest.score) * 100)}%` : '--'}</strong>
          <small>predicao sem ajuste externo</small>
        </div>
        <div>
          <span>Probabilidades SWL</span>
          <strong>C {spaceWeather?.probabilities?.C ?? '--'} / M {spaceWeather?.probabilities?.M ?? '--'} / X {spaceWeather?.probabilities?.X ?? '--'}</strong>
          <small>previsao publica</small>
        </div>
      </div>
      <div className="metric-grid">
        {cards.map((card) => (
          <article className={`metric-card ${card.risk ? riskClass(card.risk) : ''}`} key={card.label}>
            <span>{card.label}</span>
            <strong>{card.value}</strong>
            <small>{card.detail}</small>
          </article>
        ))}
      </div>
      <div className="wide-panel">
        <h3>Estado operacional</h3>
        <div className="timeline">
          {analyses.slice(0, 5).map((item) => (
            <div key={item.id} className="timeline-row">
              <span className={`dot ${riskClass(item.risk_level)}`} />
              <strong>{item.risk_level}</strong>
              <span>score {Number(item.score).toFixed(3)}</span>
              <small>{item.created_at}</small>
            </div>
          ))}
          {!analyses.length && <p className="muted">Nenhuma analise registrada ainda.</p>}
        </div>
      </div>
    </section>
  );
}

function SolarAnalysis({ latest, setLatest, refresh, notify, modelStatus, spaceWeather }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [samples, setSamples] = useState([]);
  const [busy, setBusy] = useState(false);
  const recommendedSamples = samples.filter((sample) => ['validated_aia_131', 'recommended_aia_131'].includes(sample.template));

  useEffect(() => {
    apiJson('/samples').then(setSamples).catch(() => setSamples([]));
  }, []);

  const selectFile = (event) => {
    const next = event.target.files?.[0];
    setFile(next || null);
    setPreview(next ? URL.createObjectURL(next) : '');
  };

  const loadSample = async (sample) => {
    const response = await fetch(`${API_BASE}${sample.url}`);
    const blob = await response.blob();
    const filename = sample.filename || `${sample.id}.jpg`;
    const type = filename.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';
    const next = new File([blob], filename, { type });
    setFile(next);
    setPreview(URL.createObjectURL(next));
    notify(`${sample.level || 'AIA 131'} carregado para analise.`);
  };

  const analyze = async () => {
    if (!file) {
      notify('Selecione uma imagem solar primeiro.');
      return;
    }
    setBusy(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const result = await apiJson('/predict', { method: 'POST', body: form });
      setLatest(result);
      await refresh();
      notify('Analise de solar flare concluida.');
    } catch (error) {
      notify(error.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="two-column">
      <div className="panel">
        <div className="panel-heading">
          <h3>Imagem solar</h3>
          <span className="status-chip">Templates validados</span>
        </div>
        <label className="drop-zone">
          <input type="file" accept="image/*" onChange={selectFile} />
          {preview ? <img src={preview} alt="Preview solar" /> : <span>Enviar imagem SDO AIA 131</span>}
        </label>
        <div className="template-grid">
          {recommendedSamples.map((sample) => (
            <button key={sample.id} className={`template-card ${riskClass(sample.level)}`} onClick={() => loadSample(sample)}>
              <span>{sample.channel || 'SDO AIA 131'}</span>
              <strong>{sample.level}</strong>
              <small>{sample.filename}</small>
            </button>
          ))}
        </div>
        <button className="primary-button" onClick={analyze} disabled={busy}>{busy ? 'Analisando...' : 'Analisar Solar Flare'}</button>
      </div>
      <div className="panel result-panel">
        <div className="panel-heading">
          <h3>Resultado do modelo</h3>
          <span className="status-chip">{modelStatus?.mode === 'trained' ? 'Modelo treinado' : 'Modo demo'}</span>
        </div>
        {latest ? (
          <>
            <div className={`risk-orbit ${riskClass(latest.risk_level)}`}>
              <strong>{latest.risk_level}</strong>
              <span>risco previsto 24h</span>
            </div>
            <div className="score-grid">
              {latest.class_probabilities ? (
                <>
                  <div><span>Prob. LOW</span><strong>{fmtPercent(latest.class_probabilities.LOW)}</strong></div>
                  <div><span>Prob. MEDIUM</span><strong>{fmtPercent(latest.class_probabilities.MEDIUM)}</strong></div>
                  <div><span>Prob. HIGH</span><strong>{fmtPercent(latest.class_probabilities.HIGH)}</strong></div>
                </>
              ) : (
                <>
                  <div><span>Chance C-class</span><strong>{fmtPercent(latest.flare_probabilities?.C ?? latest.score)}</strong></div>
                  <div><span>Chance M-class</span><strong>{fmtPercent(latest.flare_probabilities?.M)}</strong></div>
                  <div><span>Chance X-class</span><strong>{fmtPercent(latest.flare_probabilities?.X)}</strong></div>
                </>
              )}
              <div><span>Confianca</span><strong>{fmtPercent(latest.confidence)}</strong></div>
            </div>
            {!latest.class_probabilities && (
              <div className="swl-compare">
                <span>SpaceWeatherLive</span>
                <strong>
                  C {spaceWeather?.probabilities?.C ?? '--'}% /
                  M {spaceWeather?.probabilities?.M ?? '--'}% /
                  X {spaceWeather?.probabilities?.X ?? '--'}%
                </strong>
              </div>
            )}
            <p>{latest.explanation_short}</p>
            {!latest.class_probabilities && latest.features?.raw_flare_probabilities && (
              <p className="muted">
                Modelo visual bruto: C {fmtPercent(latest.features.raw_flare_probabilities.C)}, M {fmtPercent(latest.features.raw_flare_probabilities.M)}, X {fmtPercent(latest.features.raw_flare_probabilities.X)}.
              </p>
            )}
            {latest.features?.raw_score && (
              <p className="muted">Score bruto do modelo: {fmtPercent(latest.features.raw_score)}.</p>
            )}
          </>
        ) : (
          <p className="muted">O resultado aparecera aqui apos a primeira analise.</p>
        )}
      </div>
    </section>
  );
}

function Assistant({ latest, reports, refresh, notify }) {
  const [question, setQuestion] = useState('Explique o risco solar atual e o que a equipe deve fazer.');
  const [answer, setAnswer] = useState(null);
  const [busy, setBusy] = useState(false);
  const [reportBusy, setReportBusy] = useState(false);
  const [routineBusy, setRoutineBusy] = useState(false);
  const [routine, setRoutine] = useState(null);

  const ask = async () => {
    setBusy(true);
    try {
      const data = await apiJson('/copilot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, analysis_id: latest?.id }),
      });
      setAnswer(data);
    } catch (error) {
      notify(error.message);
    } finally {
      setBusy(false);
    }
  };

  const createReport = async () => {
    setReportBusy(true);
    try {
      await apiJson('/generate-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      await refresh();
      notify('Relatorio gerado.');
    } catch (error) {
      notify(error.message);
    } finally {
      setReportBusy(false);
    }
  };

  const runRoutine = async () => {
    setRoutineBusy(true);
    try {
      const result = await apiJson('/rpa/run', { method: 'POST' });
      setRoutine(result);
      await refresh();
      notify('Rotina operacional concluida.');
    } catch (error) {
      notify(error.message);
    } finally {
      setRoutineBusy(false);
    }
  };

  return (
    <section className="two-column">
      <div className="panel">
        <h3>Assistente tecnico</h3>
        <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows="7" />
        <div className="action-row">
          <button className="primary-button" onClick={ask} disabled={busy}>{busy ? 'Gerando...' : 'Gerar explicacao'}</button>
          <button className="secondary-button" onClick={createReport} disabled={reportBusy || !latest}>{reportBusy ? 'Gerando...' : 'Gerar relatorio'}</button>
          <button className="secondary-button" onClick={runRoutine} disabled={routineBusy}>{routineBusy ? 'Executando...' : 'Executar rotina'}</button>
        </div>
        <p className="muted">Contexto ativo: {latest ? `analise #${latest.id} / ${latest.risk_level}` : 'nenhuma analise'}</p>
      </div>
      <div className="panel response-panel">
        <h3>Saida operacional</h3>
        {answer ? (
          <>
            <section><strong>Resposta</strong><p>{answer.answer}</p></section>
            <section><strong>Resumo tecnico</strong><p>{answer.technical_summary}</p></section>
            <section><strong>Resumo para leigos</strong><p>{answer.plain_summary}</p></section>
            <section><strong>Plano de mitigacao</strong><p>{answer.mitigation_plan}</p></section>
          </>
        ) : (
          <p className="muted">Use a ultima analise para gerar explicacao tecnica, resumo simples e plano de mitigacao.</p>
        )}
        <section>
          <strong>Relatorios recentes</strong>
          {reports.slice(0, 3).map((report) => (
            <p key={report.id}>
              <a href={`${API_BASE}/reports/${report.id}/download/txt`}>{report.title}</a>
            </p>
          ))}
          {!reports.length && <p className="muted">Nenhum relatorio gerado ainda.</p>}
        </section>
        {routine && (
          <section>
            <strong>Ultima rotina</strong>
            <p>Analise #{routine.analysis.id}: risco {routine.analysis.risk_level}, relatorio #{routine.report.id}.</p>
            <p className="muted">A rotina carregou uma imagem validada, executou a leitura, salvou o resultado, gerou relatorio e registrou alerta simulado.</p>
          </section>
        )}
      </div>
    </section>
  );
}

createRoot(document.getElementById('root')).render(<App />);
