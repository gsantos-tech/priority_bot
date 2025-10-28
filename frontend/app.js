async function loadSummary() {
  try {
    // Cache-busting para evitar dados antigos no navegador
    const res = await fetch(`../reports/summary.json?t=${Date.now()}`);
    const data = await res.json();
    renderDataset(data.dataset_counts, data.data_path);
    renderThresholds(data.threshold_summaries);
    renderMatrizes(data.threshold_summaries);
    renderCharts(data.dataset_counts, data.threshold_summaries);
  } catch (e) {
    console.error('Falha ao carregar summary.json', e);
  }
}

function renderDataset(counts, dataPath) {
  const container = document.getElementById('datasetCards');
  const total = Object.values(counts).reduce((a,b) => a+b, 0);
  const items = [
    { label: 'Total de exemplos', value: total },
    { label: 'Classe 1', value: counts['1'] || 0 },
    { label: 'Classe 5', value: counts['5'] || 0 },
    { label: 'Classe 10', value: counts['10'] || 0 },
  ];
  const cards = items.map(it => `
    <div class="card">
      <h3>${it.label}</h3>
      <div class="value">${it.value}</div>
    </div>
  `).join('');
  const source = `
    <div class="card">
      <h3>Fonte de dados</h3>
      <div class="value" style="font-size:14px;color:#9ca3af">${dataPath || 'desconhecida'}</div>
    </div>
  `;
  container.innerHTML = source + cards;
}

function pct(x) { return (x*100).toFixed(1) + '%'; }

function renderThresholds(ths) {
  const container = document.getElementById('thresholds');
  container.innerHTML = ths.map(t => {
    const c10 = t.class_metrics['10'] || {};
    return `
      <div class="threshold">
        <div class="head">
          <h3>Limiar ${t.threshold.toFixed(2)}</h3>
          <span>Accuracy ${ (t.accuracy*100).toFixed(1) }%</span>
        </div>
        <div class="metrics">
          Macro F1: ${ (t.macro_f1||0).toFixed(2) }
          &nbsp;|&nbsp; Classe 10 – Precisão: ${ (c10.precision||0).toFixed(2) }, Recall: ${ (c10.recall||0).toFixed(2) }, F1: ${ (c10['f1-score']||0).toFixed(2) }
          <br>
          <a href="../reports/${t.pdf_path}" target="_blank">Abrir relatório PDF</a>
        </div>
      </div>
    `;
  }).join('');
}

function renderMatrizes(ths) {
  const container = document.getElementById('matrizGrid');
  container.innerHTML = ths.map(t => `
    <div class="matriz">
      <img src="../reports/${t.cm_path}" alt="Matriz de confusão th=${t.threshold}">
      <div style="margin-top:6px; color:#9ca3af; font-size:14px;">th=${t.threshold.toFixed(2)}</div>
    </div>
  `).join('');
}

function renderCharts(counts, ths) {
  // Distribuição do Dataset (barras)
  const dsCtx = document.getElementById('chartDataset').getContext('2d');
  const dsLabels = ['Classe 1', 'Classe 5', 'Classe 10'];
  const dsData = [counts['1']||0, counts['5']||0, counts['10']||0];
  new Chart(dsCtx, {
    type: 'bar',
    data: {
      labels: dsLabels,
      datasets: [{
        label: 'Exemplos',
        data: dsData,
        backgroundColor: ['#60a5fa','#f59e0b','#ef4444']
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } }
    }
  });

  // Macro F1 por limiar (barras)
  const f1Ctx = document.getElementById('chartMacroF1').getContext('2d');
  const thLabels = ths.map(t => t.threshold.toFixed(2));
  const f1Data = ths.map(t => (t.macro_f1||0));
  new Chart(f1Ctx, {
    type: 'bar',
    data: {
      labels: thLabels,
      datasets: [{
        label: 'Macro F1',
        data: f1Data,
        backgroundColor: '#22c55e'
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, max: 1 } }
    }
  });

  // Classe 10: Precisão vs Recall (linhas)
  const c10Ctx = document.getElementById('chartC10').getContext('2d');
  const c10Prec = ths.map(t => (t.class_metrics['10']?.precision||0));
  const c10Rec = ths.map(t => (t.class_metrics['10']?.recall||0));
  new Chart(c10Ctx, {
    type: 'line',
    data: {
      labels: thLabels,
      datasets: [
        { label: 'Precisão (10)', data: c10Prec, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.25)' },
        { label: 'Recall (10)', data: c10Rec, borderColor: '#f43f5e', backgroundColor: 'rgba(244,63,94,0.25)' }
      ]
    },
    options: {
      scales: { y: { beginAtZero: true, max: 1 } }
    }
  });
}

window.addEventListener('DOMContentLoaded', loadSummary);