async function loadSummary() {
  try {
    const res = await fetch('../reports/summary.json');
    const data = await res.json();
    renderDataset(data.dataset_counts);
    renderThresholds(data.threshold_summaries);
    renderMatrizes(data.threshold_summaries);
  } catch (e) {
    console.error('Falha ao carregar summary.json', e);
  }
}

function renderDataset(counts) {
  const container = document.getElementById('datasetCards');
  const total = Object.values(counts).reduce((a,b) => a+b, 0);
  const items = [
    { label: 'Total de exemplos', value: total },
    { label: 'Classe 1', value: counts['1'] || 0 },
    { label: 'Classe 5', value: counts['5'] || 0 },
    { label: 'Classe 10', value: counts['10'] || 0 },
  ];
  container.innerHTML = items.map(it => `
    <div class="card">
      <h3>${it.label}</h3>
      <div class="value">${it.value}</div>
    </div>
  `).join('');
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

window.addEventListener('DOMContentLoaded', loadSummary);