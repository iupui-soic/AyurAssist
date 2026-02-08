const API_BASE = 'https://sunbiz--ayurparam-service-fastapi-app.modal.run';

// Wake up GPU container while the user types (fire-and-forget)
fetch(API_BASE + '/warmup').catch(() => {});

const input = document.getElementById('symptomInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingEl = document.getElementById('loading');
const errorEl = document.getElementById('error');
const resultsEl = document.getElementById('results');

// Example buttons
document.getElementById('examples').addEventListener('click', (e) => {
  if (e.target.classList.contains('example-btn')) {
    input.value = e.target.dataset.value;
    analyzeBtn.disabled = false;
  }
});

// Enable/disable button based on input
input.addEventListener('input', () => {
  analyzeBtn.disabled = !input.value.trim();
});

// Enter key submits
input.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && input.value.trim()) {
    analyze();
  }
});

// Click submits
analyzeBtn.addEventListener('click', analyze);

async function analyze() {
  const text = input.value.trim();
  if (!text) return;

  // Set loading state
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';
  loadingEl.classList.remove('hidden');
  errorEl.classList.add('hidden');
  resultsEl.classList.add('hidden');

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);

  try {
    const response = await fetch(API_BASE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (!response.ok) throw new Error('HTTP ' + response.status);

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    const msg = err.name === 'AbortError'
      ? 'Request timeout. AI is processing, please try again.'
      : 'Cannot connect to backend. Check Modal deployment.';
    errorEl.textContent = '\u26A0\uFE0F ' + msg;
    errorEl.classList.remove('hidden');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze';
    loadingEl.classList.add('hidden');
  }
}

function esc(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function renderResults(data) {
  const treatment = data.results && data.results[0] && data.results[0].treatment_info;
  if (!treatment) {
    errorEl.textContent = '\u26A0\uFE0F No treatment information found for this symptom.';
    errorEl.classList.remove('hidden');
    return;
  }

  const resp = treatment.ayurparam_responses || {};
  let html = '';

  // Clinical Entities & Medical Codes
  html += '<div class="info-grid">';

  // Clinical Entities
  html += '<div class="clinical-entities-card"><h3>CLINICAL ENTITIES</h3><div class="entity-tags">';
  if (data.clinical_entities && data.clinical_entities.length) {
    data.clinical_entities.forEach((ent) => {
      html += '<span class="entity-tag">' + esc(ent.word) + '</span>';
    });
  }
  html += '</div></div>';

  // Medical Codes
  const umls = data.umls_cui || 'N/A';
  const snomed = data.snomed_code || (data.results[0] && data.results[0].snomed_code) || 'N/A';
  html += '<div class="medical-codes-card"><h3>MEDICAL CODES</h3>';
  html += '<div class="code-line"><strong>UMLS:</strong> ' + esc(umls) + '</div>';
  html += '<div class="code-line"><strong>SNOMED:</strong> ' + esc(snomed) + '</div>';
  html += '</div>';

  html += '</div>';

  // Condition Banner
  html += '<div class="condition-banner">';
  html += '<h2>' + esc(treatment.condition_name || '') + '</h2>';
  html += '<p>' + esc(treatment.sanskrit_name || '') + '</p>';
  html += '</div>';

  // Render each AyurParam response section
  var sections = [
    { key: 'overview_dosha_causes', title: '\uD83D\uDCCB Overview, Dosha & Causes' },
    { key: 'symptoms', title: '\uD83E\uDE7A Symptoms (Purvarupa & Rupa)' },
    { key: 'single_drug_remedies', title: '\uD83C\uDF3F Single Drug Remedies (Ottamooli)' },
    { key: 'classical_formulations', title: '\uD83D\uDCDC Classical Formulations' },
    { key: 'panchakarma_diet_lifestyle_yoga', title: '\uD83C\uDF7D\uFE0F Panchakarma, Diet, Lifestyle & Yoga' },
    { key: 'prognosis_modern_warnings', title: '\u26A0\uFE0F Prognosis & Warning Signs' },
  ];

  sections.forEach(function(sec) {
    var text = resp[sec.key];
    if (text) {
      html += '<div class="card"><h3>' + sec.title + '</h3>';
      html += '<div style="white-space:pre-line;font-size:0.9rem;line-height:1.7">' + esc(text) + '</div>';
      html += '</div>';
    }
  });

  // Disclaimer
  if (treatment.disclaimer) {
    html += '<div class="disclaimer">\u2695\uFE0F ' + esc(treatment.disclaimer) + '</div>';
  }

  resultsEl.innerHTML = html;
  resultsEl.classList.remove('hidden');
}