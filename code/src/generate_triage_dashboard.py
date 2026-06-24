"""
Explainability & Clinical Triage Dashboard
Generates a fully self-contained HTML report from evaluation JSON outputs.
No external dependencies required — works offline.

Usage:
    python generate_triage_dashboard.py \
        --results_dir job_outputs/results/ \
        --output job_outputs/triage_dashboard.html
"""

import json
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', type=str, default='job_outputs/results/')
    p.add_argument('--output',      type=str, default='job_outputs/triage_dashboard.html')
    p.add_argument('--gradcam_dir', type=str, default='job_outputs/gradcam/',
                   help='Directory containing gradcam_cropped.json and gradcam_uncropped.json')
    return p.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def classify(case, threshold=0.5):
    pred = case['fused_prob'] >= threshold
    real = case['true_label'] == 1
    if pred and real:     return 'TP'
    if not pred and not real: return 'TN'
    if pred and not real: return 'FP'
    return 'FN'


def build_html(conditions_data, master_table, significance, gradcam_data=None):
    cdata_js   = json.dumps(conditions_data)
    master_js  = json.dumps(master_table)
    sig_js     = json.dumps(significance)
    gradcam_js = json.dumps(gradcam_data or {})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MRNet Explainability & Clinical Triage Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
/* ── TOKENS ─────────────────────────────────────────────────────── */
:root {{
  --bg:        #f8fafc;
  --surface:   #ffffff;
  --card:      #ffffff;
  --border:    #e2e8f0;
  --border2:   #cbd5e1;
  --accent:    #2563eb;
  --accent2:   #7c3aed;
  --teal:      #0891b2;
  --green:     #059669;
  --amber:     #d97706;
  --red:       #dc2626;
  --text:      #0f172a;
  --text2:     #334155;
  --muted:     #64748b;
  --muted2:    #94a3b8;
  --sh:        0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.05);
  --sh-md:     0 4px 16px rgba(0,0,0,.08), 0 1px 4px rgba(0,0,0,.04);
  --sh-lg:     0 8px 32px rgba(0,0,0,.10), 0 2px 8px rgba(0,0,0,.05);
  --radius:    12px;
}}

/* ── RESET ──────────────────────────────────────────────────────── */
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:var(--bg);color:var(--text);line-height:1.5;
  -webkit-font-smoothing:antialiased;
}}

/* ── LAYOUT ─────────────────────────────────────────────────────── */
.wrapper{{max-width:1440px;margin:0 auto;padding:28px 24px 60px}}

/* ── HEADER ─────────────────────────────────────────────────────── */
.header{{
  display:flex;align-items:flex-start;justify-content:space-between;
  gap:16px;flex-wrap:wrap;
  padding-bottom:20px;margin-bottom:24px;
  border-bottom:1px solid var(--border);
}}
.header-logo{{
  display:flex;align-items:center;gap:12px;
}}
.logo-mark{{
  width:40px;height:40px;border-radius:10px;
  background:linear-gradient(135deg,#2563eb,#7c3aed);
  display:flex;align-items:center;justify-content:center;
  font-size:18px;flex-shrink:0;
}}
.header-title h1{{
  font-size:1.25rem;font-weight:800;color:var(--text);letter-spacing:-0.3px;
}}
.header-title p{{font-size:.78rem;color:var(--muted);margin-top:2px}}
.header-right{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
.status-pill{{
  display:inline-flex;align-items:center;gap:6px;
  padding:5px 12px;border-radius:999px;font-size:.72rem;font-weight:600;
  background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;
  letter-spacing:.3px;
}}
.status-dot{{width:7px;height:7px;border-radius:50%;background:#22c55e;
  animation:blink 2s infinite}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.ucd-badge{{
  padding:5px 12px;border-radius:999px;font-size:.72rem;font-weight:600;
  background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;
}}

/* ── TABS ────────────────────────────────────────────────────────── */
.tab-bar{{
  display:flex;gap:2px;margin-bottom:24px;
  border-bottom:2px solid var(--border);
}}
.tab{{
  padding:10px 20px;border:none;border-radius:0;font-family:inherit;
  font-size:.82rem;font-weight:600;cursor:pointer;
  background:transparent;color:var(--muted);
  border-bottom:2px solid transparent;margin-bottom:-2px;
  transition:all .18s;letter-spacing:.1px;
}}
.tab:hover:not(.active){{color:var(--text2);background:#f1f5f9}}
.tab.active{{color:var(--accent);border-bottom-color:var(--accent)}}

/* ── SECTIONS ────────────────────────────────────────────────────── */
.section{{display:none}}.section.active{{display:block}}

/* ── KPI GRID ────────────────────────────────────────────────────── */
.kpi-grid{{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));
  gap:14px;margin-bottom:22px;
}}
.kpi{{
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:18px 20px;
  box-shadow:var(--sh);transition:box-shadow .2s,transform .2s;
  position:relative;overflow:hidden;
}}
.kpi:hover{{box-shadow:var(--sh-md);transform:translateY(-2px)}}
.kpi-accent{{
  position:absolute;top:0;left:0;right:0;height:3px;
  border-radius:var(--radius) var(--radius) 0 0;
}}
.kpi-label{{
  font-size:.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.8px;color:var(--muted);margin-bottom:8px;
}}
.kpi-value{{font-size:1.8rem;font-weight:800;letter-spacing:-1px;color:var(--text)}}
.kpi-sub{{font-size:.72rem;color:var(--muted2);margin-top:5px}}

/* ── CARDS ───────────────────────────────────────────────────────── */
.card{{
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:22px 24px;
  box-shadow:var(--sh);margin-bottom:18px;
}}
.card-title{{
  font-size:.72rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.8px;color:var(--muted);margin-bottom:16px;
  display:flex;align-items:center;gap:8px;
}}

/* ── CHART GRID ──────────────────────────────────────────────────── */
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px}}
@media(max-width:780px){{.chart-grid{{grid-template-columns:1fr}}}}

/* ── SVG CHARTS ──────────────────────────────────────────────────── */
.svg-chart{{width:100%;overflow:visible}}
.chart-axis-label{{font-size:10px;fill:#94a3b8;font-family:inherit}}
.chart-tick{{font-size:10px;fill:#64748b;font-family:inherit}}
.chart-bar{{rx:4;transition:opacity .15s}}
.chart-bar:hover{{opacity:.85;cursor:default}}

/* ── MASTER TABLE ────────────────────────────────────────────────── */
.table-wrap{{overflow-x:auto;border-radius:10px;border:1px solid var(--border)}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
thead th{{
  background:#f8fafc;padding:11px 16px;text-align:left;
  font-size:.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.7px;color:var(--muted);border-bottom:1px solid var(--border);
  white-space:nowrap;
}}
tbody tr{{border-bottom:1px solid #f1f5f9;transition:background .12s}}
tbody tr:last-child{{border-bottom:none}}
tbody tr:hover{{background:#f8fafc}}
tbody td{{padding:10px 16px;white-space:nowrap;color:var(--text2)}}
.pill{{
  display:inline-block;padding:2px 10px;border-radius:999px;
  font-size:.72rem;font-weight:700;
}}

/* ── CONDITION TABS ──────────────────────────────────────────────── */
.cond-tabs{{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap}}
.cond-tab{{
  padding:7px 16px;border-radius:999px;
  border:1.5px solid var(--border);
  font-family:inherit;font-size:.78rem;font-weight:600;cursor:pointer;
  background:var(--surface);color:var(--muted2);transition:all .18s;
}}
.cond-tab.active{{border-color:var(--accent);color:var(--accent);background:#eff6ff}}
.cond-tab:hover:not(.active){{border-color:var(--border2);color:var(--text2)}}

/* ── TRIAGE LIST ─────────────────────────────────────────────────── */
.triage-headers{{
  display:grid;grid-template-columns:80px 1fr 210px 90px;
  gap:12px;padding:0 12px 8px;
  font-size:.66rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.7px;color:var(--muted2);
}}
.case-row{{
  display:grid;grid-template-columns:80px 1fr 210px 90px;
  gap:12px;align-items:center;
  padding:9px 12px;border-radius:8px;margin-bottom:4px;
  border:1px solid transparent;
  transition:background .12s,border-color .12s;cursor:default;
}}
.case-row:hover{{background:#f8fafc;border-color:var(--border)}}
.pid{{font-weight:700;font-size:.8rem;color:var(--text2)}}
.bar-track{{background:#f1f5f9;border-radius:999px;height:7px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:999px;transition:width .3s ease}}
.prob-label{{font-size:.7rem;color:var(--muted);margin-top:3px}}
.plane-mini{{display:flex;gap:4px;align-items:center}}
.plane-seg{{width:28px;height:7px;border-radius:4px}}
.outcome-pill{{
  display:inline-block;padding:3px 10px;border-radius:999px;
  font-size:.7rem;font-weight:700;text-align:center;
}}
.filter-bar{{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}}
.filter-btn{{
  padding:5px 14px;border-radius:999px;border:1.5px solid var(--border);
  font-family:inherit;font-size:.73rem;font-weight:600;cursor:pointer;
  background:white;color:var(--muted2);transition:all .18s;
}}
.filter-btn.active{{border-color:var(--accent);color:var(--accent);background:#eff6ff}}

/* ── COMPARISON BARS ─────────────────────────────────────────────── */
.comp-item{{margin-bottom:18px}}
.comp-label{{display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:6px}}
.two-bars{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.bar-group-label{{font-size:.66rem;color:var(--muted2);margin-bottom:3px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}}
.h-bar-track{{background:#f1f5f9;border-radius:999px;height:9px;overflow:hidden}}
.h-bar-fill{{height:100%;border-radius:999px;transition:width .4s ease}}

/* ── SIG CARDS ───────────────────────────────────────────────────── */
.sig-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}}
.sig-card{{
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:18px;box-shadow:var(--sh);
}}
.sig-card h4{{
  font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;
  color:var(--muted);margin-bottom:12px;
}}
.sig-row{{
  display:flex;justify-content:space-between;align-items:center;
  padding:5px 0;border-bottom:1px solid #f1f5f9;font-size:.8rem;
  color:var(--text2);
}}
.sig-row:last-child{{border-bottom:none}}
.sig-val{{font-weight:700}}

/* ── SCROLLBAR ───────────────────────────────────────────────────── */
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:#f1f5f9}}
::-webkit-scrollbar-thumb{{background:#cbd5e1;border-radius:99px}}

/* ── GRAD-CAM ────────────────────────────────────────────────────── */
.gc-filter-label{{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.7px;color:var(--muted2);margin-bottom:6px}}
.gc-card{{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px;box-shadow:var(--sh);margin-bottom:16px}}
.gc-card-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;flex-wrap:wrap;gap:8px}}
.gc-card-meta{{font-size:.8rem;color:var(--text2);line-height:1.5}}
.gc-card-meta b{{color:var(--text)}}
.gc-mode-badge{{padding:3px 10px;border-radius:999px;font-size:.7rem;font-weight:700}}
.gc-img{{width:100%;border-radius:8px;border:1px solid var(--border);display:block}}
.gc-grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}
@media(max-width:900px){{.gc-grid-3{{grid-template-columns:1fr}}}}

/* ── FOOTER ──────────────────────────────────────────────────────── */
.footer{{
  margin-top:40px;padding-top:16px;border-top:1px solid var(--border);
  font-size:.72rem;color:var(--muted2);
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;
}}
</style>
</head>
<body>
<div class="wrapper">

<!-- HEADER -->
<div class="header">
  <div class="header-logo">
    <div class="logo-mark">🏥</div>
    <div class="header-title">
      <h1>MRNet Explainability &amp; Clinical Triage Dashboard</h1>
      <p>Knee MRI · AlexNet-ImageNet Baseline · Semantic R-CNN Cropping · University College Dublin</p>
    </div>
  </div>
  <div class="header-right">
    <span class="status-pill"><span class="status-dot"></span>Analysis Complete</span>
    <span class="ucd-badge">UCD 2025</span>
  </div>
</div>

<!-- TABS -->
<div class="tab-bar">
  <button class="tab active" onclick="showTab('overview')">Overview</button>
  <button class="tab" onclick="showTab('triage')">Clinical Triage</button>
  <button class="tab" onclick="showTab('performance')">Performance Analysis</button>
  <button class="tab" onclick="showTab('significance')">Statistical Tests</button>
  <button class="tab" onclick="showTab('gradcam')">🔬 Grad-CAM</button>
</div>

<!-- ═══════ OVERVIEW ═══════ -->
<div id="tab-overview" class="section active">
  <div class="kpi-grid" id="kpiGrid"></div>
  <div class="chart-grid">
    <div class="card">
      <div class="card-title">Fused AUC — Cropped vs Uncropped</div>
      <div id="aucBarChart"></div>
    </div>
    <div class="card">
      <div class="card-title">Per-Condition Metrics (Cropped)</div>
      <div id="metricBarChart"></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Master Results Table</div>
    <div class="table-wrap" id="masterTable"></div>
  </div>
</div>

<!-- ═══════ TRIAGE ═══════ -->
<div id="tab-triage" class="section">
  <div class="cond-tabs" id="condTabsTriage"></div>
  <div class="card">
    <div class="card-title" style="justify-content:space-between;flex-wrap:wrap;gap:8px">
      <span>Confidence-Ranked Patient Queue</span>
      <span id="triageStats" style="font-size:.72rem;font-weight:400;text-transform:none;letter-spacing:0;color:var(--muted)"></span>
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px;font-size:.73rem;color:var(--muted2)">
      <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#059669;vertical-align:middle;margin-right:4px"></span>True Positive</span>
      <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#0891b2;vertical-align:middle;margin-right:4px"></span>True Negative</span>
      <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#dc2626;vertical-align:middle;margin-right:4px"></span>False Positive</span>
      <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#d97706;vertical-align:middle;margin-right:4px"></span>False Negative ⚠️ (missed)</span>
    </div>
    <div class="triage-headers">
      <span>Patient</span><span>Fused Confidence</span><span>Axial · Coronal · Sagittal</span><span>Outcome</span>
    </div>
    <div id="triageList" style="max-height:560px;overflow-y:auto"></div>
    <div class="filter-bar">
      <button class="filter-btn active" id="fAll"  onclick="filterTriage('all')">All Cases</button>
      <button class="filter-btn" id="fTP"   onclick="filterTriage('TP')">True Positives</button>
      <button class="filter-btn" id="fTN"   onclick="filterTriage('TN')">True Negatives</button>
      <button class="filter-btn" id="fFP"   onclick="filterTriage('FP')">False Positives</button>
      <button class="filter-btn" id="fFN"   onclick="filterTriage('FN')">False Negatives ⚠️</button>
    </div>
  </div>
</div>

<!-- ═══════ PERFORMANCE ═══════ -->
<div id="tab-performance" class="section">
  <div class="cond-tabs" id="condTabsPerf"></div>
  <div id="perfContent"></div>
</div>

<!-- ═══════ STATISTICS ═══════ -->
<div id="tab-significance" class="section">
  <div class="card" style="margin-bottom:18px">
    <p style="font-size:.82rem;color:var(--muted);line-height:1.7">
      Paired bootstrap AUC-difference tests (1,000 bootstrap iterations, two-sided, seed = 42).
      A confidence interval that excludes 0 indicates statistical significance at α = 0.05.
    </p>
  </div>
  <div class="sig-grid" id="sigGrid"></div>
</div>

<!-- ═══════ GRAD-CAM ═══════ -->
<div id="tab-gradcam" class="section">
  <div class="card" style="margin-bottom:16px">
    <p style="font-size:.82rem;color:var(--muted);line-height:1.8">
      <strong>Grad-CAM</strong> (Gradient-weighted Class Activation Mapping) highlights which regions
      of the MRI scan the AlexNet model focuses on when making a prediction.
    </p>
  </div>
  <div style="display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start;margin-bottom:20px">
    <div>
      <div class="gc-filter-label">Condition</div>
      <div class="cond-tabs" id="gcCondTabs" style="margin-bottom:0"></div>
    </div>
    <div>
      <div class="gc-filter-label">View Mode</div>
      <div class="cond-tabs" id="gcModeTabs" style="margin-bottom:0"></div>
    </div>
  </div>
  <div id="gcStats" style="font-size:.78rem;color:var(--muted);margin-bottom:16px;font-weight:500"></div>
  <div id="gcGrid"></div>
</div>

<div class="footer">
  <span>MRNet Explainability &amp; Clinical Triage Dashboard · University College Dublin</span>
  <span id="footerDate"></span>
</div>
</div>

<script>
const CONDITIONS_DATA = {cdata_js};
const MASTER_TABLE    = {master_js};
const SIGNIFICANCE    = {sig_js};
const GRADCAM_DATA    = {gradcam_js};

const COND_LABELS = {{acl:'ACL Tear', meniscus:'Meniscus Tear', abnormal:'Abnormality'}};
const COND_ICONS  = {{acl:'🦵', meniscus:'🔵', abnormal:'⚠️'}};
const PLANES      = ['axial','coronal','sagittal'];
const PLANE_SHORT = {{axial:'Ax',coronal:'Co',sagittal:'Sa'}};

function pct(v)   {{ return (v*100).toFixed(1)+'%'; }}
function fmt2(v)  {{ return (v*100).toFixed(2)+'%'; }}
function fmtDiff(v) {{
  const s = (v*100).toFixed(2);
  return (v>0?'+':'')+s+'%';
}}

function classify(c, thr) {{
  thr = thr || 0.5;
  const pred = c.fused_prob >= thr, real = c.true_label === 1;
  if(pred && real)   return 'TP';
  if(!pred && !real) return 'TN';
  if(pred && !real)  return 'FP';
  return 'FN';
}}

function outcomeStyle(o) {{
  if(o==='TP') return 'background:#f0fdf4;color:#15803d;border:1px solid #86efac';
  if(o==='TN') return 'background:#ecfeff;color:#0e7490;border:1px solid #67e8f9';
  if(o==='FP') return 'background:#fef2f2;color:#dc2626;border:1px solid #fca5a5';
  return 'background:#fffbeb;color:#d97706;border:1px solid #fcd34d';
}}

function barColor(prob) {{
  if(prob >= 0.8) return '#059669';
  if(prob >= 0.5) return '#2563eb';
  if(prob >= 0.3) return '#d97706';
  return '#dc2626';
}}

function aucBadge(v) {{
  if(v>=.9)  return {{bg:'#f0fdf4',text:'#15803d'}};
  if(v>=.8)  return {{bg:'#eff6ff',text:'#1d4ed8'}};
  if(v>=.7)  return {{bg:'#faf5ff',text:'#7c3aed'}};
  return {{bg:'#fffbeb',text:'#d97706'}};
}}

/* ── TAB ROUTING ──────────────────────────────────────────────────── */
function showTab(id) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  const idx = {{overview:0,triage:1,performance:2,significance:3,gradcam:4}}[id];
  document.querySelectorAll('.tab')[idx].classList.add('active');
  if(id === 'triage')      initTriage();
  if(id === 'performance') initPerf();
  if(id === 'significance') initSig();
  if(id === 'gradcam')     initGradCam();
}}

/* ── SVG CHARTS ───────────────────────────────────────────────────── */
function svgBar(items, opts) {{
  const W = opts.width || 520;
  const H = opts.height || 200;
  const PAD = {{top:10, right:16, bottom:36, left:44}};
  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top  - PAD.bottom;
  const minY = opts.minY || 0;
  const maxY = opts.maxY || 100;

  const n = items.length;
  const grpW = cW / n;
  const nSets = items[0].values.length;
  const barW = Math.min(28, (grpW * 0.7) / nSets);
  const gap  = barW * 0.25;

  const toY = v => PAD.top + cH - ((v - minY) / (maxY - minY)) * cH;

  const colors = ['#2563eb','#7c3aed','#059669','#0891b2'];

  let s = `<svg class="svg-chart" viewBox="0 0 ${{W}} ${{H}}" style="max-height:${{H}}px">`;

  const nTicks = 5;
  for(let i=0;i<=nTicks;i++) {{
    const v = minY + (maxY - minY) * i / nTicks;
    const y = toY(v);
    s += `<line x1="${{PAD.left}}" y1="${{y}}" x2="${{PAD.left+cW}}" y2="${{y}}"
            stroke="#f1f5f9" stroke-width="1"/>`;
    s += `<text x="${{PAD.left-6}}" y="${{y+4}}" text-anchor="end" class="chart-tick">${{v.toFixed(0)}}%</text>`;
  }}

  items.forEach((grp, gi) => {{
    const cx = PAD.left + grpW * gi + grpW/2;
    const totalW = nSets * barW + (nSets-1) * gap;
    grp.values.forEach((v, si) => {{
      const x = cx - totalW/2 + si*(barW+gap);
      const y = toY(v);
      const h = toY(minY) - y;
      s += `<rect x="${{x.toFixed(1)}}" y="${{y.toFixed(1)}}"
              width="${{barW}}" height="${{Math.max(2,h).toFixed(1)}}"
              fill="${{colors[si]}}" rx="3" opacity="0.9"/>`;
      s += `<text x="${{(x+barW/2).toFixed(1)}}" y="${{(y-4).toFixed(1)}}"
              text-anchor="middle" class="chart-tick" style="font-size:9px">${{v.toFixed(1)}}</text>`;
    }});
    s += `<text x="${{cx.toFixed(1)}}" y="${{(PAD.top+cH+16).toFixed(1)}}"
            text-anchor="middle" class="chart-tick">${{grp.label}}</text>`;
  }});

  s += `<line x1="${{PAD.left}}" y1="${{PAD.top}}" x2="${{PAD.left}}" y2="${{PAD.top+cH}}"
          stroke="#cbd5e1" stroke-width="1"/>`;
  s += `<line x1="${{PAD.left}}" y1="${{PAD.top+cH}}" x2="${{PAD.left+cW}}" y2="${{PAD.top+cH}}"
          stroke="#cbd5e1" stroke-width="1"/>`;

  if(opts.legend) {{
    opts.legend.forEach((l,i) => {{
      const lx = PAD.left + i * 110;
      const ly = H - 4;
      s += `<rect x="${{lx}}" y="${{ly-8}}" width="10" height="10" fill="${{colors[i]}}" rx="2"/>`;
      s += `<text x="${{lx+14}}" y="${{ly+1}}" class="chart-tick">${{l}}</text>`;
    }});
  }}

  s += '</svg>';
  return s;
}}

/* ── OVERVIEW ─────────────────────────────────────────────────────── */
(function buildOverview() {{
  const conds = ['acl','meniscus','abnormal'];
  const cropped   = MASTER_TABLE.filter(r => r.data_mode === 'cropped');
  const uncropped = MASTER_TABLE.filter(r => r.data_mode === 'uncropped');

  const avgC = cropped.reduce((s,r)=>s+r.auc,0) / (cropped.length||1);
  const avgU = uncropped.reduce((s,r)=>s+r.auc,0) / (uncropped.length||1);
  const best = [...MASTER_TABLE].sort((a,b)=>b.auc-a.auc)[0] || {{}};
  const totalPats = Object.values(CONDITIONS_DATA).reduce((s,c)=>s+(c.cases||[]).length,0);

  const kpis = [
    {{label:'Avg AUC (Cropped)',   val:fmt2(avgC),   sub:'across all conditions',      accent:'#2563eb'}},
    {{label:'Avg AUC (Uncropped)', val:fmt2(avgU),   sub:'baseline performance',        accent:'#7c3aed'}},
    {{label:'Cropping Gain',       val:fmtDiff(avgC-avgU), sub:'mean AUC improvement',   accent:'#059669'}},
    {{label:'Best AUC',            val:fmt2(best.auc||0), sub:(best.condition||'')+'  '+(best.data_mode||''), accent:'#059669'}},
    {{label:'Test Patients',       val:totalPats,    sub:'across 3 conditions',          accent:'#0891b2'}},
    {{label:'Conditions',          val:'3',          sub:'ACL · Meniscus · Abnormal',    accent:'#2563eb'}},
  ];
  const g = document.getElementById('kpiGrid');
  kpis.forEach(k => {{
    g.insertAdjacentHTML('beforeend', `
    <div class="kpi">
      <div class="kpi-accent" style="background:${{k.accent}}"></div>
      <div class="kpi-label">${{k.label}}</div>
      <div class="kpi-value" style="color:${{k.accent}}">${{k.val}}</div>
      <div class="kpi-sub">${{k.sub}}</div>
    </div>`);
  }});

  // AUC grouped bar chart
  const aucItems = conds.map(c => {{
    const cu = MASTER_TABLE.find(r=>r.condition===c&&r.data_mode==='uncropped');
    const cc = MASTER_TABLE.find(r=>r.condition===c&&r.data_mode==='cropped');
    return {{
      label: COND_LABELS[c].replace(' Tear',''),
      values: [cc?+(cc.auc*100).toFixed(2):0, cu?+(cu.auc*100).toFixed(2):0]
    }};
  }});
  document.getElementById('aucBarChart').innerHTML =
    svgBar(aucItems, {{width:480, height:200, minY:65, maxY:100, legend:['Cropped','Uncropped']}});

  // Metric chart (cropped only)
  const metItems = ['sensitivity','specificity','f1'].map(m => {{
    return {{
      label: {{sensitivity:'Sensitivity',specificity:'Specificity',f1:'F1 Score'}}[m],
      values: conds.map(c => {{
        const r = MASTER_TABLE.find(x=>x.condition===c&&x.data_mode==='cropped');
        return r ? +(r[m]*100).toFixed(1) : 0;
      }})
    }};
  }});
  document.getElementById('metricBarChart').innerHTML =
    svgBar(metItems, {{width:480, height:200, minY:0, maxY:105,
      legend: conds.map(c=>COND_LABELS[c].replace(' Tear',''))
    }});

  // Table
  let html = `<table><thead><tr>
    <th>Condition</th><th>Architecture</th><th>Data Mode</th>
    <th>AUC (95% CI)</th><th>Sensitivity</th><th>Specificity</th><th>F1</th>
  </tr></thead><tbody>`;
  MASTER_TABLE.forEach(r => {{
    const ac = aucBadge(r.auc);
    const peer = MASTER_TABLE.find(x=>x.condition===r.condition&&x.architecture===r.architecture&&x.data_mode!==r.data_mode);
    const delta = peer ? (r.auc - peer.auc) * 100 : null;
    const dTag = delta !== null
      ? `<span style="font-size:.7rem;color:${{delta>0?'#15803d':'#dc2626'}};margin-left:4px">${{delta>0?'▲':'▼'}}${{Math.abs(delta).toFixed(1)}}%</span>`
      : '';
    html += `<tr>
      <td style="font-weight:600;color:var(--text)">${{COND_ICONS[r.condition]}} ${{COND_LABELS[r.condition]}}</td>
      <td style="color:var(--muted)">${{r.architecture}}</td>
      <td><span class="pill" style="background:${{r.data_mode==='cropped'?'#f0fdf4':'#faf5ff'}};color:${{r.data_mode==='cropped'?'#15803d':'#7c3aed'}}">${{r.data_mode}}</span></td>
      <td>
        <span class="pill" style="background:${{ac.bg}};color:${{ac.text}}">${{(r.auc*100).toFixed(2)}}%</span>
        <span style="font-size:.72rem;color:var(--muted2);margin-left:4px">[${{(r.auc_ci_low*100).toFixed(1)}}–${{(r.auc_ci_high*100).toFixed(1)}}%]</span>
        ${{dTag}}
      </td>
      <td>${{(r.sensitivity*100).toFixed(1)}}%</td>
      <td>${{(r.specificity*100).toFixed(1)}}%</td>
      <td>${{(r.f1*100).toFixed(1)}}%</td>
    </tr>`;
  }});
  html += '</tbody></table>';
  document.getElementById('masterTable').innerHTML = html;
}})();

/* ── TRIAGE ───────────────────────────────────────────────────────── */
let triageInited = false;
let curCond = 'acl', curFilter = 'all';

function initTriage() {{
  if(triageInited) {{ renderTriage(); return; }}
  triageInited = true;
  const tabs = document.getElementById('condTabsTriage');
  Object.keys(CONDITIONS_DATA).forEach(c => {{
    const btn = document.createElement('button');
    btn.className = 'cond-tab' + (c==='acl'?' active':'');
    btn.textContent = COND_ICONS[c] + ' ' + COND_LABELS[c];
    btn.onclick = () => {{
      document.querySelectorAll('#condTabsTriage .cond-tab').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      curCond = c; curFilter = 'all';
      document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
      document.getElementById('fAll').classList.add('active');
      renderTriage();
    }};
    tabs.appendChild(btn);
  }});
  renderTriage();
}}

function filterTriage(f) {{
  curFilter = f;
  ['All','TP','TN','FP','FN'].forEach(x => {{
    const b = document.getElementById('f'+x);
    if(b) b.classList.toggle('active', (f==='all'&&x==='All') || x===f);
  }});
  renderTriage();
}}

function renderTriage() {{
  const cd = CONDITIONS_DATA[curCond];
  if(!cd || !cd.cases) {{
    document.getElementById('triageList').innerHTML =
      '<p style="padding:20px;color:var(--muted);text-align:center">No data available.</p>';
    return;
  }}
  const thr = cd.threshold || 0.5;
  const classified = cd.cases.map(c => ({{...c, outcome:classify(c,thr)}}));
  const tp=classified.filter(c=>c.outcome==='TP').length;
  const tn=classified.filter(c=>c.outcome==='TN').length;
  const fp=classified.filter(c=>c.outcome==='FP').length;
  const fn=classified.filter(c=>c.outcome==='FN').length;
  document.getElementById('triageStats').textContent =
    classified.length+' patients · TP: '+tp+' · TN: '+tn+' · FP: '+fp+' · FN: '+fn;

  const shown = curFilter==='all' ? classified : classified.filter(c=>c.outcome===curFilter);
  const list = document.getElementById('triageList');
  if(!shown.length) {{
    list.innerHTML='<p style="padding:20px;color:var(--muted);text-align:center">No cases match this filter.</p>';
    return;
  }}

  list.innerHTML = shown.map(c => {{
    const prob = c.fused_prob;
    const col  = barColor(prob);
    const planes = PLANES.map(p => {{
      const v = (c.plane_probs||{{}})[p];
      if(v==null) return `<div class="plane-seg" style="background:#f1f5f9" title="N/A"></div>`;
      const pc = v>=.8?'#059669':v>=.5?'#2563eb':'#dc2626';
      return `<div class="plane-seg" style="background:${{pc}};opacity:${{(0.25+v*0.75).toFixed(2)}}" title="${{PLANE_SHORT[p]}}: ${{pct(v)}}"></div>`;
    }}).join('');

    return `<div class="case-row">
      <span class="pid">#${{c.patient_id}}</span>
      <div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${{(prob*100).toFixed(1)}}%;background:${{col}}"></div>
        </div>
        <div class="prob-label">${{pct(prob)}} · Ground truth: ${{c.true_label===1?'Positive':'Negative'}}</div>
      </div>
      <div class="plane-mini">${{planes}}</div>
      <span class="outcome-pill" style="${{outcomeStyle(c.outcome)}}">${{c.outcome}}</span>
    </div>`;
  }}).join('');
}}

/* ── PERFORMANCE ──────────────────────────────────────────────────── */
let perfInited = false;
let curPerfCond = 'acl';

function initPerf() {{
  if(!perfInited) {{
    perfInited = true;
    const tabs = document.getElementById('condTabsPerf');
    Object.keys(CONDITIONS_DATA).forEach(c => {{
      const btn = document.createElement('button');
      btn.className = 'cond-tab' + (c==='acl'?' active':'');
      btn.textContent = COND_ICONS[c] + ' ' + COND_LABELS[c];
      btn.onclick = () => {{
        document.querySelectorAll('#condTabsPerf .cond-tab').forEach(b=>b.classList.remove('active'));
        btn.classList.add('active');
        curPerfCond = c;
        renderPerf();
      }};
      tabs.appendChild(btn);
    }});
  }}
  renderPerf();
}}

function renderPerf() {{
  const c  = curPerfCond;
  const cd = CONDITIONS_DATA[c];
  const rows = MASTER_TABLE.filter(r => r.condition === c);
  const el = document.getElementById('perfContent');
  if(!rows.length || !cd) {{
    el.innerHTML='<div class="card"><p style="color:var(--muted)">No data available.</p></div>';
    return;
  }}

  const thr = (cd && cd.threshold) || 0.5;
  const cases = (cd && cd.cases) || [];

  // Per-plane confusion
  const planeCounts = {{}};
  PLANES.forEach(p => {{
    const pp = cases.filter(x=>x.plane_probs&&x.plane_probs[p]!=null);
    const tp=pp.filter(x=>x.plane_probs[p]>=thr&&x.true_label===1).length;
    const tn=pp.filter(x=>x.plane_probs[p]<thr&&x.true_label===0).length;
    const fp=pp.filter(x=>x.plane_probs[p]>=thr&&x.true_label===0).length;
    const fn=pp.filter(x=>x.plane_probs[p]<thr&&x.true_label===1).length;
    planeCounts[p] = {{tp,tn,fp,fn,n:pp.length}};
  }});

  // Confidence histogram data
  const posProbs = cases.filter(x=>x.true_label===1).map(x=>x.fused_prob);
  const negProbs = cases.filter(x=>x.true_label===0).map(x=>x.fused_prob);
  const binEdges = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0];
  const histPos  = binEdges.slice(0,-1).map(b=>posProbs.filter(p=>p>=b&&p<b+.1).length);
  const histNeg  = binEdges.slice(0,-1).map(b=>negProbs.filter(p=>p>=b&&p<b+.1).length);

  // Metric bars
  const metrics = ['auc','sensitivity','specificity','f1'];
  const mLabels = {{auc:'AUC',sensitivity:'Sensitivity',specificity:'Specificity',f1:'F1 Score'}};
  const mColors = {{auc:'#2563eb',sensitivity:'#059669',specificity:'#7c3aed',f1:'#0891b2'}};

  let metBarsHtml = metrics.map(m => {{
    const cu = rows.find(r=>r.data_mode==='uncropped');
    const cc = rows.find(r=>r.data_mode==='cropped');
    if(!cu||!cc) return '';
    const delta = cc[m] - cu[m];
    return `<div class="comp-item">
      <div class="comp-label">
        <span style="font-weight:600;color:${{mColors[m]}}">${{mLabels[m]}}</span>
        <span style="color:var(--muted2)">
          Uncropped: <b style="color:var(--text2)">${{pct(cu[m])}}</b>
          &nbsp;→&nbsp;
          Cropped: <b style="color:${{delta>=0?'#15803d':'#dc2626'}}">${{pct(cc[m])}}</b>
          &nbsp;<span style="color:${{delta>=0?'#15803d':'#dc2626'}}">${{delta>=0?'▲':'▼'}}${{Math.abs(delta*100).toFixed(1)}}%</span>
        </span>
      </div>
      <div class="two-bars">
        <div>
          <div class="bar-group-label">Uncropped</div>
          <div class="h-bar-track">
            <div class="h-bar-fill" style="width:${{(cu[m]*100).toFixed(1)}}%;background:#7c3aed88"></div>
          </div>
        </div>
        <div>
          <div class="bar-group-label">Cropped</div>
          <div class="h-bar-track">
            <div class="h-bar-fill" style="width:${{(cc[m]*100).toFixed(1)}}%;background:${{mColors[m]}}"></div>
          </div>
        </div>
      </div>
    </div>`;
  }}).join('');

  // Per-plane SVG chart
  const planeItems = PLANES.map(p => ({{
    label: p.charAt(0).toUpperCase()+p.slice(1),
    values: [planeCounts[p].tp, planeCounts[p].tn, planeCounts[p].fp, planeCounts[p].fn]
  }}));
  const planeChartHtml = svgBar(planeItems, {{
    width:480, height:190, minY:0, maxY:Math.max(20,...PLANES.flatMap(p=>[planeCounts[p].tp,planeCounts[p].tn,planeCounts[p].fp,planeCounts[p].fn]))+5,
    legend:['TP','TN','FP','FN']
  }});

  // Histogram SVG — simple grouped bars
  const maxBin = Math.max(1, ...histPos, ...histNeg);
  const histH = 180, histW = 500, hPad = {{t:10,r:16,b:36,l:36}};
  const hcW = histW - hPad.l - hPad.r;
  const hcH = histH - hPad.t  - hPad.b;
  const binW = hcW / 10;
  const bw = binW * 0.35;
  let histSvg = `<svg class="svg-chart" viewBox="0 0 ${{histW}} ${{histH}}" style="max-height:${{histH}}px">`;
  for(let i=0;i<=5;i++) {{
    const v = maxBin * i/5;
    const y = hPad.t + hcH - (v/maxBin)*hcH;
    histSvg += `<line x1="${{hPad.l}}" y1="${{y.toFixed(1)}}" x2="${{hPad.l+hcW}}" y2="${{y.toFixed(1)}}" stroke="#f1f5f9" stroke-width="1"/>`;
    histSvg += `<text x="${{hPad.l-4}}" y="${{(y+4).toFixed(1)}}" text-anchor="end" class="chart-tick">${{v.toFixed(0)}}</text>`;
  }}
  histPos.forEach((v,i) => {{
    const x = hPad.l + i*binW + binW*0.15;
    const h = Math.max(0,(v/maxBin)*hcH);
    histSvg += `<rect x="${{x.toFixed(1)}}" y="${{(hPad.t+hcH-h).toFixed(1)}}" width="${{bw.toFixed(1)}}" height="${{h.toFixed(1)}}" fill="#059669" rx="3" opacity="0.8"/>`;
  }});
  histNeg.forEach((v,i) => {{
    const x = hPad.l + i*binW + binW*0.15 + bw + 2;
    const h = Math.max(0,(v/maxBin)*hcH);
    histSvg += `<rect x="${{x.toFixed(1)}}" y="${{(hPad.t+hcH-h).toFixed(1)}}" width="${{bw.toFixed(1)}}" height="${{h.toFixed(1)}}" fill="#0891b2" rx="3" opacity="0.8"/>`;
  }});
  binEdges.slice(0,-1).forEach((b,i) => {{
    const x = hPad.l + i*binW + binW/2;
    histSvg += `<text x="${{x.toFixed(1)}}" y="${{(hPad.t+hcH+14).toFixed(1)}}" text-anchor="middle" class="chart-tick" style="font-size:9px">${{(b*100).toFixed(0)}}%</text>`;
  }});
  histSvg += `<line x1="${{hPad.l}}" y1="${{hPad.t}}" x2="${{hPad.l}}" y2="${{hPad.t+hcH}}" stroke="#cbd5e1" stroke-width="1"/>`;
  histSvg += `<line x1="${{hPad.l}}" y1="${{hPad.t+hcH}}" x2="${{hPad.l+hcW}}" y2="${{hPad.t+hcH}}" stroke="#cbd5e1" stroke-width="1"/>`;
  histSvg += `<rect x="${{hPad.l}}" y="${{histH-8}}" width="10" height="10" fill="#059669" rx="2"/>`;
  histSvg += `<text x="${{hPad.l+14}}" y="${{histH-0}}" class="chart-tick">Positive</text>`;
  histSvg += `<rect x="${{hPad.l+80}}" y="${{histH-8}}" width="10" height="10" fill="#0891b2" rx="2"/>`;
  histSvg += `<text x="${{hPad.l+94}}" y="${{histH-0}}" class="chart-tick">Negative</text>`;
  histSvg += '</svg>';

  el.innerHTML = `
  <div class="chart-grid">
    <div class="card">
      <div class="card-title">Per-Plane Outcome Counts</div>
      ${{planeChartHtml}}
    </div>
    <div class="card">
      <div class="card-title">Fused Confidence Distribution</div>
      ${{histSvg}}
    </div>
  </div>
  <div class="card">
    <div class="card-title">Metric Comparison — Cropped vs Uncropped</div>
    ${{metBarsHtml}}
  </div>`;
}}

/* ── SIGNIFICANCE ─────────────────────────────────────────────────── */
function initSig() {{
  const grid = document.getElementById('sigGrid');
  if(grid.childElementCount > 0) return;

  const keyLabels = {{
    cropping_effect_baseline:    'Cropping Effect (Baseline)',
    cropping_effect_comparative: 'Cropping Effect (Comparative)',
    pretraining_effect_uncropped:'Pretraining Effect (Uncropped)',
    pretraining_effect_cropped:  'Pretraining Effect (Cropped)',
  }};

  let any = false;
  Object.entries(SIGNIFICANCE).forEach(([cond, tests]) => {{
    Object.entries(tests).forEach(([key, v]) => {{
      any = true;
      const sig = v['significant_at_0.05'];
      const diff = v.auc_diff_b_minus_a;
      grid.insertAdjacentHTML('beforeend', `
      <div class="sig-card">
        <h4>${{COND_LABELS[cond] || cond}} — ${{keyLabels[key] || key}}</h4>
        <div class="sig-row">
          <span>AUC Difference (B − A)</span>
          <span class="sig-val" style="color:${{diff>=0?'#15803d':'#dc2626'}}">${{fmtDiff(diff)}}</span>
        </div>
        <div class="sig-row">
          <span>95% Confidence Interval</span>
          <span class="sig-val">[${{(v.ci_low*100).toFixed(2)}}%, ${{(v.ci_high*100).toFixed(2)}}%]</span>
        </div>
        <div class="sig-row">
          <span>p-value (two-sided)</span>
          <span class="sig-val">${{v.p_two_sided != null ? v.p_two_sided.toFixed(3) : 'N/A'}}</span>
        </div>
        <div class="sig-row">
          <span>Significant at α = 0.05</span>
          <span class="sig-val" style="color:${{sig?'#15803d':'#dc2626'}}">${{sig ? '✅ Yes' : '❌ No'}}</span>
        </div>
      </div>`);
    }});
  }});

  if(!any) {{
    grid.innerHTML = `<div class="sig-card" style="grid-column:1/-1">
      <p style="color:var(--muted)">No significance tests are available yet.
      They are generated when both cropped and uncropped models exist for the same condition.</p>
    </div>`;
  }}
}}


/* ── GRAD-CAM ─────────────────────────────────────────────────────── */
let gcCond = 'acl', gcMode = 'both';
let gcInited = false;

function initGradCam() {{
  if(!gcInited) {{
    gcInited = true;
    const condTabs = document.getElementById('gcCondTabs');
    ['acl','meniscus','abnormal'].forEach((c,i) => {{
      const btn = document.createElement('button');
      btn.className = 'cond-tab' + (i===0?' active':'');
      btn.textContent = COND_ICONS[c] + ' ' + COND_LABELS[c].replace(' Tear','');
      btn.onclick = () => {{ gcCond = c; setGcActive('gcCondTabs', btn); renderGradCam(); }};
      condTabs.appendChild(btn);
    }});
    const modeTabs = document.getElementById('gcModeTabs');
    [{{id:'both',label:'Both (Compare)'}},{{id:'cropped',label:'🟢 Cropped'}},{{id:'uncropped',label:'🟣 Uncropped'}}].forEach((m,i) => {{
      const btn = document.createElement('button');
      btn.className = 'cond-tab' + (i===0?' active':'');
      btn.textContent = m.label;
      btn.onclick = () => {{ gcMode = m.id; setGcActive('gcModeTabs', btn); renderGradCam(); }};
      modeTabs.appendChild(btn);
    }});
  }}
  renderGradCam();
}}

function setGcActive(tabsId, activeBtn) {{
  document.querySelectorAll('#'+tabsId+' .cond-tab').forEach(b => b.classList.remove('active'));
  activeBtn.classList.add('active');
}}

function renderGradCam() {{
  const grid  = document.getElementById('gcGrid');
  const stats = document.getElementById('gcStats');
  if(!GRADCAM_DATA || !Object.keys(GRADCAM_DATA).length) {{
    grid.innerHTML = '<div class="card" style="text-align:center;padding:40px;color:var(--muted)">No Grad-CAM data embedded. Re-run the dashboard generator with --gradcam_dir set.</div>';
    stats.textContent = '';
    return;
  }}
  const modes = gcMode === 'both' ? ['uncropped','cropped'] : [gcMode];
  
  let allIdxs = new Set();
  modes.forEach(m => {{
    (GRADCAM_DATA[m]||[]).filter(e => e.condition === gcCond).forEach(e => {{
       (e.cases||[]).forEach(c => allIdxs.add(c.idx));
    }});
  }});
  
  if(allIdxs.size === 0) {{
    grid.innerHTML = '<div class="card" style="text-align:center;padding:40px;color:var(--muted)">No data for this combination.</div>';
    stats.textContent = '';
    return;
  }}
  
  stats.innerHTML = `<span style="color:var(--text);font-weight:600">${{COND_LABELS[gcCond]}}</span>&nbsp;&middot;&nbsp;Multi-Plane View`;

  grid.innerHTML = Array.from(allIdxs).sort((a,b)=>a-b).map(idx => {{
    let true_label = 0;
    modes.forEach(m => {{
      (GRADCAM_DATA[m]||[]).filter(e=>e.condition===gcCond).forEach(e => {{
        const cc = (e.cases||[]).find(c=>c.idx===idx);
        if(cc) true_label = cc.true_label;
      }});
    }});
    const gtHtml = true_label === 1
        ? '<span style="color:#15803d;font-weight:600">Positive</span>'
        : '<span style="color:#0891b2;font-weight:600">Negative</span>';
        
    let contentHtml = '';
    modes.forEach((m, mIdx) => {{
      const isCropped = m === 'cropped';
      const mBg = isCropped?'#f0fdf4':'#faf5ff';
      const mTx = isCropped?'#15803d':'#7c3aed';
      const mBd = isCropped?'#bbf7d0':'#ddd8fe';
      const mIc = isCropped?'🟢':'🟣';
      
      const isLast = mIdx === modes.length - 1;
      const borderStyle = isLast ? 'none' : '1px dashed var(--border)';
      
      let rowHtml = `<div style="margin-bottom:16px;padding-bottom:12px;border-bottom:${{borderStyle}}">
        <div style="margin-bottom:12px"><span class="gc-mode-badge" style="background:${{mBg}};color:${{mTx}};border:1px solid ${{mBd}}">${{mIc}} ${{m.toUpperCase()}}</span></div>
        <div class="gc-grid-3">`;
      
      ['axial','coronal','sagittal'].forEach(plane => {{
        const entry = (GRADCAM_DATA[m]||[]).find(e => e.condition === gcCond && e.plane === plane);
        const caseData = entry ? (entry.cases||[]).find(c => c.idx === idx) : null;
        
        if(caseData) {{
          rowHtml += `<div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;font-size:.75rem">
              <span style="font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.5px">${{plane}}</span>
              <span style="color:var(--muted)">Slice #${{caseData.slice_idx}}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:4px">
              <span class="outcome-pill" style="${{outcomeStyle(caseData.outcome)}}">${{caseData.outcome}}</span>
              <span style="font-size:.75rem;color:var(--text2);font-weight:600">Pred: ${{(caseData.pred_prob*100).toFixed(1)}}%</span>
            </div>
            <img class="gc-img" src="data:image/png;base64,${{caseData.img_b64}}" loading="lazy" alt="Grad-CAM"/>
          </div>`;
        }} else {{
           rowHtml += `<div style="display:flex;align-items:center;justify-content:center;padding:40px;color:var(--muted);font-size:.75rem;border:1px dashed var(--border);border-radius:8px">${{plane.charAt(0).toUpperCase()+plane.slice(1)}} N/A</div>`;
        }}
      }});
      rowHtml += `</div></div>`;
      contentHtml += rowHtml;
    }});

    return `<div class="gc-card">
      <div class="gc-card-header">
        <div class="gc-card-meta"><b>Patient #${{idx}}</b></div>
        <div class="gc-card-meta">Ground Truth: ${{gtHtml}}</div>
      </div>
      ${{contentHtml}}
    </div>`;
  }}).join('');
}}

document.getElementById('footerDate').textContent =
  'Generated: ' + new Date().toLocaleDateString('en-IE', {{
    year:'numeric', month:'long', day:'numeric', hour:'2-digit', minute:'2-digit'
  }});
</script>
</body>
</html>"""


def main():
    args = parse_args()
    rd   = Path(args.results_dir)

    conditions_data = {}
    for cond in ['acl', 'meniscus', 'abnormal']:
        path = rd / f'{cond}_triage.json'
        if path.exists():
            conditions_data[cond] = load_json(path)
            print(f"  loaded {path.name} — {len(conditions_data[cond]['cases'])} cases")
        else:
            print(f"  warning: {path} not found, skipping")

    mt_path = rd / 'master_table.json'
    master_table = load_json(mt_path) if mt_path.exists() else []
    print(f"  master_table.json — {len(master_table)} rows")

    significance = {}
    for cond in ['acl', 'meniscus', 'abnormal']:
        path = rd / f'significance_{cond}.json'
        if path.exists():
            significance[cond] = load_json(path)
            print(f"  significance_{cond}.json — loaded")

    gradcam_data = {}
    gd = Path(args.gradcam_dir)
    for mode in ['cropped', 'uncropped']:
        gpath = gd / f'gradcam_{mode}.json'
        if gpath.exists():
            gradcam_data[mode] = load_json(gpath)
            n_cases = sum(len(m.get('cases',[])) for m in gradcam_data[mode])
            print(f"  gradcam_{mode}.json — {len(gradcam_data[mode])} models, {n_cases} cases")
        else:
            print(f"  gradcam_{mode}.json not found, skipping")

    html = build_html(conditions_data, master_table, significance, gradcam_data)
    out  = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')
    print(f"\nDashboard written → {out}")
    print(f"Open with: open {out}")


if __name__ == '__main__':
    main()

