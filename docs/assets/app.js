import { CLUSTER_COLORS, equityColor, clamp, INDICATOR_LABELS, fmt } from "./utils.js";

// Resolve from this module so paths work on GitHub Pages, local /docs server, or nested URLs.
const DATA_BASE = new URL("../outputs/", import.meta.url);
const DATA_GEOJSON = new URL("grid_points.geojson", DATA_BASE).href;
const DATA_META = new URL("metadata.json", DATA_BASE).href;
const DATA_SUMMARY = new URL("cluster_summary.csv", DATA_BASE).href;
const DATA_Z = new URL("cluster_feature_zscores.csv", DATA_BASE).href;
const DATA_POINT_ADVICE = new URL("grid_point_advice.json", DATA_BASE).href;

const els = {
  colorMode: document.getElementById("colorMode"),
  clusterFilter: document.getElementById("clusterFilter"),
  equityMin: document.getElementById("equityMin"),
  equityMax: document.getElementById("equityMax"),
  applyFilters: document.getElementById("applyFilters"),
  legend: document.getElementById("legend"),
  dataPath: document.getElementById("dataPath"),
  selectionEmpty: document.getElementById("selectionEmpty"),
  selection: document.getElementById("selection"),
  selGridId: document.getElementById("selGridId"),
  selCluster: document.getElementById("selCluster"),
  selEquity: document.getElementById("selEquity"),
  selTop: document.getElementById("selTop"),
  clusterLink: document.getElementById("clusterLink"),
  clearSelection: document.getElementById("clearSelection"),

  // report section
  reportCluster: document.getElementById("reportCluster"),
  reportAnchor: document.getElementById("report"),
  clusterName: document.getElementById("clusterName"),
  statN: document.getElementById("statN"),
  statEquityMean: document.getElementById("statEquityMean"),
  statEquityMedian: document.getElementById("statEquityMedian"),
  statEquityBand: document.getElementById("statEquityBand"),
  statPerf: document.getElementById("statPerf"),
  statNeed: document.getElementById("statNeed"),
  direNeeds: document.getElementById("direNeeds"),
  priorityQueue: document.getElementById("priorityQueue"),
  pcaS: document.getElementById("pcaS"),
  pcaN: document.getElementById("pcaN"),

  pointAdviceCompact: document.getElementById("pointAdviceCompact"),
  pointAdviceCompactText: document.getElementById("pointAdviceCompactText"),
  pointNeedPanelMeta: document.getElementById("pointNeedPanelMeta"),
  pointDireNeeds: document.getElementById("pointDireNeeds"),
  pointQueue: document.getElementById("pointQueue"),
};

els.dataPath.textContent = "outputs/grid_points.geojson";

let meta = null;
let map = null;
let layer = null;
let geo = null;
let summaryRows = [];
let zRows = [];
let zChart = null;
const EQUITY_HIST_BINS_MAX = 10;

/** @type {Record<string, any> | null} */
let pointAdviceByGrid = null;
/** @type {object | null} */
let selectedPointProps = null;

function clusterName(c) {
  if (!meta?.config?.cluster_names) return `Cluster ${c}`;
  // keys may be strings in JSON
  return meta.config.cluster_names[String(c)] ?? meta.config.cluster_names[c] ?? `Cluster ${c}`;
}

function featureName(code) {
  return INDICATOR_LABELS[code] ?? code;
}

function humanizeTopFeatures(raw) {
  const text = String(raw ?? "").trim();
  if (!text) return "—";
  const names = text
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((code) => featureName(code));
  return names.length ? names.join(", ") : text;
}

function passesFilters(props) {
  const cf = els.clusterFilter.value;
  if (cf !== "all" && String(props.cluster) !== cf) return false;

  const emin = clamp(Number(els.equityMin.value), 0, 100);
  const emax = clamp(Number(els.equityMax.value), 0, 100);
  const eq = Number(props.equity_score);
  if (Number.isFinite(eq) && (eq < Math.min(emin, emax) || eq > Math.max(emin, emax))) return false;

  return true;
}

function selectedEquityRange() {
  const a = clamp(Number(els.equityMin.value), 0, 100);
  const b = clamp(Number(els.equityMax.value), 0, 100);
  return [Math.min(a, b), Math.max(a, b)];
}

function histogramBinsForSpan(span) {
  // Keep bins readable for narrow ranges (e.g. 46-47) while preserving detail.
  const bySpan = Math.round(span * 2);
  return Math.max(4, Math.min(EQUITY_HIST_BINS_MAX, bySpan));
}

function fmtBinEdge(v, span) {
  if (span <= 2) return v.toFixed(2);
  if (span <= 10) return v.toFixed(1);
  return v.toFixed(0);
}

function markerStyle(props) {
  const mode = els.colorMode.value;
  if (mode === "cluster") {
    return { color: CLUSTER_COLORS[props.cluster] ?? "#888", fillColor: CLUSTER_COLORS[props.cluster] ?? "#888" };
  }
  const eq = Number(props.equity_score);
  const [emin, emax] = selectedEquityRange();
  const span = Math.max(1e-9, emax - emin);
  // Stretch color mapping to the selected range (ex: 40-60) for better separation.
  const c = equityColor(clamp((eq - emin) / span, 0, 1));
  return { color: c, fillColor: c };
}

function getEquityHistogram(features, rangeMin, rangeMax, bins = EQUITY_HIST_BINS_MAX) {
  const counts = Array.from({ length: bins }, () => 0);
  let total = 0;
  const span = Math.max(1e-9, rangeMax - rangeMin);
  for (const feature of features ?? []) {
    const score = Number(feature?.properties?.equity_score);
    if (!Number.isFinite(score)) continue;
    if (score < rangeMin || score > rangeMax) continue;
    const idx = Math.min(bins - 1, Math.floor(((score - rangeMin) / span) * bins));
    counts[idx] += 1;
    total += 1;
  }
  return { counts, total };
}

function renderEquityHistogram() {
  if (!geo?.features?.length) {
    return `<div class="legendHint">Distribution loading...</div>`;
  }
  const [emin, emax] = selectedEquityRange();
  const span = Math.max(1e-9, emax - emin);
  const bins = histogramBinsForSpan(span);
  const { counts, total } = getEquityHistogram(geo.features, emin, emax, bins);
  if (!total) {
    return `<div class="legendHint">No equity scores in ${emin.toFixed(0)}-${emax.toFixed(0)}.</div>`;
  }

  const maxCount = Math.max(...counts, 1);
  const bars = counts
    .map((count, i) => {
      const lo = emin + i * (span / bins);
      const hi = emin + (i + 1) * (span / bins);
      const h = Math.max(4, Math.round((count / maxCount) * 36));
      return `<div class="histBar" style="height:${h}px" title="${fmtBinEdge(lo, span)}-${fmtBinEdge(hi, span)}: ${count}"></div>`;
    })
    .join("");
  const catRows = counts
    .map((count, i) => {
      const lo = emin + i * (span / bins);
      const hi = emin + (i + 1) * (span / bins);
      return `<div class="histCat"><span>${fmtBinEdge(lo, span)}-${fmtBinEdge(hi, span)}</span><b>${count.toLocaleString()}</b></div>`;
    })
    .join("");

  return `
    <div class="histWrap">
      <div class="histHeader">
        <span>Distribution ${emin.toFixed(0)}-${emax.toFixed(0)}</span>
        <span>n=${total.toLocaleString()}</span>
      </div>
      <div class="histBars">${bars}</div>
      <div class="histLabels">
        <span>${fmtBinEdge(emin, span)}</span>
        <span>${fmtBinEdge((emin + emax) / 2, span)}</span>
        <span>${fmtBinEdge(emax, span)}</span>
      </div>
      <div class="histCats">${catRows}</div>
    </div>
  `;
}

function renderLegend() {
  const mode = els.colorMode.value;
  if (mode === "cluster") {
    els.legend.innerHTML = `
      <div class="legendTitle">Legend: Cluster</div>
      ${[0, 1, 2, 3]
        .map(
          (c) => `
        <div class="legendRow">
          <div class="swatch" style="background:${CLUSTER_COLORS[c]}"></div>
          <div>${clusterName(c)}</div>
        </div>`
        )
        .join("")}
    `;
    return;
  }
  const [emin, emax] = selectedEquityRange();
  els.legend.innerHTML = `
    <div class="legendTitle">Legend: Equity score</div>
    <div class="ramp"></div>
    <div class="rampLabels"><span>${emin.toFixed(0)}</span><span>${((emin + emax) / 2).toFixed(0)}</span><span>${emax.toFixed(0)}</span></div>
    ${renderEquityHistogram()}
    <div class="legendHint">Red = lower (within selected range)</div>
  `;
}

function clearSelection() {
  els.selectionEmpty.classList.remove("hidden");
  els.selection.classList.add("hidden");
  selectedPointProps = null;
  renderPointLevelAdvice();
  window.dispatchEvent(new CustomEvent("equity-selection-changed"));
}

function setSelection(props) {
  selectedPointProps = props;
  els.selectionEmpty.classList.add("hidden");
  els.selection.classList.remove("hidden");
  els.selGridId.textContent = props.grid_id ?? "—";
  els.selCluster.textContent = clusterName(props.cluster);
  els.selEquity.textContent = Number.isFinite(Number(props.equity_score)) ? Number(props.equity_score).toFixed(1) : "—";
  els.selTop.textContent = humanizeTopFeatures(props.top3_features);
  els.clusterLink.href = `#report`;

  // Update report to this cluster and scroll down
  setReportCluster(props.cluster);
  setTimeout(() => {
    els.reportAnchor?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 0);

  window.dispatchEvent(new CustomEvent("equity-selection-changed"));
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} loading ${url}`);
  return r.json();
}

function parseCsv(url) {
  return new Promise((resolve, reject) => {
    Papa.parse(url, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (res) => resolve(res.data),
      error: (err) => reject(err),
    });
  });
}

function renderPca() {
  const p = meta?.pca_weights?.service_performance ?? {};
  const n = meta?.pca_weights?.service_need ?? {};
  els.pcaS.textContent = JSON.stringify(p, null, 2);
  els.pcaN.textContent = JSON.stringify(n, null, 2);
}

function renderSummary(c) {
  const row = summaryRows.find((r) => String(r.cluster) === String(c));
  els.clusterName.textContent = clusterName(c);
  if (!row) return;
  els.statN.textContent = row.n_grids_scored?.toLocaleString?.() ?? String(row.n_grids_scored ?? "—");
  els.statEquityMean.textContent = fmt(row.equity_mean, 2);
  els.statEquityMedian.textContent = fmt(row.equity_median, 2);
  els.statEquityBand.textContent = `${fmt(row.equity_p10, 2)} → ${fmt(row.equity_p90, 2)}`;
  els.statPerf.textContent = fmt(row.performance_mean, 2);
  els.statNeed.textContent = fmt(row.need_mean, 2);
}

function renderZChart(c) {
  const row = zRows.find((r) => String(r.cluster) === String(c));
  if (!row) return;

  const feats = Object.keys(row).filter(
    (k) => k !== "cluster" && row[k] !== null && row[k] !== undefined && !Number.isNaN(row[k])
  );
  const items = feats
    .map((k) => ({ k, z: Number(row[k]) }))
    .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))
    .slice(0, 6)
    .reverse();

  const labels = items.map((d) => INDICATOR_LABELS[d.k] ?? d.k);
  const data = items.map((d) => d.z);
  const colors = items.map((d) => (d.z >= 0 ? "rgba(34,197,94,.65)" : "rgba(239,68,68,.65)"));
  const borders = items.map((d) => (d.z >= 0 ? "rgba(34,197,94,1)" : "rgba(239,68,68,1)"));

  const ctx = document.getElementById("zChart");
  if (zChart) zChart.destroy();
  zChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "z-score", data, backgroundColor: colors, borderColor: borders, borderWidth: 1 }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          min: -2,
          max: 2,
          grid: { color: "rgba(255,255,255,.06)" },
          ticks: { color: "rgba(232,234,240,.75)", stepSize: 1 },
        },
        y: { grid: { display: false }, ticks: { color: "rgba(232,234,240,.85)" } },
      },
    },
  });
}

function renderHeuristics(c) {
  const h = meta?.heuristics?.[String(c)] ?? meta?.heuristics?.[c];
  if (!h) {
    els.direNeeds.textContent = "—";
    els.priorityQueue.textContent = "—";
    return;
  }

  const priClass = (p) => (p || "").toLowerCase();
  els.direNeeds.innerHTML = (h.needs ?? [])
    .map((n) => {
      const actions = (n.actions ?? []).map((a) => `<li>${a}</li>`).join("");
      return `
        <div class="needCard">
          <div class="pill ${priClass(n.priority)}">${n.priority} · #${n.rank}</div>
          <div class="needTitle">${n.title}</div>
          <div class="needDesc">${n.desc}</div>
          <ul class="smallNote" style="margin:0;padding-left:18px">${actions}</ul>
        </div>
      `;
    })
    .join("");

  els.priorityQueue.innerHTML = (h.queue ?? [])
    .map(([num, action, why]) => {
      return `
        <div class="queueItem">
          <div class="queueNum">${num}</div>
          <div>
            <div class="queueAction">${action}</div>
            <div class="queueWhy">→ ${why}</div>
          </div>
        </div>
      `;
    })
    .join("");
}

function pointPriClass(p) {
  const x = (p || "").toLowerCase();
  if (x === "info") return "info";
  return x;
}

function renderPointLevelAdvice() {
  const setEmpty = (msg) => {
    if (els.pointDireNeeds) els.pointDireNeeds.textContent = "—";
    if (els.pointQueue) els.pointQueue.textContent = "—";
    if (els.pointNeedPanelMeta) els.pointNeedPanelMeta.textContent = msg;
    if (els.pointAdviceCompact) els.pointAdviceCompact.classList.add("hidden");
  };

  if (!selectedPointProps || !selectedPointProps.grid_id) {
    setEmpty("Select a point on the map for grid-level needs and actions.");
    if (els.pointAdviceCompact) els.pointAdviceCompact.classList.add("hidden");
    return;
  }

  if (pointAdviceByGrid == null) {
    setEmpty("Point advice file not found (re-run the pipeline to generate grid_point_advice.json).");
    if (els.pointAdviceCompact) els.pointAdviceCompact.classList.add("hidden");
    return;
  }

  if (Object.keys(pointAdviceByGrid).length === 0) {
    setEmpty("Point advice is empty. Run the pipeline (it writes grid_point_advice.json from your grid scores and optional data/*.csv files).");
    if (els.pointAdviceCompact) els.pointAdviceCompact.classList.add("hidden");
    return;
  }

  const gid = String(selectedPointProps.grid_id);
  const block = pointAdviceByGrid[gid] ?? null;
  if (!block) {
    setEmpty(`No point-level advice for grid ${gid} (re-run the pipeline with the same input grid set).`);
    return;
  }

  const needs = (block.needs ?? []).filter((n) => n && n.title);
  const visibleNeeds = needs.filter((n) => n.feature !== "file_context");
  const sols = block.solutions ?? [];

  if (els.pointNeedPanelMeta) {
    const feat = (visibleNeeds.find((n) => n.feature) || {}).feature;
    els.pointNeedPanelMeta.textContent = `Grid ${gid}` + (feat ? ` · strongest signal: ${featureName(feat)}` : "");
  }

  if (els.pointDireNeeds) {
    if (!visibleNeeds.length) {
      els.pointDireNeeds.textContent = "—";
    } else {
      els.pointDireNeeds.innerHTML = visibleNeeds
        .map((n) => {
          const tag = `${INDICATOR_LABELS[n.feature] || n.feature} (z ${n.z})`;
          return `
        <div class="needCard">
          <div class="pill ${pointPriClass(n.priority)}">${n.priority} · ${tag}</div>
          <div class="needTitle">${n.title}</div>
          <div class="needDesc">${n.desc}</div>
        </div>
      `;
        })
        .join("");
    }
  }

  if (els.pointQueue) {
    if (!sols.length) {
      els.pointQueue.textContent = "—";
    } else {
      els.pointQueue.innerHTML = sols
        .map((s, i) => {
          return `
        <div class="queueItem">
          <div class="queueNum">${String(i + 1).padStart(2, "0")}</div>
          <div>
            <div class="queueAction">${s}</div>
            <div class="queueWhy">Point-level action (from feature templates)</div>
          </div>
        </div>
      `;
        })
        .join("");
    }
  }

  if (els.pointAdviceCompact && els.pointAdviceCompactText) {
    const dataNeeds = needs.filter((n) => n.feature && n.feature !== "file_context");
    const t1 = dataNeeds[0]?.title;
    const t2 = dataNeeds[1]?.title;
    els.pointAdviceCompactText.textContent = t1
      ? t2
        ? `${t1} · ${t2}`
        : t1
      : (needs[0] && needs[0].title) || "—";
    els.pointAdviceCompact.classList.remove("hidden");
  }
}

function setReportCluster(c) {
  const v = String(c);
  if (els.reportCluster) els.reportCluster.value = v;
  renderSummary(v);
  renderZChart(v);
  renderHeuristics(v);
  renderPointLevelAdvice();
  window.dispatchEvent(new CustomEvent("equity-selection-changed"));
}

function rebuildLayer() {
  if (!map || !geo) return;
  if (layer) layer.remove();

  layer = L.geoJSON(geo, {
    filter: (feature) => passesFilters(feature.properties ?? {}),
    pointToLayer: (feature, latlng) => {
      const props = feature.properties ?? {};
      const style = markerStyle(props);
      return L.circleMarker(latlng, {
        radius: 5,
        weight: 1,
        opacity: 0.9,
        fillOpacity: 0.85,
        ...style,
      });
    },
    onEachFeature: (feature, l) => {
      const p = feature.properties ?? {};
      l.on("click", () => setSelection(p));
      l.bindTooltip(
        `<div style="font-family: ui-sans-serif, system-ui; font-size:12px">
          <div><b>${p.grid_id ?? "grid"}</b></div>
          <div>${clusterName(p.cluster)}</div>
          <div>Equity: ${Number.isFinite(Number(p.equity_score)) ? Number(p.equity_score).toFixed(1) : "—"}</div>
        </div>`,
        { sticky: true }
      );
    },
  }).addTo(map);
}

async function init() {
  const [m, g, summary, z, paJ] = await Promise.all([
    fetchJson(DATA_META),
    fetchJson(DATA_GEOJSON),
    parseCsv(DATA_SUMMARY),
    parseCsv(DATA_Z),
    fetchJson(DATA_POINT_ADVICE).catch(() => null),
  ]);
  pointAdviceByGrid = paJ?.by_grid != null ? paJ.by_grid : null;
  meta = m;
  geo = g;
  summaryRows = summary;
  zRows = z;

  window.dispatchEvent(
    new CustomEvent("equity-dashboard-ready", {
      detail: { meta, geo, summaryRows },
    })
  );

  map = L.map("map", { zoomControl: true, minZoom: 12 }).setView([37.77, -122.44], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);

  clearSelection();
  renderLegend();
  renderPca();
  setReportCluster("0");

  rebuildLayer();

  const syncHeroScrollCue = () => {
    document.body.classList.toggle("heroCollapsed", window.scrollY > 320);
  };
  window.addEventListener("scroll", syncHeroScrollCue, { passive: true });
  syncHeroScrollCue();
}

els.applyFilters.addEventListener("click", () => {
  renderLegend();
  rebuildLayer();
});
els.colorMode.addEventListener("change", () => {
  renderLegend();
  rebuildLayer();
});
els.clearSelection.addEventListener("click", () => clearSelection());
els.reportCluster?.addEventListener("change", (e) => setReportCluster(e.target.value));

init().catch((err) => {
  console.error(err);
  alert(
    `Failed to load dashboard data.\n\n${String(err?.message || err)}\n\n` +
      "If you opened this as a file, run: python -m http.server 5173 --directory docs\n" +
      "Ensure docs/outputs contains the pipeline files (run run_pipeline.py --output-dir docs/outputs)."
  );
});

