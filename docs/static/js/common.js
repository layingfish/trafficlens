/* Shared helpers: API, state, theme tokens, ECharts + Leaflet utilities. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const State = {
  ds: null,          // current dataset key
  meta: null,        // dataset meta
  geo: {},           // geojson cache by dataset
  cluster: null,     // last clustering result {dataset, labels, zones}
  windows: [],       // built forecasting datasets
  activeWindow: null,
  models: [],        // algorithm catalog
};

async function api(path, opts = {}) {
  if (STATIC) return structuredClone(await staticApi(path, opts));
  const init = { headers: {} , ...opts };
  if (opts.body && typeof opts.body !== 'string') {
    init.body = JSON.stringify(opts.body);
    init.headers['Content-Type'] = 'application/json';
    init.method = init.method || 'POST';
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.json();
}

/* ---------- static copy (GitHub Pages): /api/* answered from files written by tools/build_static.py ---------- */
const STATIC = window.TL_STATIC === true;
const _files = {};
function sGet(path, bin = false) {
  if (!_files[path]) _files[path] = fetch(path).then(r => {
    if (!r.ok) throw new Error(r.status === 404 ? '静态演示版没有预先计算这个组合' : r.statusText);
    return bin ? r.arrayBuffer() : r.json();
  }).catch(e => { delete _files[path]; throw e; });
  return _files[path];
}

async function staticApi(path, opts) {
  const u = new URL(path, 'http://x'), p = u.pathname, q = u.searchParams, b = opts.body || {};
  const method = opts.method || (opts.body ? 'POST' : 'GET');
  const fk = b.feature === 'stats' ? 'stats' : `${b.feature}_${b.norm}`;
  let m;
  if (p === '/api/mode') return { readonly: true };
  if (p === '/api/cluster/run') {
    const pk = Object.keys(b.params || {}).sort().map(k => `${k}-${b.params[k]}`).join('_');
    const [r, e] = await Promise.all([sGet(`api/cluster/run/${b.dataset}/${fk}/${b.method}_${pk}.json`), sGet(`api/cluster/embed/${b.dataset}/${fk}_${b.embed}.json`)]);
    if (r.__error) throw new Error(r.__error);
    return { ...r, ...e };
  }
  if (p === '/api/cluster/elbow') return sGet(`api/cluster/elbow/${b.dataset}/${fk}.json`);
  if (method !== 'GET') throw new Error('静态演示版只能浏览，不能构建数据集、训练或删除结果');
  if ((m = p.match(/^\/api\/eda\/temporal\/(\w+)$/))) return StaticCalc.temporal(m[1], (q.get('regions') || '').split('|').filter(Boolean), q.get('freq') || '1h');
  if ((m = p.match(/^\/api\/eval\/(\w+)\/series$/))) return StaticCalc.series(m[1], +q.get('region'), +q.get('h'));
  if ((m = p.match(/^\/api\/eval\/(\w+)\/sample$/))) return StaticCalc.sample(m[1], +q.get('i'), +q.get('region'));
  if (p === '/api/eval/compare') { const all = await sGet('api/eval/compare.json'); return (q.get('ids') || '').split(',').map(id => all[id]).filter(Boolean); }
  return sGet(p.slice(1) + '.json');
}

/** The few endpoints whose answer depends on free inputs, recomputed in the browser from precomputed parts. */
const StaticCalc = {
  async temporal(key, ids, freq) {
    const ix = await sGet(`api/eda/temporal/${key}/index.json`);
    ids = ids.filter(id => id in ix.pos);
    if (!ids.length) ids = ix.ids.slice(0, 3);
    const R = await Promise.all(ids.map(id => sGet(`api/eda/temporal/${key}/r/${ix.pos[id]}.json`)));
    const dc = await sGet(`api/eda/temporal/${key}/d/${ix.pos[ids[0]]}.json`);
    const r1 = x => Math.round(x * 10) / 10;
    // resample hourly values by summing into bins that start at midnight, like pandas resample(freq).sum()
    const step = { '1h': 1, '2h': 2, '4h': 4, '1D': 24 }[freq] || 1;
    const bins = [], binOf = ix.t.map(t => {
      const lab = step === 1 ? t : `${t.slice(0, 11)}${String(Math.floor(+t.slice(11, 13) / step) * step).padStart(2, '0')}:00:00`;
      if (bins[bins.length - 1] !== lab) bins.push(lab);
      return bins.length - 1;
    });
    const resample = h => { const o = new Array(bins.length).fill(0); h.forEach((v, i) => o[binOf[i]] += v ?? 0); return o.map(r1); };
    // weekday x hour heatmap and hour-of-day box stats of the summed selection
    const agg = ix.t.map((_, i) => R.reduce((s, r) => s + (r.h[i] ?? 0), 0));
    const sum = [...Array(7)].map(() => new Array(24).fill(0)), cnt = sum.map(r => r.slice()), byHour = [...Array(24)].map(() => []);
    ix.t.forEach((t, i) => {
      const dow = (new Date(Date.UTC(+t.slice(0, 4), +t.slice(5, 7) - 1, +t.slice(8, 10))).getUTCDay() + 6) % 7, hr = +t.slice(11, 13);
      sum[dow][hr] += agg[i]; cnt[dow][hr]++; byHour[hr].push(agg[i]);
    });
    const pct = (s, q) => { const x = q / 100 * (s.length - 1), lo = Math.floor(x), hi = Math.ceil(x); return s[lo] + (s[hi] - s[lo]) * (x - lo); };
    const obj = f => Object.fromEntries(ids.map((id, k) => [id, f(R[k])]));
    return {
      regions: ids.map((id, k) => ({ id, label: R[k].label })),
      series: { t: bins, v: obj(r => resample(r.h)) },
      profiles: obj(r => r.profiles),
      heatmap: { dow: ['周一', '周二', '周三', '周四', '周五', '周六', '周日'], values: sum.map((row, d) => row.map((s, h) => cnt[d][h] ? r1(s / cnt[d][h]) : 0)) },
      box: byHour.map(v => { const s = v.sort((a, b) => a - b); return [0, 25, 50, 75, 100].map(q => r1(pct(s, q))); }),
      decomp: dc && !dc.error ? { t: ix.t, ...dc } : dc,
      acf: obj(r => r.acf),
      daily: { t: ix.daily_t, v: obj(r => r.daily), offday: ix.offday },
    };
  },

  async arrays(jid) {
    const meta = await sGet(`api/eval/${jid}/arrays.json`);
    const [win, raw, pred] = await Promise.all([sGet(`api/eval/win/${meta.wid}/meta.json`), sGet(`api/eval/win/${meta.wid}/raw.bin`, true), sGet(`api/eval/${jid}/pred.bin`, true)]);
    const N = win.N, H = win.H, q = new Int16Array(pred), sc = meta.scale;
    return {
      ...win, raw: new Float32Array(raw),
      pred: (i, h, n) => q[(i * H + h) * N + n] * sc[n],
    };
  },
  pick(N, region, get) { if (region >= 0) return get(region); let s = 0; for (let n = 0; n < N; n++) s += get(n); return s; },

  async series(jid, region, h) {
    const a = await this.arrays(jid), { L, H, N, starts, times, raw } = a;
    h = Math.min(Math.max(h, 1), H) - 1;
    const y = [], p = [], t = [];
    starts.forEach((s, i) => {
      const tt = s + L + h;
      t.push(times[tt]);
      y.push(this.pick(N, region, n => raw[tt * N + n]));
      p.push(this.pick(N, region, n => a.pred(i, h, n)));
    });
    const mean = y.reduce((x, v) => x + v, 0) / y.length;
    let ae = 0, ssr = 0, sst = 0;
    y.forEach((v, i) => { ae += Math.abs(p[i] - v); ssr += (p[i] - v) ** 2; sst += (v - mean) ** 2; });
    return { t, y, p, metrics: { MAE: ae / y.length, R2: sst > 0 ? 1 - ssr / sst : null } };
  },

  async sample(jid, i, region) {
    const a = await this.arrays(jid), { L, H, N, starts, times, raw } = a;
    i = Math.min(Math.max(i, 0), starts.length - 1);
    const s = starts[i], row = tt => this.pick(N, region, n => raw[tt * N + n]);
    const span = (from, n) => [...Array(n).keys()].map(k => from + k);
    return {
      i, t_in: times.slice(s, s + L), t_out: times.slice(s + L, s + L + H),
      x: span(s, L).map(row), y: span(s + L, H).map(row), p: span(0, H).map(h => this.pick(N, region, n => a.pred(i, h, n))),
    };
  },
};

function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg; t.className = 'toast show' + (err ? ' err' : '');
  clearTimeout(t._h); t._h = setTimeout(() => t.className = 'toast', err ? 5000 : 2600);
}

async function busy(el, fn) {
  el = typeof el === 'string' ? $(el) : el;
  el && el.classList.add('loading');
  try { return await fn(); } catch (e) { toast(e.message, true); console.error(e); } finally { el && el.classList.remove('loading'); }
}

/* ---------- tokens ---------- */
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function T() {
  return {
    surface: cssVar('--surface-1'), surface2: cssVar('--surface-2'), text: cssVar('--text-primary'),
    text2: cssVar('--text-secondary'), muted: cssVar('--text-muted'), grid: cssVar('--grid'), border: cssVar('--border'),
    series: [1, 2, 3, 4, 5, 6, 7, 8].map(i => cssVar('--s' + i)),
    seq: cssVar('--seq').split(',').map(s => s.trim()),
    neg: cssVar('--div-neg'), mid: cssVar('--div-mid'), pos: cssVar('--div-pos'), other: cssVar('--other'),
    accent: cssVar('--accent'),
  };
}
const isDark = () => getComputedStyle(document.documentElement).colorScheme === 'dark';

/* ---------- number formatting ---------- */
function fmt(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '-';
  const a = Math.abs(v);
  const sig = x => Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(2);
  if (a >= 1e9) return sig(v / 1e9) + 'B';
  if (a >= 1e6) return sig(v / 1e6) + 'M';
  if (a >= 1e4) return (v / 1e3).toFixed(1) + 'k';
  if (a >= 100) return v.toFixed(0);
  return v.toFixed(d);
}
const fmtFull = (v, d = 2) => (v === null || v === undefined) ? '-' : Number(v).toLocaleString('zh-CN', { maximumFractionDigits: d });
const fmtTime = s => s.replace('T', ' ').slice(0, 16);

/* ---------- ECharts ---------- */
const Charts = {};
function chart(id) {
  const el = typeof id === 'string' ? document.getElementById(id) : id;
  if (!Charts[el.id]) Charts[el.id] = echarts.init(el, null, { renderer: 'canvas' });
  return Charts[el.id];
}
window.addEventListener('resize', () => Object.values(Charts).forEach(c => c.resize()));
function resizeVisible() { requestAnimationFrame(() => Object.values(Charts).forEach(c => { if (c.getDom().offsetParent) c.resize(); })); }

/** Base option with recessive axes/grid and crosshair tooltip. */
function baseOpt(extra = {}) {
  const t = T();
  return {
    color: t.series,
    textStyle: { fontFamily: getComputedStyle(document.body).fontFamily, color: t.text2 },
    animationDuration: 300,
    grid: { left: 58, right: 18, top: 34, bottom: 34, containLabel: false },
    tooltip: {
      trigger: 'axis', confine: true, backgroundColor: t.surface, borderColor: t.border,
      textStyle: { color: t.text, fontSize: 13 }, padding: [8, 12], extraCssText: 'border-radius:3px;box-shadow:0 4px 16px rgba(0,0,0,.08);', axisPointer: { type: 'line', lineStyle: { color: t.muted, type: 'dashed' } },
      valueFormatter: v => fmtFull(v),
    },
    legend: { top: 0, right: 0, icon: 'roundRect', itemWidth: 14, itemHeight: 3, itemGap: 14, textStyle: { color: t.text2, fontSize: 13 } },
    ...extra,
  };
}
function axisStyle(extra = {}) {
  const t = T();
  return {
    axisLine: { lineStyle: { color: t.border } }, axisTick: { show: false },
    axisLabel: { color: t.muted, fontSize: 12 }, splitLine: { lineStyle: { color: t.grid } },
    nameTextStyle: { color: t.text2, fontSize: 12 }, ...extra,
  };
}
const valAxis = (extra = {}) => ({ type: 'value', ...axisStyle({ axisLabel: { color: T().muted, fontSize: 12, formatter: v => fmt(v) } }), ...extra });
const timeAxis = (extra = {}) => ({ type: 'time', ...axisStyle({ splitLine: { show: false } }), ...extra });
const catAxis = (data, extra = {}) => ({ type: 'category', data, ...axisStyle({ splitLine: { show: false } }), ...extra });
const line = (name, data, extra = {}) => ({ name, type: 'line', data, showSymbol: false, symbolSize: 8, lineStyle: { width: 2 }, emphasis: { focus: 'series' }, ...extra });
const zoom = () => [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 4, borderColor: 'transparent', textStyle: { color: T().muted, fontSize: 12 }, fillerColor: 'rgba(43,95,158,.10)', handleStyle: { color: T().surface, borderColor: T().accent } }];

/* ---------- color scales ---------- */
function quantileScale(values, ramp) {
  const v = values.filter(x => x !== null && Number.isFinite(x)).sort((a, b) => a - b);
  const n = ramp.length;
  const breaks = [];
  for (let i = 1; i < n; i++) breaks.push(v[Math.min(v.length - 1, Math.floor(i / n * v.length))]);
  const f = x => {
    if (x === null || !Number.isFinite(x)) return T().other;
    let i = 0; while (i < breaks.length && x > breaks[i]) i++;
    return ramp[i];
  };
  f.min = v[0]; f.max = v[v.length - 1]; f.ramp = ramp; f.breaks = breaks;
  return f;
}
function linearScale(min, max, ramp) {
  const f = x => {
    if (x === null || !Number.isFinite(x)) return T().other;
    const r = max > min ? (x - min) / (max - min) : 0.5;
    return ramp[Math.max(0, Math.min(ramp.length - 1, Math.floor(r * ramp.length)))];
  };
  f.min = min; f.max = max; f.ramp = ramp;
  return f;
}
function catColor(i) { const s = T().series; return (i === null || i < 0 || i >= s.length) ? T().other : s[i]; }

/* ---------- Leaflet choropleth ---------- */
class GeoMap {
  constructor(id, { onClick } = {}) {
    this.el = document.getElementById(id);
    this.map = L.map(this.el, { zoomControl: true, attributionControl: true, preferCanvas: true, zoomSnap: 0.25 }).setView([45, 10], 8);
    this.tiles = null;
    this.onClick = onClick;
    this.layer = null; this.legend = null; this.key = null;
    this.setTiles();
  }
  setTiles() {
    if (this.tiles) this.map.removeLayer(this.tiles);
    this.tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18, attribution: '© OpenStreetMap contributors', className: 'basemap',
    }).addTo(this.map);
  }
  setGeo(key, geo) {
    if (this.key === key && this.layer) return;
    if (this.layer) this.map.removeLayer(this.layer);
    this.key = key;
    this.idx = {};
    this.layer = L.geoJSON(geo, {
      style: () => ({ weight: 0.6, color: T().surface, fillOpacity: 0.8, fillColor: T().other }),
      onEachFeature: (f, lyr) => {
        this.idx[f.properties.rid] = lyr;
        lyr.on('click', () => this.onClick && this.onClick(f.properties.rid, f.properties.label));
        lyr.on('mouseover', () => lyr.setStyle({ weight: 2, color: T().text }));
        lyr.on('mouseout', () => lyr.setStyle({ weight: 0.6, color: T().surface }));
      },
    }).addTo(this.map);
    this._fitted = false;
    this.invalidate();
  }
  /** fill: rid -> color; tip: rid -> html */
  paint(fill, tip) {
    if (!this.layer) return;
    this.layer.eachLayer(l => {
      const rid = l.feature.properties.rid;
      l.setStyle({ fillColor: fill(rid) });
      l.unbindTooltip();
      l.bindTooltip(tip(rid), { sticky: true });
    });
  }
  setLegend(html) {
    if (this.legend) this.map.removeControl(this.legend);
    this.legend = L.control({ position: 'bottomright' });
    this.legend.onAdd = () => { const d = L.DomUtil.create('div', 'map-legend'); d.innerHTML = html; return d; };
    this.legend.addTo(this.map);
  }
  /** Call when the map becomes visible: Leaflet can't measure a hidden container. */
  invalidate() {
    setTimeout(() => {
      if (!this.el.offsetParent) return;
      this.map.invalidateSize();
      if (!this._fitted && this.layer) { this.map.fitBounds(this.layer.getBounds(), { padding: [10, 10] }); this._fitted = true; }
    }, 60);
  }
}
function rampLegend(title, scale, f = fmt) {
  return `<div class="ttl">${title}</div><div class="ramp">${scale.ramp.map(c => `<span style="background:${c}"></span>`).join('')}</div>
    <div class="ends"><span>${f(scale.min)}</span><span>${f(scale.max)}</span></div>`;
}
function catLegend(title, items) {
  return `<div class="ttl">${title}</div>` + items.map(([c, l]) => `<div class="item"><i style="background:${c}"></i>${l}</div>`).join('');
}

async function getGeo(key) {
  if (!State.geo[key]) State.geo[key] = await api('/api/geo/' + key);
  return State.geo[key];
}

/* ---------- table helper with sortable headers ---------- */
function renderTable(el, cols, rows, { onRow, sortKey, desc = true, selected } = {}) {
  el = typeof el === 'string' ? $(el) : el;
  el._state = el._state || { sortKey, desc };
  const st = el._state;
  const sorted = st.sortKey ? [...rows].sort((a, b) => {
    const x = a[st.sortKey], y = b[st.sortKey];
    if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1;
    return (x > y ? 1 : x < y ? -1 : 0) * (st.desc ? -1 : 1);
  }) : rows;
  el.innerHTML = `<table><thead><tr>${cols.map(c => `<th data-k="${c.key}" class="${c.cls || ''}">${c.label}${st.sortKey === c.key ? (st.desc ? ' ↓' : ' ↑') : ''}</th>`).join('')}</tr></thead>
    <tbody>${sorted.map((r, i) => `<tr data-i="${rows.indexOf(r)}" class="${selected && selected(r) ? 'sel' : ''}">${cols.map(c => `<td class="${c.cls || ''}">${c.render ? c.render(r[c.key], r) : (r[c.key] ?? '-')}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  $$('th', el).forEach(th => th.onclick = () => {
    const k = th.dataset.k; if (st.sortKey === k) st.desc = !st.desc; else { st.sortKey = k; st.desc = true; }
    renderTable(el, cols, rows, { onRow, selected });
  });
  if (onRow) $$('tbody tr', el).forEach(tr => tr.onclick = (e) => onRow(rows[+tr.dataset.i], e));
}

function kpi(k, v, s = '') { return `<div class="kpi"><div class="k">${k}</div><div class="v" title="${String(v).replace(/"/g, '')}">${v}</div>${s ? `<div class="s">${s}</div>` : ''}</div>`; }
function regionLabel(id) { const r = State.meta.regions.find(x => x.id === id); return r ? r.label : id; }
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
