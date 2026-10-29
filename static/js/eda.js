/* Page 1 — exploratory analysis: temporal & spatial dynamics. */
const EDA = {
  sel: [],            // [{id, slot}] slot = fixed color index so colors follow the region
  overview: null, spatial: null, map: null, lisaMap: null,
  mode: 'metric', timer: null,

  init() {
    this.map = new GeoMap('mapEda', { onClick: (rid) => this.addRegion(rid) });
    this.lisaMap = new GeoMap('mapLisa', { onClick: (rid) => this.addRegion(rid) });
    $('#edaFreq').onchange = () => this.loadTemporal();
    $('#edaTop').onclick = () => { this.sel = []; this.overview.table.slice(0, 3).forEach(r => this.addRegion(r.id, false)); this.loadTemporal(); };
    $('#edaRegionInput').onchange = (e) => {
      const r = State.meta.regions.find(x => x.label === e.target.value || x.id === e.target.value);
      if (r) this.addRegion(r.id); e.target.value = '';
    };
    $$('#mapMode button').forEach(b => b.onclick = () => {
      $$('#mapMode button').forEach(x => x.classList.toggle('on', x === b));
      this.mode = b.dataset.mode; this.stop(); this.setupSlider(); this.paintMap();
    });
    $('#mapMetric').onchange = () => this.paintMap();
    $('#hourType').onchange = () => this.paintMap();
    $('#hourScale').onchange = () => this.paintMap();
    $('#timeSlider').oninput = () => this.paintMap();
    $('#playBtn').onclick = () => this.timer ? this.stop() : this.play();
  },

  async load() {
    const key = State.ds, m = State.meta;
    $('#edaSubtitle').textContent = `${m.title}，共 ${m.n_regions} 个区域，原始粒度 ${m.freq}，时间范围 ${m.start.slice(0, 10)} 至 ${m.end.slice(0, 10)}。`;
    $('#regionList').innerHTML = m.regions.map(r => `<option value="${escapeHtml(r.label)}">`).join('');
    $('#edaFreq').innerHTML = m.freq_options.map(f => `<option value="${f}" ${f === '1h' ? 'selected' : ''}>${f === 'native' ? '原始 ' + m.freq : f}</option>`).join('');
    const geo = await getGeo(key);
    this.map.setGeo(key, geo); this.lisaMap.setGeo(key, geo);
    await busy('#page-eda', async () => {
      this.overview = await api('/api/eda/overview/' + key);
      this.renderOverview();
      this.sel = [];
      this.overview.table.slice(0, 3).forEach(r => this.addRegion(r.id, false));
      await this.loadTemporal();
    });
    busy($('#mapEda').closest('.card'), async () => {
      this.spatial = await api('/api/eda/spatial/' + key);
      this.renderSpatial();
    });
  },

  renderOverview() {
    const o = this.overview, k = o.kpi, m = o.meta;
    $('#edaKpis').innerHTML = [
      kpi('区域数', m.n_regions, m.title.split('（')[0]),
      kpi('时间跨度', `${Math.round((new Date(m.end) - new Date(m.start)) / 864e5)} 天`, `${m.n_steps.toLocaleString()} 个 ${m.freq} 时间点`),
      kpi('总流量', fmt(k.total), m.unit),
      kpi('全网小时峰值', fmt(k.peak_value), fmtTime(k.peak_time)),
      kpi('区域-小时平均', fmt(k.mean_per_region_hour), '每区域每小时'),
      kpi('变异系数中位数', k.cv_median.toFixed(2), '小时序列 std/mean'),
    ].join('');
    const t = T();
    chart('chTotal').setOption(baseOpt({
      legend: { show: false },
      xAxis: timeAxis(), yAxis: valAxis(), dataZoom: zoom(), grid: { left: 56, right: 18, top: 16, bottom: 44 },
      series: [line('全网流量', o.total_hourly.t.map((x, i) => [x, o.total_hourly.v[i]]), { areaStyle: { opacity: 0.08 } })],
    }), true);
    chart('chDaily').setOption(baseOpt({
      legend: { show: false }, grid: { left: 56, right: 18, top: 10, bottom: 24 },
      tooltip: { ...baseOpt().tooltip, trigger: 'axis', axisPointer: { type: 'shadow' } },
      xAxis: catAxis(o.total_daily.t, { axisLabel: { color: t.muted, fontSize: 11, formatter: v => v.slice(5) } }),
      yAxis: valAxis(),
      series: [{ name: '日总量', type: 'bar', barCategoryGap: '20%', data: o.total_daily.v.map((v, i) => ({ value: v, itemStyle: { color: t.series[0], opacity: o.total_daily.offday[i] ? 1 : 0.45, borderRadius: [3, 3, 0, 0] } })) }],
    }), true);
    const cols = [
      { key: 'label', label: '区域', cls: 'l' },
      { key: 'mean', label: '小时均值', render: v => fmt(v) },
      { key: 'std', label: '标准差', render: v => fmt(v) },
      { key: 'cv', label: 'CV', render: v => v.toFixed(2) },
      { key: 'max', label: '最大值', render: v => fmt(v) },
      { key: 'peak_hour', label: '峰值时刻', render: v => `${v}:00` },
      { key: 'offday_ratio', label: '休/工比', render: v => v.toFixed(2) },
      { key: 'night_share', label: '夜间占比', render: v => v.toFixed(2) },
      { key: 'day_night_ratio', label: '昼夜比', render: v => v.toFixed(2) },
    ];
    $('#regionTable')._state = null;
    renderTable('#regionTable', cols, o.table, { sortKey: 'mean', onRow: r => this.addRegion(r.id) });
  },

  addRegion(id, reload = true) {
    if (this.sel.find(s => s.id === id)) return;
    if (this.sel.length >= 5) { toast('最多同时分析 5 个区域，请先移除一个'); return; }
    const used = new Set(this.sel.map(s => s.slot));
    let slot = 0; while (used.has(slot)) slot++;
    this.sel.push({ id, slot });
    this.renderChips();
    if (reload) { this.loadTemporal(); toast('已加入时间分析：' + regionLabel(id)); }
  },
  removeRegion(id) { this.sel = this.sel.filter(s => s.id !== id); this.renderChips(); if (this.sel.length) this.loadTemporal(); },
  renderChips() {
    $('#edaRegionChips').innerHTML = this.sel.map(s => `<span class="chip"><i style="background:${catColor(s.slot)}"></i>${escapeHtml(regionLabel(s.id))}<button data-id="${escapeHtml(s.id)}" title="移除">×</button></span>`).join('');
    $$('#edaRegionChips button').forEach(b => b.onclick = () => this.removeRegion(b.dataset.id));
  },

  async loadTemporal() {
    if (!this.sel.length) return;
    const q = encodeURIComponent(this.sel.map(s => s.id).join('|'));
    const d = await busy($('#chSeries').closest('.grid2'), () => api(`/api/eda/temporal/${State.ds}?regions=${q}&freq=${$('#edaFreq').value}`));
    if (d) this.renderTemporal(d);
  },

  renderTemporal(d) {
    const t = T();
    const ids = d.regions.map(r => r.id), labels = d.regions.map(r => r.label);
    const col = id => catColor((this.sel.find(s => s.id === id) || {}).slot);
    // series
    chart('chSeries').setOption(baseOpt({
      grid: { left: 56, right: 18, top: 30, bottom: 44 }, xAxis: timeAxis(), yAxis: valAxis(), dataZoom: zoom(),
      series: ids.map((id, k) => line(labels[k], d.series.t.map((x, i) => [x, d.series.v[id][i]]), { color: col(id), lineStyle: { width: 1.5 } })),
    }), true);
    // daily profile
    const hrs = [...Array(24).keys()].map(h => h + ':00');
    chart('chProfile').setOption(baseOpt({
      xAxis: catAxis(hrs, { boundaryGap: false }), yAxis: valAxis(),
      series: ids.flatMap((id, k) => [
        line(labels[k], d.profiles[id].workday, { color: col(id) }),
        line(labels[k], d.profiles[id].offday, { color: col(id), lineStyle: { width: 2, type: [4, 3] } }),
      ]),
    }), true);
    // heatmap
    const hv = d.heatmap.values, flat = hv.flat();
    chart('chHeat').setOption(baseOpt({
      legend: { show: false }, tooltip: { ...baseOpt().tooltip, trigger: 'item', formatter: p => `${d.heatmap.dow[p.value[1]]} ${p.value[0]}:00<br><b>${fmtFull(p.value[2], 0)}</b>` },
      grid: { left: 44, right: 64, top: 8, bottom: 28 },
      xAxis: catAxis([...Array(24).keys()], { splitArea: { show: false } }), yAxis: catAxis(d.heatmap.dow, { inverse: true }),
      visualMap: { min: Math.min(...flat), max: Math.max(...flat), calculable: false, orient: 'vertical', right: 0, top: 'middle', itemHeight: 120, itemWidth: 10, inRange: { color: t.seq }, textStyle: { color: t.muted, fontSize: 11 }, formatter: v => fmt(v) },
      series: [{ type: 'heatmap', data: hv.flatMap((row, di) => row.map((v, h) => [h, di, v])), itemStyle: { borderColor: t.surface, borderWidth: 1 } }],
    }), true);
    // box
    chart('chBox').setOption(baseOpt({
      legend: { show: false }, tooltip: { ...baseOpt().tooltip, trigger: 'item', formatter: p => `${p.name}<br>最小 ${fmt(p.value[1])}<br>Q1 ${fmt(p.value[2])}<br>中位 ${fmt(p.value[3])}<br>Q3 ${fmt(p.value[4])}<br>最大 ${fmt(p.value[5])}` },
      xAxis: catAxis(hrs), yAxis: valAxis(),
      series: [{ type: 'boxplot', data: d.box, itemStyle: { color: t.seq[1], borderColor: t.seq[4], borderWidth: 1.2 } }],
    }), true);
    // ACF
    const nl = d.acf[ids[0]].length;
    chart('chAcf').setOption(baseOpt({
      xAxis: { ...valAxis(), type: 'value', min: 0, max: (nl - 1) / 24, interval: 1, name: '滞后（天）', nameLocation: 'middle', nameGap: 22, axisLabel: { color: t.muted, fontSize: 12 } },
      yAxis: valAxis({ min: -0.5, max: 1 }),
      tooltip: { ...baseOpt().tooltip, valueFormatter: v => v.toFixed(3) },
      series: ids.map((id, k) => line(labels[k], d.acf[id].map((v, i) => [i / 24, v]), { color: col(id), lineStyle: { width: 1.5 } })),
    }), true);
    // decomposition
    const dc = d.decomp;
    if (dc && !dc.error) {
      $('#decompTitle').textContent = `区域 ${labels[0]} 的小时数据，分解为趋势、日季节、周季节与残差。`;
      const comps = [['观测值', dc.observed], ['趋势', dc.trend], ...Object.entries(dc.seasonal).map(([k, v]) => ['季节 ' + k.replace('周期 ', ''), v]), ['残差', dc.resid]];
      const n = comps.length, h = 100 / n;
      chart('chDecomp').setOption(baseOpt({
        legend: { show: false },
        axisPointer: { link: [{ xAxisIndex: 'all' }] },
        grid: comps.map((_, i) => ({ left: 64, right: 18, top: `${i * h + 3}%`, height: `${h - 7}%` })),
        xAxis: comps.map((_, i) => timeAxis({ gridIndex: i, axisLabel: { show: i === n - 1, color: t.muted, fontSize: 11 } })),
        yAxis: comps.map(([nm], i) => valAxis({ gridIndex: i, name: nm, nameLocation: 'middle', nameGap: 48, splitNumber: 2, scale: true })),
        series: comps.map(([nm, v], i) => line(nm, dc.t.map((x, j) => [x, v[j]]), { xAxisIndex: i, yAxisIndex: i, color: i === n - 1 ? t.muted : t.series[0], lineStyle: { width: 1.2 } })),
      }), true);
    }
    // daily totals
    const off = d.daily.offday;
    const areas = [];
    off.forEach((o, i) => { if (o) areas.push([{ xAxis: d.daily.t[i] }, { xAxis: d.daily.t[i] }]); });
    chart('chRegionDaily').setOption(baseOpt({
      grid: { left: 56, right: 18, top: 30, bottom: 28 },
      xAxis: catAxis(d.daily.t, { axisLabel: { color: t.muted, fontSize: 11, formatter: v => v.slice(5) } }), yAxis: valAxis(),
      tooltip: { ...baseOpt().tooltip, axisPointer: { type: 'shadow' } },
      series: ids.map((id, k) => ({ name: labels[k], type: 'bar', color: col(id), barGap: '10%', itemStyle: { borderRadius: [3, 3, 0, 0] },
        data: d.daily.v[id].map((v, i) => ({ value: v, itemStyle: { opacity: off[i] ? 1 : 0.45 } })) })),
    }), true);
  },

  /* ---------------- spatial ---------------- */
  renderSpatial() {
    const s = this.spatial, t = T();
    this.setupSlider(); this.paintMap();
    const hrs = [...Array(24).keys()].map(h => h + ':00');
    $('#moranNote').textContent = `平均流量的全局 I 为 ${s.moran.global_mean.toFixed(3)}，空间权重取 k = 6 近邻。`;
    chart('chMoranHour').setOption(baseOpt({
      legend: { show: false }, xAxis: catAxis(hrs, { boundaryGap: false }), yAxis: valAxis({ scale: true }),
      tooltip: { ...baseOpt().tooltip, valueFormatter: v => v.toFixed(3) },
      series: [line("Moran's I（工作日小时均值）", s.moran.by_hour, { showSymbol: true, symbolSize: 6 })],
    }), true);
    chart('chMoranDay').setOption(baseOpt({
      legend: { show: false }, xAxis: catAxis(s.moran.days, { boundaryGap: false, axisLabel: { color: t.muted, fontSize: 11, formatter: v => v.slice(5) } }), yAxis: valAxis({ scale: true }),
      tooltip: { ...baseOpt().tooltip, valueFormatter: v => v.toFixed(3) },
      series: [line("Moran's I（日总量）", s.moran.by_day, { showSymbol: true, symbolSize: 6 })],
    }), true);
    // LISA
    const lc = { HH: t.pos, LL: t.neg, HL: isDark() ? '#8C5038' : '#E3AD95', LH: isDark() ? '#3A6599' : '#93AECF', ns: t.other };
    const ln = { HH: '高-高 聚集', LL: '低-低 聚集', HL: '高-低 异常', LH: '低-高 异常', ns: '不显著' };
    const idx = Object.fromEntries(s.regions.map((r, i) => [r.id, i]));
    this.lisaMap.paint(rid => lc[s.lisa.cls[idx[rid]]], rid => {
      const i = idx[rid]; return `<b>${escapeHtml(s.regions[i].label)}</b><br>${ln[s.lisa.cls[i]]}<br>Ii = ${s.lisa.I[i]}，p = ${s.lisa.p[i]}`;
    });
    const counts = {}; s.lisa.cls.forEach(c => counts[c] = (counts[c] || 0) + 1);
    this.lisaMap.setLegend(catLegend('LISA 类型', Object.keys(lc).map(k => [lc[k], `${ln[k]}（${counts[k] || 0}）`])));
    // distance decay with binned mean
    const pts = s.distance_decay, maxD = Math.max(...pts.map(p => p[0]));
    const nb = 12, bins = Array.from({ length: nb }, () => [0, 0]);
    pts.forEach(([dd, c]) => { const b = Math.min(nb - 1, Math.floor(dd / maxD * nb)); bins[b][0] += c; bins[b][1]++; });
    chart('chDecay').setOption(baseOpt({
      grid: { left: 50, right: 18, top: 30, bottom: 40 },
      tooltip: { ...baseOpt().tooltip, trigger: 'item', formatter: p => `距离 ${p.value[0].toFixed(1)} km<br>相关 ${p.value[1].toFixed(3)}` },
      xAxis: valAxis({ name: '质心距离 (km)', nameLocation: 'middle', nameGap: 24, axisLabel: { color: t.muted, fontSize: 12 } }),
      yAxis: valAxis({ name: '相关系数', max: 1, scale: true, axisLabel: { color: t.muted, fontSize: 12 } }),
      series: [
        { name: '区域对', type: 'scatter', data: pts, symbolSize: 5, itemStyle: { color: t.series[0], opacity: 0.35 } },
        line('分箱均值', bins.map((b, i) => b[1] ? [(i + 0.5) * maxD / nb, b[0] / b[1]] : null).filter(Boolean), { color: t.series[1], showSymbol: true, symbolSize: 8, z: 5 }),
      ],
    }), true);
  },

  setupSlider() {
    const s = this.spatial; if (!s) return;
    const m = this.mode;
    $('#metricCtl').hidden = m !== 'metric';
    $('#hourCtl').hidden = m !== 'hour';
    $('#scaleCtl').hidden = m === 'metric';
    $('#sliderCtl').hidden = m === 'metric';
    const sl = $('#timeSlider');
    if (m === 'hour') { sl.max = 23; sl.value = 8; }
    if (m === 'day') { sl.max = s.daily.t.length - 1; sl.value = 0; }
  },

  paintMap() {
    const s = this.spatial; if (!s) return;
    const t = T(), idx = Object.fromEntries(s.regions.map((r, i) => [r.id, i]));
    let vals, title, f = fmt, scale;
    if (this.mode === 'metric') {
      const k = $('#mapMetric').value;
      vals = s.metrics[k]; title = $('#mapMetric').selectedOptions[0].text;
      if (k === 'peak_hour') { scale = linearScale(0, 23, t.seq); f = v => Math.round(v) + ':00'; }
      else { scale = quantileScale(vals, t.seq); if (k !== 'mean') f = v => v.toFixed(2); }
    } else {
      const i = +$('#timeSlider').value;
      let all;
      if (this.mode === 'hour') {
        const P = s.hour_profile[$('#hourType').value];
        vals = P[i]; all = P.flat();
        title = `${$('#hourType').selectedOptions[0].text} ${i}:00 平均流量`;
        $('#sliderLabel').textContent = `${String(i).padStart(2, '0')}:00`;
      } else {
        vals = s.daily.values[i]; all = s.daily.values.flat();
        title = `${s.daily.t[i]} 日总流量`;
        $('#sliderLabel').textContent = s.daily.t[i];
      }
      scale = quantileScale($('#hourScale').value === 'global' ? all : vals, t.seq);
      if ($('#hourScale').value === 'global') { scale.min = Math.min(...all); scale.max = Math.max(...all); }
    }
    this.map.paint(rid => scale(vals[idx[rid]]), rid => `<b>${escapeHtml(s.regions[idx[rid]].label)}</b><br>${title}: ${f(vals[idx[rid]])}<br><span style="opacity:.7">点击加入时间分析</span>`);
    this.map.setLegend(rampLegend(title + '（分位数色阶）', scale, f));
  },
  play() {
    $('#playBtn').textContent = '暂停';
    this.timer = setInterval(() => {
      const sl = $('#timeSlider'); sl.value = (+sl.value + 1) % (+sl.max + 1); this.paintMap();
    }, 700);
  },
  stop() { clearInterval(this.timer); this.timer = null; $('#playBtn').textContent = '播放'; },
  onShow() { this.map.invalidate(); this.lisaMap.invalidate(); },
};
