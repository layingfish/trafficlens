/* Page 5 — prediction display & evaluation. */
const METRICS = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'WAPE', 'R2'];
const HIGHER_BETTER = { R2: true };
const mfmt = (k, v) => v == null ? '-' : (k === 'R2' ? v.toFixed(4) : ['MAPE', 'sMAPE', 'WAPE'].includes(k) ? v.toFixed(2) + '%' : fmt(v));

const Eval = {
  cmp: [], cur: null, ov: null, map: null, sampleT: null,

  init() {
    this.map = new GeoMap('mapEval');
    $('#evData').onchange = () => { this.cmp = []; this.refresh(); };
    $('#evSort').onchange = () => this.refresh();
    $('#cmpMetric').onchange = () => this.compare();
    $('#evRegion').onchange = () => { this.series(); this.sample(); };
    $('#evH').oninput = () => { $('#evHLabel').textContent = `h = ${$('#evH').value}`; clearTimeout(this._h); this._h = setTimeout(() => this.series(), 120); };
    $('#evSample').oninput = () => { clearTimeout(this.sampleT); this.sampleT = setTimeout(() => this.sample(), 80); };
    $('#evHMetric').onchange = () => this.horizonChart();
    $('#evRMetric').onchange = () => this.regionCharts();
  },

  async refresh(full = true) {
    const jobs = (Train.jobs.length ? Train.jobs : await api('/api/jobs')).filter(j => j.metrics);
    $('#evalEmpty').hidden = jobs.length > 0;
    if (!jobs.length) { $('#leaderboard').innerHTML = ''; return; }
    const groups = {};
    jobs.forEach(j => (groups[j.data_id] = groups[j.data_id] || { name: j.data_name, n: 0 }).n++);
    const prev = $('#evData').value;
    const sameDs = jobs.find(j => (j.data_config || {}).dataset === State.ds) || jobs[0];
    const pick = groups[prev] ? prev : (groups[State.activeWindow] ? State.activeWindow : sameDs.data_id);
    $('#evData').innerHTML = Object.entries(groups).map(([id, g]) => `<option value="${id}" ${id === pick ? 'selected' : ''}>${escapeHtml(g.name)}（${g.n} 个模型）</option>`).join('');
    const rows = jobs.filter(j => j.data_id === pick).map(j => ({ ...j, ...j.metrics, valMAE: j.val_metrics ? j.val_metrics.MAE : null }));
    const sk = $('#evSort').value;
    const best = {};
    METRICS.forEach(k => { const v = rows.map(r => r[k]).filter(x => x != null); best[k] = HIGHER_BETTER[k] ? Math.max(...v) : Math.min(...v); });
    rows.sort((a, b) => (a[sk] == null) - (b[sk] == null) || (HIGHER_BETTER[sk] ? b[sk] - a[sk] : a[sk] - b[sk]));
    const el = $('#leaderboard');
    el.innerHTML = `<table><thead><tr><th class="l">对比</th><th class="l">#</th><th class="l">模型</th><th class="l">类别</th>${METRICS.map(k => `<th>${k}</th>`).join('')}<th>验证 MAE</th><th>训练耗时</th><th></th></tr></thead><tbody>
      ${rows.map((r, i) => `<tr data-id="${r.id}" class="${this.cur === r.id ? 'sel' : ''}">
        <td class="l"><input type="checkbox" data-c="${r.id}" ${this.cmp.includes(r.id) ? 'checked' : ''}></td><td class="l">${i + 1}</td>
        <td class="l"><b>${r.model_name}</b>${r.status === 'stopped' ? '<span class="flag">提前停止</span>' : ''}</td><td class="l">${r.category}</td>
        ${METRICS.map(k => `<td class="${r[k] === best[k] ? 'best' : ''}">${mfmt(k, r[k])}</td>`).join('')}
        <td>${mfmt('MAE', r.valMAE)}</td><td>${(r.train_time || 0).toFixed(1)}s</td>
        <td><button class="btn ghost small rw" data-del="${r.id}" title="删除该结果">✕</button></td></tr>`).join('')}</tbody></table>`;
    $$('tbody tr', el).forEach(tr => tr.onclick = (e) => {
      if (e.target.dataset.c || e.target.dataset.del) return;
      this.open(tr.dataset.id);
    });
    $$('[data-c]', el).forEach(cb => cb.onchange = () => {
      if (cb.checked) { if (this.cmp.length >= 8) { cb.checked = false; toast('最多同时对比 8 个模型'); return; } this.cmp.push(cb.dataset.c); }
      else this.cmp = this.cmp.filter(x => x !== cb.dataset.c);
      this.compare();
    });
    $$('[data-del]', el).forEach(b => b.onclick = async () => {
      if (!confirm('删除该训练结果？')) return;
      await api('/api/jobs/' + b.dataset.del, { method: 'DELETE' });
      this.cmp = this.cmp.filter(x => x !== b.dataset.del);
      if (this.cur === b.dataset.del) { this.cur = null; $('#evalDetail').hidden = true; }
      await Train.refreshJobs(); this.refresh();
    });
    if (full) {
      if (!this.cmp.length) this.cmp = rows.slice(0, Math.min(4, rows.length)).map(r => r.id);
      $$('[data-c]', el).forEach(cb => cb.checked = this.cmp.includes(cb.dataset.c));
      this.compare();
      if (!this.cur || !rows.find(r => r.id === this.cur)) this.open(rows[0].id);
    }
  },

  async compare() {
    $('#compareBlock').hidden = this.cmp.length < 1;
    if (!this.cmp.length) return;
    const d = await api('/api/eval/compare?ids=' + this.cmp.join(','));
    const order = this.cmp.filter(id => d.find(x => x.id === id));
    const col = id => catColor(order.indexOf(id));
    const k = $('#cmpMetric').value, t = T();
    const rows = [...d].sort((a, b) => HIGHER_BETTER[k] ? a.overall[k] - b.overall[k] : b.overall[k] - a.overall[k]);
    chart('chCmpBar').setOption(baseOpt({
      legend: { show: false }, grid: { left: 150, right: 60, top: 10, bottom: 24 },
      tooltip: { ...baseOpt().tooltip, axisPointer: { type: 'shadow' }, valueFormatter: v => mfmt(k, v) },
      xAxis: valAxis(), yAxis: catAxis(rows.map(r => r.model_name), { axisLabel: { color: t.text2, fontSize: 12 } }),
      series: [{ type: 'bar', name: k, barWidth: '55%', data: rows.map(r => ({ value: r.overall[k], itemStyle: { color: col(r.id), borderRadius: [0, 4, 4, 0] } })),
        label: { show: true, position: 'right', color: t.text2, fontSize: 12, formatter: p => mfmt(k, p.value) } }],
    }), true);
    const H = Math.max(...d.map(r => r.per_horizon[k].length));
    $('#cmpHorizonNote').textContent = `指标 ${k}，步长 1 至 ${H}`;
    chart('chCmpH').setOption(baseOpt({
      legend: { ...baseOpt().legend, type: 'scroll' },
      tooltip: { ...baseOpt().tooltip, valueFormatter: v => mfmt(k, v) },
      xAxis: catAxis([...Array(H).keys()].map(i => i + 1), { name: '预测步长 h', nameLocation: 'middle', nameGap: 22 }), yAxis: valAxis({ scale: true }),
      series: d.map(r => line(r.model_name, r.per_horizon[k], { color: col(r.id), showSymbol: H <= 24 })),
    }), true);
    resizeVisible();
  },

  async open(id) {
    this.cur = id;
    $$('#leaderboard tbody tr').forEach(tr => tr.classList.toggle('sel', tr.dataset.id === id));
    const ov = await busy('#page-eval', () => api(`/api/eval/${id}/overview`));
    if (!ov) return;
    this.ov = ov;
    $('#evalDetail').hidden = false;
    const j = ov.job, m = ov.metrics.overall, v = j.val_metrics || {};
    $('#evTitle').textContent = j.model_name;
    $('#evSub').textContent = `数据集 ${j.data_name}，测试集共 ${ov.n_test} 个窗口。`;
    $('#evKpis').innerHTML = METRICS.map(k => kpi(k === 'R2' ? 'R²' : k, mfmt(k, m[k]), `验证集 ${mfmt(k, v[k])}`)).join('')
      + kpi('训练耗时', `${(j.train_time || 0).toFixed(1)} 秒`, `推理 ${((j.infer_time || 0) * 1000).toFixed(0)} 毫秒`);
    $('#evRegion').innerHTML = `<option value="-1">全部区域合计</option>` + ov.regions.map((r, i) => `<option value="${i}">${escapeHtml(r.label)}</option>`).join('');
    $('#evH').max = ov.H; $('#evH').value = 1; $('#evHLabel').textContent = 'h = 1';
    $('#evSample').max = ov.n_test - 1; $('#evSample').value = Math.floor(ov.n_test / 2);
    $('#evExportR').href = STATIC ? `api/eval/${id}/regions.csv` : `/api/eval/${id}/export?kind=regions`;
    $('#evExportP').href = `/api/eval/${id}/export?kind=predictions`;
    $('#evExportP').hidden = STATIC;  // the full prediction table is tens of MB, too large for the static copy
    this.series(); this.sample(); this.scatter(); this.horizonChart();
    if (State.geo[ov.dataset] || ov.dataset) this.map.setGeo(ov.dataset, await getGeo(ov.dataset));
    this.regionCharts();
    resizeVisible(); this.map.invalidate();
  },

  async series() {
    const t = T(), h = +$('#evH').value, r = +$('#evRegion').value;
    const d = await api(`/api/eval/${this.cur}/series?region=${r}&h=${h}`);
    $('#evSeriesNote').textContent = `第 ${h} 步预测，该序列 MAE ${fmt(d.metrics.MAE)}，R² ${d.metrics.R2 == null ? '-' : d.metrics.R2.toFixed(3)}`;
    chart('chEvSeries').setOption(baseOpt({
      grid: { left: 56, right: 18, top: 30, bottom: 44 }, xAxis: timeAxis(), yAxis: valAxis({ scale: true }), dataZoom: zoom(),
      series: [
        line('真实值', d.t.map((x, i) => [x, d.y[i]]), { color: t.text2, lineStyle: { width: 1.6 } }),
        line('预测值', d.t.map((x, i) => [x, d.p[i]]), { color: t.series[1], lineStyle: { width: 2 } }),
      ],
    }), true);
  },

  async sample() {
    const t = T(), i = +$('#evSample').value, r = +$('#evRegion').value;
    const d = await api(`/api/eval/${this.cur}/sample?i=${i}&region=${r}`);
    $('#evSampleLabel').textContent = `样本 ${d.i}，起点 ${fmtTime(d.t_in[d.t_in.length - 1])}`;
    const last = [d.t_in[d.t_in.length - 1], d.x[d.x.length - 1]];
    chart('chEvSample').setOption(baseOpt({
      xAxis: timeAxis(), yAxis: valAxis({ scale: true }),
      series: [
        line('历史输入', d.t_in.map((x, k) => [x, d.x[k]]), { color: t.series[0], showSymbol: d.x.length <= 48, symbolSize: 5 }),
        line('真实值', [last, ...d.t_out.map((x, k) => [x, d.y[k]])], { color: t.text2, showSymbol: true, symbolSize: 6 }),
        line('预测值', [last, ...d.t_out.map((x, k) => [x, d.p[k]])], { color: t.series[1], showSymbol: true, symbolSize: 6, lineStyle: { width: 2, type: [5, 3] } }),
      ],
    }), true);
  },

  scatter() {
    const t = T(), ov = this.ov;
    const vals = ov.scatter.flat(), lo = Math.min(...vals), hi = Math.max(...vals);
    chart('chEvScatter').setOption(baseOpt({
      legend: { show: false }, grid: { left: 56, right: 18, top: 16, bottom: 40 },
      tooltip: { ...baseOpt().tooltip, trigger: 'item', formatter: p => p.seriesType === 'scatter' ? `真实 ${fmtFull(p.value[0])}<br>预测 ${fmtFull(p.value[1])}` : '' },
      xAxis: valAxis({ name: '真实值', nameLocation: 'middle', nameGap: 24, min: lo, max: hi }),
      yAxis: valAxis({ name: '预测值', min: lo, max: hi }),
      series: [
        { type: 'scatter', data: ov.scatter, symbolSize: 4, itemStyle: { color: t.series[0], opacity: 0.35 }, large: true },
        { type: 'line', data: [[lo, lo], [hi, hi]], showSymbol: false, silent: true, lineStyle: { color: t.muted, type: 'dashed', width: 1.5 } },
      ],
    }), true);
    const hst = ov.residual_hist, mids = hst.counts.map((_, i) => (hst.edges[i] + hst.edges[i + 1]) / 2);
    chart('chEvResid').setOption(baseOpt({
      legend: { show: false }, grid: { left: 56, right: 18, top: 16, bottom: 40 },
      tooltip: { ...baseOpt().tooltip, axisPointer: { type: 'shadow' }, formatter: ps => `残差 ≈ ${fmt(mids[ps[0].dataIndex])}<br>频数 ${ps[0].value}` },
      xAxis: catAxis(mids.map(x => fmt(x)), { name: '预测 − 真实', nameLocation: 'middle', nameGap: 24 }), yAxis: valAxis(),
      series: [{ type: 'bar', barCategoryGap: '8%', data: hst.counts.map((c, i) => ({ value: c, itemStyle: { color: mids[i] < 0 ? t.neg : t.pos, borderRadius: [2, 2, 0, 0] } })) }],
    }), true);
  },

  horizonChart() {
    const k = $('#evHMetric').value, ph = this.ov.metrics.per_horizon[k], t = T();
    chart('chEvH').setOption(baseOpt({
      legend: { show: false }, grid: { left: 56, right: 18, top: 16, bottom: 40 },
      tooltip: { ...baseOpt().tooltip, axisPointer: { type: 'shadow' }, valueFormatter: v => mfmt(k, v) },
      xAxis: catAxis(ph.map((_, i) => i + 1), { name: '预测步长 h', nameLocation: 'middle', nameGap: 24 }), yAxis: valAxis({ scale: k === 'R2' }),
      series: [{ type: 'bar', name: k, data: ph, barCategoryGap: '25%', itemStyle: { color: t.series[0], borderRadius: [4, 4, 0, 0] } }],
    }), true);
  },

  regionCharts() {
    const k = $('#evRMetric').value, ov = this.ov, t = T();
    const vals = ov.metrics.per_region[k];
    const items = ov.regions.map((r, i) => ({ ...r, i, v: vals[i] })).filter(x => x.v != null);
    const worst = [...items].sort((a, b) => HIGHER_BETTER[k] ? a.v - b.v : b.v - a.v).slice(0, 15).reverse();
    chart('chEvRegionBar').setOption(baseOpt({
      legend: { show: false }, grid: { left: 130, right: 56, top: 8, bottom: 24 },
      tooltip: { ...baseOpt().tooltip, axisPointer: { type: 'shadow' }, valueFormatter: v => mfmt(k, v) },
      xAxis: valAxis(), yAxis: catAxis(worst.map(x => x.label), { axisLabel: { color: t.text2, fontSize: 12, width: 120, overflow: 'truncate' } }),
      series: [{ type: 'bar', data: worst.map(x => x.v), barWidth: '60%', itemStyle: { color: t.series[1], borderRadius: [0, 4, 4, 0] },
        label: { show: true, position: 'right', color: t.text2, fontSize: 11, formatter: p => mfmt(k, p.value) } }],
    }), true);
    const idx = Object.fromEntries(ov.regions.map((r, i) => [r.id, i]));
    const ramp = HIGHER_BETTER[k] ? [...t.seq].reverse() : t.seq;
    const sc = quantileScale(vals, ramp);
    $('#evMapNote').textContent = `指标 ${k}，${HIGHER_BETTER[k] ? '颜色越深表示越差' : '颜色越深表示误差越大'}；未参与预测的区域显示为灰色。`;
    this.map.paint(rid => idx[rid] === undefined ? t.other : sc(vals[idx[rid]]),
      rid => idx[rid] === undefined ? `${escapeHtml(regionLabel(rid))}<br>未参与预测` : `<b>${escapeHtml(ov.regions[idx[rid]].label)}</b><br>${k}: ${mfmt(k, vals[idx[rid]])}`);
    this.map.setLegend(rampLegend(`分区域 ${k}`, sc, v => mfmt(k, v)));
  },
  onShow() { this.refresh(); this.map.invalidate(); },
};
