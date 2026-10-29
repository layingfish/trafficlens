/* Page 2 — clustering regions into functional zones. */
const Cluster = {
  cfg: null, map: null, res: null,

  async init() {
    this.map = new GeoMap('mapCluster');
    this.cfg = await api('/api/cluster/methods');
    $('#clFeature').innerHTML = Object.entries(this.cfg.features).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
    $('#clNorm').innerHTML = Object.entries(this.cfg.norms).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
    $('#clMethod').innerHTML = Object.entries(this.cfg.methods).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
    $('#clMethod').onchange = () => this.renderParams();
    $('#clFeature').onchange = () => { $('#clNorm').disabled = $('#clFeature').value === 'stats'; };
    $('#clRun').onclick = () => this.run();
    $('#clElbow').onclick = () => this.elbow();
    this.renderParams();
  },

  renderParams() {
    const m = this.cfg.methods[$('#clMethod').value];
    $('#clParams').innerHTML = m.params.map(p => p.type === 'select'
      ? `<label>${p.label}<select data-p="${p.name}">${p.options.map(o => `<option ${o === p.default ? 'selected' : ''}>${o}</option>`).join('')}</select></label>`
      : `<label>${p.label}<input data-p="${p.name}" type="number" value="${p.default}" min="${p.min}" max="${p.max}" step="${p.step || 1}" style="width:90px"></label>`).join('');
  },

  params() {
    const o = {};
    $$('#clParams [data-p]').forEach(el => o[el.dataset.p] = el.tagName === 'SELECT' ? el.value : +el.value);
    return o;
  },

  async load() {
    this.map.setGeo(State.ds, await getGeo(State.ds));
    if (!this.res || this.res.dataset !== State.ds) await this.run();
  },

  async run() {
    const body = { dataset: State.ds, method: $('#clMethod').value, feature: $('#clFeature').value, norm: $('#clNorm').value, params: this.params(), embed: $('#clEmbed').value };
    const r = await busy('#page-cluster', () => api('/api/cluster/run', { body }));
    if (!r) return;
    r.dataset = State.ds; r.body = body;
    this.res = r; State.cluster = r;
    this.render();
    Dataset.refreshClusterOptions();
  },

  render() {
    const r = this.res, t = T();
    const ks = Object.keys(r.centers).map(Number).filter(k => k >= 0).sort((a, b) => a - b);
    const zoneOf = Object.fromEntries(r.zones.map(z => [z.cluster, z]));
    const name = k => k < 0 ? '噪声点' : `簇 ${k + 1}：${zoneOf[k] ? zoneOf[k].zone : ''}`;
    const sc = r.scores;
    $('#clScores').innerHTML = [
      kpi('簇数', ks.length, r.n_noise ? `噪声点 ${r.n_noise} 个` : ($('#clMethod').selectedOptions[0] || {}).textContent || r.body.method),
      kpi('轮廓系数 Silhouette', sc.silhouette == null ? '-' : sc.silhouette.toFixed(3), '取值 -1 至 1，越高越好'),
      kpi('Calinski-Harabasz', sc.calinski_harabasz == null ? '-' : sc.calinski_harabasz.toFixed(1), '越高越好'),
      kpi('Davies-Bouldin', sc.davies_bouldin == null ? '-' : sc.davies_bouldin.toFixed(3), '越低越好'),
    ].join('');
    if (ks.length > 8) toast('簇数超过 8，第 9 个及以后的簇以灰色显示', true);
    // map
    const lab = Object.fromEntries(r.regions.map((x, i) => [x.id, r.labels[i]]));
    this.map.paint(rid => catColor(lab[rid]), rid => `<b>${escapeHtml(regionLabel(rid))}</b><br>${name(lab[rid])}`);
    const sizes = {}; r.labels.forEach(l => sizes[l] = (sizes[l] || 0) + 1);
    const legendKeys = [...ks, ...(r.n_noise ? [-1] : [])];
    this.map.setLegend(catLegend('功能区', legendKeys.map(k => [catColor(k), `${name(k)}（${sizes[k] || 0}）`])));
    // centers
    const hrs = [...Array(24).keys()].map(h => h + ':00');
    const centerChart = (id, key) => chart(id).setOption(baseOpt({
      xAxis: catAxis(hrs, { boundaryGap: false }), yAxis: valAxis({ scale: true, axisLabel: { color: t.muted, fontSize: 12, formatter: v => v.toFixed(1) } }),
      tooltip: { ...baseOpt().tooltip, valueFormatter: v => v.toFixed(3) },
      legend: { ...baseOpt().legend, type: 'scroll' },
      series: ks.map(k => line(`簇 ${k + 1}`, r.centers[k][key], { color: catColor(k) })),
    }), true);
    centerChart('chCenterWd', 'workday');
    centerChart('chCenterWe', 'offday');
    const dows = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
    chart('chCenterWk').setOption(baseOpt({
      xAxis: catAxis([...Array(168).keys()], { boundaryGap: false, axisLabel: { color: t.muted, fontSize: 12, interval: 23, formatter: v => dows[Math.floor(v / 24)] } }),
      yAxis: valAxis({ scale: true, axisLabel: { color: t.muted, fontSize: 12, formatter: v => v.toFixed(1) } }),
      tooltip: { ...baseOpt().tooltip, valueFormatter: v => v.toFixed(3) },
      axisPointer: { label: { formatter: p => `${dows[Math.floor(p.value / 24)]} ${p.value % 24}:00` } },
      series: ks.map(k => line(`簇 ${k + 1}`, r.centers[k].weekly, { color: catColor(k), lineStyle: { width: 1.6 } })),
    }), true);
    // embedding
    $('#embedNote').textContent = r.explained_variance ? `PCA 前两个主成分分别解释 ${(r.explained_variance[0] * 100).toFixed(1)}% 与 ${(r.explained_variance[1] * 100).toFixed(1)}% 的方差。` : 't-SNE 嵌入，点间距离只反映局部邻近关系。';
    chart('chEmbed').setOption(baseOpt({
      grid: { left: 40, right: 16, top: 30, bottom: 30 },
      tooltip: { ...baseOpt().tooltip, trigger: 'item', formatter: p => `<b>${escapeHtml(p.data.label)}</b><br>${name(p.data.k)}` },
      xAxis: valAxis({ scale: true, axisLabel: { show: false } }), yAxis: valAxis({ scale: true, axisLabel: { show: false } }),
      legend: { ...baseOpt().legend, type: 'scroll', icon: 'circle', itemWidth: 8, itemHeight: 8 },
      series: legendKeys.map(k => ({
        name: k < 0 ? '噪声点' : `簇 ${k + 1}`, type: 'scatter', symbolSize: 9, itemStyle: { color: catColor(k), borderColor: t.surface, borderWidth: 1.5 },
        data: r.embedding.map((e, i) => ({ value: e, label: r.regions[i].label, k: r.labels[i] })).filter(d => d.k === k),
      })),
    }), true);
    // zone table
    const cols = [
      { key: 'cluster', label: '簇', render: (v) => `<span class="swatch" style="background:${catColor(v)}"></span>${v < 0 ? '噪声' : v + 1}` },
      { key: 'zone', label: '推断功能区', cls: 'l', render: (v, z) => `<span class="zone-name" title="${escapeHtml(z.reason)}">${v}</span>` },
      { key: 'size', label: '区域数' },
      { key: 'work', label: '工作时段指数', render: v => v.toFixed(2) },
      { key: 'evening', label: '晚间指数', render: v => v.toFixed(2) },
      { key: 'night', label: '夜间指数', render: v => v.toFixed(2) },
      { key: 'offday_ratio', label: '休/工比', render: v => v.toFixed(2) },
      { key: 'peak_hour_workday', label: '工作日峰', render: v => v + ':00' },
      { key: 'peak_hour_offday', label: '休息日峰', render: v => v + ':00' },
      { key: 'level', label: '量级 log', render: v => v.toFixed(2) },
    ];
    $('#zoneTable')._state = null;
    renderTable('#zoneTable', cols, r.zones, { onRow: z => toast(`${z.zone}：${z.reason}`) });
  },

  async elbow() {
    const body = { dataset: State.ds, feature: $('#clFeature').value, norm: $('#clNorm').value, kmax: 10 };
    const e = await busy('#page-cluster', () => api('/api/cluster/elbow', { body }));
    if (!e) return;
    $('#elbowCard').hidden = false;
    const best = e.k[e.silhouette.indexOf(Math.max(...e.silhouette))];
    const mk = (id, name, data, color) => chart(id).setOption(baseOpt({
      legend: { show: false }, title: { text: name, left: 0, top: 0, textStyle: { fontSize: 12, color: T().text2, fontWeight: 500 } },
      xAxis: catAxis(e.k, { name: 'k', nameLocation: 'middle', nameGap: 22 }), yAxis: valAxis({ scale: true }),
      tooltip: { ...baseOpt().tooltip, valueFormatter: v => v.toFixed(3) },
      series: [line(name, data, { color, showSymbol: true, symbolSize: 8 })],
    }), true);
    mk('chElbow', '惯性 Inertia', e.inertia, T().series[0]);
    mk('chSil', `轮廓系数（最佳 k = ${best}）`, e.silhouette, T().series[1]);
    resizeVisible();
    $('#elbowCard').scrollIntoView({ behavior: 'smooth', block: 'center' });
  },
  onShow() { this.map.invalidate(); },
};
