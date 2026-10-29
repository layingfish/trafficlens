/* Page 3 — forecasting dataset construction (input/output window lengths, splits). */
const Dataset = {
  init() {
    $('#dsRegionMode').onchange = () => this.toggleRegionMode();
    ['dsL', 'dsH', 'dsFreq'].forEach(id => $('#' + id).addEventListener('input', () => this.hint()));
    $('#dsForm').onsubmit = (e) => { e.preventDefault(); this.build(); };
  },

  load() {
    const m = State.meta;
    $('#dsFreq').innerHTML = m.freq_options.map(f => `<option value="${f}" ${f === '1h' ? 'selected' : ''}>${f === 'native' ? '原始 ' + m.freq : f}</option>`).join('');
    $('#dsManualSel').innerHTML = m.regions.map(r => `<option value="${escapeHtml(r.id)}">${escapeHtml(r.label)}</option>`).join('');
    this.refreshClusterOptions();
    this.toggleRegionMode();
    this.hint();
    this.refreshList();
  },

  toggleRegionMode() {
    const m = $('#dsRegionMode').value;
    $('#dsTopN').hidden = m !== 'top';
    $('#dsClusterSel').hidden = m !== 'cluster';
    $('#dsManual').hidden = m !== 'manual';
  },

  refreshClusterOptions() {
    const c = State.cluster;
    if (!c || c.dataset !== State.ds) { $('#dsCluster').innerHTML = '<option value="">（请先在「功能区聚类」运行聚类）</option>'; return; }
    const sizes = {}; c.labels.forEach(l => sizes[l] = (sizes[l] || 0) + 1);
    $('#dsCluster').innerHTML = c.zones.filter(z => z.cluster >= 0).map(z => `<option value="${z.cluster}">簇 ${z.cluster + 1}：${z.zone}（${sizes[z.cluster]} 个区域）</option>`).join('');
  },

  stepMinutes() {
    const f = $('#dsFreq').value === 'native' ? State.meta.freq : $('#dsFreq').value;
    const m = f.match(/(\d+)(min|h|D)/);
    return m ? +m[1] * ({ min: 1, h: 60, D: 1440 })[m[2]] : 60;
  },

  hint() {
    const sm = this.stepMinutes();
    const dur = n => { const mins = n * sm; return mins >= 1440 ? `${(mins / 1440).toFixed(mins % 1440 ? 1 : 0)} 天` : mins >= 60 ? `${(mins / 60).toFixed(mins % 60 ? 1 : 0)} 小时` : `${mins} 分钟`; };
    $('#dsLenHint').textContent = `当前粒度每步 ${sm >= 60 ? sm / 60 + ' 小时' : sm + ' 分钟'}：输入 L = ${$('#dsL').value} 步 ≈ ${dur(+$('#dsL').value)}，预测 H = ${$('#dsH').value} 步 ≈ ${dur(+$('#dsH').value)}`;
    const perDay = Math.round(1440 / sm);
    const presets = [
      ['过去 6 步 → 未来 1 步', 6, 1], ['过去 12 步 → 未来 3 步', 12, 3],
      [`过去 1 天 → 未来 ${Math.max(1, Math.round(perDay / 4))} 步`, perDay, Math.max(1, Math.round(perDay / 4))],
      ['过去 1 天 → 未来 1 天', perDay, perDay], ['过去 1 周 → 未来 1 天', perDay * 7, perDay],
    ];
    $('#dsPresets').innerHTML = presets.map(([n, l, h]) => `<button type="button" class="btn" data-l="${l}" data-h="${h}">${n}（${l}→${h}）</button>`).join('');
    $$('#dsPresets button').forEach(b => b.onclick = () => { $('#dsL').value = b.dataset.l; $('#dsH').value = b.dataset.h; this.hint(); });
  },

  regions() {
    const mode = $('#dsRegionMode').value;
    if (mode === 'all') return [];
    if (mode === 'top') {
      const n = +$('#dsTopNVal').value;
      return (EDA.overview ? EDA.overview.table : []).slice(0, n).map(r => r.id);
    }
    if (mode === 'cluster') {
      const c = State.cluster, k = +$('#dsCluster').value;
      if (!c || c.dataset !== State.ds || $('#dsCluster').value === '') throw new Error('请先运行聚类');
      return c.regions.filter((_, i) => c.labels[i] === k).map(r => r.id);
    }
    const ids = $$('#dsManualSel option').filter(o => o.selected).map(o => o.value);
    if (!ids.length) throw new Error('请至少选择一个区域');
    return ids;
  },

  async build() {
    let regions;
    try { regions = this.regions(); } catch (e) { toast(e.message, true); return; }
    const body = {
      dataset: State.ds, regions, freq: $('#dsFreq').value, input_len: +$('#dsL').value, output_len: +$('#dsH').value,
      train_ratio: +$('#dsTrain').value, val_ratio: +$('#dsVal').value, train_stride: +$('#dsTrainStride').value,
      eval_stride: +$('#dsEvalStride').value, scaler: $('#dsScaler').value, log1p: $('#dsLog').checked, name: $('#dsName').value.trim(),
    };
    const s = await busy($('#dsBuild').closest('.card'), () => api('/api/windows', { body }));
    if (!s) return;
    State.activeWindow = s.id;
    this.show(s);
    await this.refreshList();
    toast('数据集已构建：' + s.config.name);
  },

  show(s) {
    const t = T(), sp = s.splits, sh = s.shapes;
    $('#dsResultName').textContent = s.config.name;
    const row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
    $('#dsSummary').className = '';
    $('#dsSummary').innerHTML = `<table class="summary-table"><tbody>
      ${row('数据集 ID', `<code>${s.id}</code>`)}
      ${row('区域数 N / 时间步 T', `${s.n_regions} / ${s.n_steps.toLocaleString()}（每天 ${s.steps_per_day} 步）`)}
      ${row('输入 L → 输出 H', `${s.config.input_len} → ${s.config.output_len} 步`)}
      ${row('训练集', `${sp.train.windows.toLocaleString()} 个窗口，${fmtTime(sp.train.from)} 至 ${fmtTime(sp.train.to)}`)}
      ${row('验证集', `${sp.val.windows.toLocaleString()} 个窗口，${fmtTime(sp.val.from)} 至 ${fmtTime(sp.val.to)}`)}
      ${row('测试集', `${sp.test.windows.toLocaleString()} 个窗口，${fmtTime(sp.test.from)} 至 ${fmtTime(sp.test.to)}`)}
      ${row('X_train / Y_train', `[${sh.X_train.join(', ')}] / [${sh.Y_train.join(', ')}]`)}
      ${row('X_test / Y_test', `[${sh.X_test.join(', ')}] / [${sh.Y_test.join(', ')}]`)}
      ${row('单变量池化训练样本', `${sh.pooled_samples_train.toLocaleString()}（窗口 × 区域，供 ML/DL 共享权重模型）`)}
      ${row('归一化', `${s.config.scaler}${s.config.log1p ? ' + log1p' : ''}（仅用训练段拟合）`)}
    </tbody></table>
    <div class="row gap" style="margin-top:12px"><button class="btn primary" id="dsGoTrain">用此数据集训练</button></div>`;
    $('#dsGoTrain').onclick = () => { State.activeWindow = s.id; Main.go('train'); };
    const p = s.preview;
    chart('chSplit').setOption(baseOpt({
      legend: { show: false }, grid: { left: 56, right: 18, top: 16, bottom: 44 }, xAxis: timeAxis(), yAxis: valAxis(), dataZoom: zoom(),
      series: [line('合计流量', p.t.map((x, i) => [x, p.v[i]]), {
        color: t.text2, lineStyle: { width: 1.2 },
        markArea: { silent: true, data: [
          [{ xAxis: p.t[0], itemStyle: { color: t.series[0], opacity: 0.12 }, label: { show: true, position: 'insideTop', color: t.text2, formatter: '训练' } }, { xAxis: p.train_end }],
          [{ xAxis: p.train_end, itemStyle: { color: t.series[1], opacity: 0.16 }, label: { show: true, position: 'insideTop', color: t.text2, formatter: '验证' } }, { xAxis: p.val_end }],
          [{ xAxis: p.val_end, itemStyle: { color: t.series[2], opacity: 0.14 }, label: { show: true, position: 'insideTop', color: t.text2, formatter: '测试' } }, { xAxis: p.t[p.t.length - 1] }],
        ] },
      })],
    }), true);
    const ex = s.example;
    chart('chWindow').setOption(baseOpt({
      xAxis: timeAxis(), yAxis: valAxis({ scale: true }),
      series: [
        line(`输入 L=${ex.t_in.length}`, ex.t_in.map((x, i) => [x, ex.v_in[i]]), { color: t.series[0], showSymbol: true, symbolSize: 6 }),
        line(`目标 H=${ex.t_out.length}`, [[ex.t_in[ex.t_in.length - 1], ex.v_in[ex.v_in.length - 1]], ...ex.t_out.map((x, i) => [x, ex.v_out[i]])], { color: t.series[1], showSymbol: true, symbolSize: 6, lineStyle: { width: 2, type: [4, 3] } }),
      ],
    }), true);
  },

  async refreshList() {
    State.windows = await api('/api/windows');
    const rows = State.windows.map(w => ({ ...w, name: w.config.name, ds: w.config.dataset, L: w.config.input_len, H: w.config.output_len, freq: w.config.freq, train: w.splits.train.windows, test: w.splits.test.windows }));
    if (!rows.length) { $('#dsList').innerHTML = '<div class="empty">暂无</div>'; }
    else renderTable('#dsList', [
      { key: 'name', label: '名称', cls: 'l' }, { key: 'id', label: 'ID', cls: 'l', render: v => `<code>${v}</code>` },
      { key: 'n_regions', label: '区域' }, { key: 'freq', label: '粒度' }, { key: 'L', label: 'L' }, { key: 'H', label: 'H' },
      { key: 'train', label: '训练窗口' }, { key: 'test', label: '测试窗口' },
    ], rows, { onRow: async r => { State.activeWindow = r.id; this.show(await api('/api/windows/' + r.id)); this.refreshList(); }, selected: r => r.id === State.activeWindow });
    Train.refreshData();
  },
};
