/* Page 4 — algorithm panel: parameter editing, job queue, live training monitor (SSE). */
const CAT_ORDER = ['统计模型', '机器学习', '深度学习', '图神经网络'];

const Train = {
  checked: new Set(), focus: null, values: {}, jobs: [], selJob: null, es: null, curve: [], pollT: null, tick: null,

  async init() {
    State.models = await api('/api/models');
    this.focus = 'lstm';
    ['snaive', 'ridge', 'lightgbm', 'lstm'].forEach(k => this.checked.add(k));
    this.renderList(); this.renderParams();
    $('#trSubmit').onclick = () => this.submit();
    $('#trAll').onclick = () => { State.models.forEach(m => this.checked.add(m.key)); this.renderList(); };
    $('#trNone').onclick = () => { this.checked.clear(); this.renderList(); };
    $('#paramReset').onclick = () => { delete this.values[this.focus]; this.renderParams(); this.renderList(); };
    $('#monStop').onclick = () => this.stop();
    this.poll();
  },

  spec(key) { return State.models.find(m => m.key === key); },
  val(key, p) { return (this.values[key] && p.name in this.values[key]) ? this.values[key][p.name] : p.default; },
  edited(key) { const v = this.values[key]; return v && Object.keys(v).length > 0; },

  renderList() {
    const by = {}; State.models.forEach(m => (by[m.category] = by[m.category] || []).push(m));
    $('#algoList').innerHTML = CAT_ORDER.filter(c => by[c]).map(c => `<div class="algo-cat">${c}<small>${by[c].length} 种</small></div>` + by[c].map(m => `
      <div class="algo ${m.key === this.focus ? 'focus' : ''}" data-k="${m.key}">
        <input type="checkbox" ${this.checked.has(m.key) ? 'checked' : ''} data-k="${m.key}">
        <span>${m.name}</span>${this.edited(m.key) ? '<span class="edited">已调参</span>' : ''}
      </div>`).join('')).join('');
    $$('#algoList .algo').forEach(el => el.onclick = (e) => {
      if (e.target.tagName === 'INPUT') return;
      this.focus = el.dataset.k; this.renderList(); this.renderParams();
    });
    $$('#algoList input').forEach(cb => cb.onchange = () => { cb.checked ? this.checked.add(cb.dataset.k) : this.checked.delete(cb.dataset.k); });
    $('#trSubmit').textContent = `训练选中算法（${this.checked.size}）`;
    $$('#algoList input').forEach(cb => cb.addEventListener('change', () => $('#trSubmit').textContent = `训练选中算法（${this.checked.size}）`));
  },

  renderParams() {
    const m = this.spec(this.focus);
    $('#paramTitle').textContent = `${m.name} 参数`;
    $('#paramDesc').textContent = `${m.category}。${m.desc}`;
    $('#paramForm').innerHTML = m.params.map(p => {
      const v = this.val(m.key, p);
      const changed = v !== p.default ? ' style="border-color:var(--accent)"' : '';
      if (p.type === 'select') return `<label>${p.label}<select data-p="${p.name}"${changed}>${p.options.map(o => `<option ${o === v ? 'selected' : ''}>${o}</option>`).join('')}</select></label>`;
      if (p.type === 'bool') return `<label class="check"><input type="checkbox" data-p="${p.name}" ${v ? 'checked' : ''}>${p.label}</label>`;
      return `<label>${p.label}<input type="number" data-p="${p.name}" value="${v}" min="${p.min ?? ''}" max="${p.max ?? ''}" step="${p.step || (p.type === 'int' ? 1 : 'any')}"${changed}></label>`;
    }).join('');
    $$('#paramForm [data-p]').forEach(el => el.onchange = () => {
      const p = m.params.find(x => x.name === el.dataset.p);
      let v = el.type === 'checkbox' ? el.checked : el.tagName === 'SELECT' ? el.value : (p.type === 'int' ? parseInt(el.value) : parseFloat(el.value));
      if (typeof v === 'number' && Number.isNaN(v)) v = p.default;
      if (typeof v === 'number') { if (p.min !== undefined) v = Math.max(p.min, v); if (p.max !== undefined) v = Math.min(p.max, v); el.value = v; }
      this.values[m.key] = this.values[m.key] || {};
      if (v === p.default) delete this.values[m.key][p.name]; else this.values[m.key][p.name] = v;
      el.style.borderColor = v !== p.default ? 'var(--accent)' : '';
      this.renderList();
    });
  },

  refreshData() {
    const sel = $('#trData');
    const ws = State.windows;
    sel.innerHTML = ws.length ? ws.map(w => `<option value="${w.id}" ${w.id === State.activeWindow ? 'selected' : ''}>${escapeHtml(w.config.name)}</option>`).join('')
      : '<option value="">（请先在「预测数据构建」构建数据集）</option>';
    sel.onchange = () => State.activeWindow = sel.value;
  },

  async submit() {
    const data_id = $('#trData').value;
    if (!data_id) { toast('请先构建预测数据集', true); Main.go('dataset'); return; }
    const models = [...this.checked].length ? CAT_ORDER.flatMap(c => State.models.filter(m => m.category === c && this.checked.has(m.key)).map(m => m.key)) : [this.focus];
    const jobs = await busy('#trSubmit', () => api('/api/train', { body: { data_id, models, params: this.values } }));
    if (!jobs) return;
    toast(`已提交 ${jobs.length} 个训练任务`);
    await this.refreshJobs();
    this.select(jobs[0].id);
  },

  async poll() {
    clearTimeout(this.pollT);
    try { await this.refreshJobs(); } catch (e) {}
    const active = this.jobs.some(j => j.status === 'running' || j.status === 'queued');
    this.pollT = setTimeout(() => this.poll(), active ? 1500 : 5000);
  },

  async refreshJobs() {
    this.jobs = await api('/api/jobs');
    const run = this.jobs.filter(j => j.status === 'running').length, q = this.jobs.filter(j => j.status === 'queued').length;
    $('#queueBadge').textContent = run || q ? `训练中 ${run}，排队 ${q}` : `已完成任务 ${this.jobs.filter(j => j.metrics).length}`;
    const el = $('#jobList');
    if (!this.jobs.length) { el.innerHTML = '<div class="empty">暂无任务</div>'; return; }
    const stTxt = { queued: '排队中', running: '训练中', done: '完成', stopped: '已停止', error: '出错' };
    el.innerHTML = this.jobs.map(j => `
      <div class="job ${j.id === this.selJob ? 'sel' : ''}" data-id="${j.id}">
        <span class="nm">${j.model_name}</span><span class="status ${j.status}">${stTxt[j.status]}</span>
        <span class="meta"><span>${escapeHtml(j.data_name)}</span>${j.metrics ? `<span>MAE ${fmt(j.metrics.MAE)}</span><span>R² ${j.metrics.R2 == null ? '-' : j.metrics.R2.toFixed(3)}</span>` : ''}${j.train_time ? `<span>${j.train_time.toFixed(1)} 秒</span>` : ''}</span>
        ${j.status === 'running' ? `<div class="mini"><div style="width:${(j.progress * 100).toFixed(1)}%"></div></div>` : ''}
      </div>`).join('');
    $$('#jobList .job').forEach(d => d.onclick = () => this.select(d.dataset.id));
    if (!this.selJob && this.jobs.length) this.select((this.jobs.find(j => j.status === 'running') || this.jobs[0]).id);
    if (Main.page === 'eval') Eval.refresh(false);
  },

  async select(id) {
    if (this.es) { this.es.close(); this.es = null; }
    this.selJob = id;
    $$('#jobList .job').forEach(d => d.classList.toggle('sel', d.dataset.id === id));
    const j = await api('/api/jobs/' + id);
    this.cur = j;
    this.curve = j.curve.slice();
    $('#monTitle').textContent = `${j.model_name} 实时监控`;
    $('#monLog').innerHTML = '';
    j.events.forEach(e => this.onEvent(e, false));
    this.drawCurve(); this.setStatus(j);
    if (j.status === 'running' || j.status === 'queued') {
      this.es = new EventSource(`/api/jobs/${id}/stream?since=${j.n_events}`);
      this.es.onmessage = (m) => {
        const d = JSON.parse(m.data);
        d.events.forEach(e => this.onEvent(e, true));
        this.cur.status = d.status; this.cur.progress = d.progress;
        this.drawCurveThrottled(); this.setStatus(this.cur);
      };
      this.es.addEventListener('end', async () => {
        this.es.close(); this.es = null;
        this.cur = await api('/api/jobs/' + id); this.setStatus(this.cur);
      });
      this.es.onerror = () => { /* browser auto-reconnects; stream resumes from last index via a fresh select */ };
    }
    clearInterval(this.tick);
    this.tick = setInterval(() => this.elapsed(), 500);
  },

  onEvent(e, live) {
    const log = $('#monLog');
    const ts = e.ts !== undefined ? `[${String(e.ts.toFixed(1)).padStart(6)}s] ` : '';
    if (e.type === 'log') log.insertAdjacentText('beforeend', ts + e.msg + '\n');
    else if (e.type === 'error') log.insertAdjacentHTML('beforeend', `<span class="err">${escapeHtml(ts + '错误: ' + e.msg + '\n' + (e.trace || ''))}</span>\n`);
    else if (e.type === 'progress') { $('#monBar').style.width = (e.value * 100).toFixed(1) + '%'; $('#monStage').textContent = e.stage || ''; }
    else if (e.type === 'curve' && live) this.curve.push(e);
    else if (e.type === 'done' && live) {
      this.cur.status = e.status;
      api('/api/jobs/' + this.selJob).then(j => { this.cur = j; this.setStatus(j); });
      this.refreshJobs();
      if (e.status === 'done') toast(`${this.cur.model_name} 训练完成`);
    }
    if (live) log.scrollTop = log.scrollHeight;
  },

  drawCurveThrottled() {
    if (this._dt) return;
    this._dt = setTimeout(() => { this._dt = null; this.drawCurve(); }, 300);
  },

  drawCurve() {
    const c = this.curve, t = T();
    const key = this.cur ? this.cur.model_key : '';
    const xName = { rf: '树的数量', lightgbm: '迭代轮数', svr: '预测步长 h' }[key] || (this.cur && this.cur.category.includes('深度') || key === 'tgcn' ? 'epoch' : 'step');
    const ser = (name, k, color, extra = {}) => line(name, c.filter(p => p[k] != null).map(p => [p.step, p[k]]), { color, showSymbol: c.length < 60, symbolSize: 6, ...extra });
    const has = k => c.some(p => p[k] != null);
    chart('chLoss').setOption(baseOpt({
      grid: { left: 52, right: 14, top: 30, bottom: 36 },
      xAxis: valAxis({ name: xName, nameLocation: 'middle', nameGap: 22, minInterval: 1, scale: true, axisLabel: { color: t.muted, fontSize: 12 } }),
      yAxis: valAxis({ scale: true, axisLabel: { color: t.muted, fontSize: 12, formatter: v => v.toPrecision(3) } }),
      tooltip: { ...baseOpt().tooltip, valueFormatter: v => v == null ? '-' : v.toFixed(5) },
      series: [...(has('train_loss') ? [ser('训练损失', 'train_loss', t.series[0])] : []), ...(has('val_loss') ? [ser('验证损失', 'val_loss', t.series[1])] : [])],
      graphic: c.length ? [] : [{ type: 'text', left: 'center', top: 'middle', style: { text: '等待训练数据', fill: t.muted, fontSize: 12 } }],
    }), true);
    chart('chValMae').setOption(baseOpt({
      legend: { show: false }, grid: { left: 56, right: 14, top: 20, bottom: 36 },
      xAxis: valAxis({ name: xName, nameLocation: 'middle', nameGap: 22, minInterval: 1, scale: true, axisLabel: { color: t.muted, fontSize: 12 } }),
      yAxis: valAxis({ scale: true }),
      series: has('val_mae') ? [ser('验证 MAE', 'val_mae', t.series[2])] : [],
      graphic: has('val_mae') ? [] : [{ type: 'text', left: 'center', top: 'middle', style: { text: '该算法不逐轮报告原始单位 MAE', fill: t.muted, fontSize: 12 } }],
    }), true);
  },

  setStatus(j) {
    const stTxt = { queued: '排队中', running: '训练中', done: '完成', stopped: '已停止', error: '出错' };
    $('#monStatus').className = 'status ' + j.status; $('#monStatus').textContent = stTxt[j.status];
    $('#monStop').hidden = !(j.status === 'running' || j.status === 'queued');
    if (j.status !== 'running' && j.status !== 'queued') $('#monBar').style.width = j.metrics ? '100%' : $('#monBar').style.width;
    const m = j.metrics;
    $('#monMetrics').innerHTML = m ? [
      kpi('测试 MAE', fmt(m.MAE)), kpi('测试 RMSE', fmt(m.RMSE)), kpi('MAPE', m.MAPE == null ? '-' : m.MAPE.toFixed(2) + '%'),
      kpi('R²', m.R2 == null ? '-' : m.R2.toFixed(4)), kpi('训练耗时', (j.train_time || 0).toFixed(1) + ' 秒'),
      `<div class="kpi"><div class="k">详细结果</div><div style="margin-top:8px"><a class="btn" onclick="Main.go('eval');Eval.open('${j.id}')">查看评估</a></div></div>`,
    ].join('') : '';
  },

  elapsed() {
    const j = this.cur; if (!j) return;
    if (j.status === 'queued') { $('#monElapsed').textContent = '等待空闲的训练槽位'; return; }
    const end = j.finished || Date.now() / 1000;
    if (j.status === 'running') $('#monElapsed').textContent = `已用时 ${(Date.now() / 1000 - j.started).toFixed(0)} 秒，进度 ${(j.progress * 100).toFixed(1)}%`;
    else $('#monElapsed').textContent = j.started ? `总用时 ${(end - j.started).toFixed(1)} 秒` : '';
  },

  async stop() {
    await api(`/api/jobs/${this.selJob}/stop`, { method: 'POST' });
    toast('已发送停止指令；深度学习/集成模型会保留当前最佳结果并完成评估');
  },
  onShow() { this.refreshData(); resizeVisible(); },
};
