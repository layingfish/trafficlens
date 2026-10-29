/* App bootstrap: dataset switching, page routing, theme. */
const Main = {
  page: 'eda',
  pages: { eda: EDA, cluster: Cluster, dataset: Dataset, train: Train, eval: Eval },
  loaded: {},

  async init() {
    $$('#nav a').forEach(a => a.onclick = () => this.go(a.dataset.page));
    $('#themeBtn').onclick = () => {
      const next = isDark() ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('tl-theme', next); } catch (e) {}
      location.reload();
    };
    $('#themeBtn').textContent = isDark() ? '浅色' : '深色';
    try { State.readonly = (await api('/api/mode')).readonly; } catch (e) { State.readonly = false; }
    document.body.classList.toggle('readonly', State.readonly);
    const dss = await api('/api/datasets');
    $('#dsSelect').innerHTML = dss.map(d => `<option value="${d.key}">${d.title}</option>`).join('');
    let saved = null; try { saved = localStorage.getItem('tl-ds'); } catch (e) {}
    if (saved && dss.find(d => d.key === saved)) $('#dsSelect').value = saved;
    $('#dsSelect').onchange = () => this.setDataset($('#dsSelect').value);
    EDA.init(); Dataset.init(); Eval.init();
    await Promise.all([Cluster.init(), Train.init()]);
    Dataset.refreshList();
    const hash = location.hash.slice(1);
    await this.setDataset($('#dsSelect').value);
    if (this.pages[hash]) this.go(hash);
    window.addEventListener('hashchange', () => {
      const h = location.hash.slice(1);
      if (this.pages[h] && h !== this.page) this.go(h);
    });
  },

  async setDataset(key) {
    State.ds = key;
    try { localStorage.setItem('tl-ds', key); } catch (e) {}
    State.meta = await api('/api/datasets/' + key);
    this.loaded = {};
    await this.loadPage(this.page);
  },

  async loadPage(p) {
    if (!this.loaded[p] && ['eda', 'cluster', 'dataset'].includes(p)) {
      this.loaded[p] = true;
      await this.pages[p].load();
    }
    this.pages[p].onShow && this.pages[p].onShow();
    resizeVisible();
  },

  go(p) {
    this.page = p;
    history.replaceState(null, '', '#' + p);
    $$('#nav a').forEach(a => a.classList.toggle('active', a.dataset.page === p));
    $$('.page').forEach(s => s.classList.toggle('active', s.id === 'page-' + p));
    window.scrollTo(0, 0);
    this.loadPage(p);
  },
};
Main.init().catch(e => { console.error(e); toast('初始化失败：' + e.message, true); });
