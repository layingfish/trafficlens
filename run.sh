#!/usr/bin/env bash
# 启动 TrafficLens 智能流量分析平台，然后在浏览器打开 http://127.0.0.1:8000
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ -x "$HOME/anaconda3/bin/python" ] && [ -z "$PYTHON" ]; then PY="$HOME/anaconda3/bin/python"; fi
"$PY" -c "import fastapi, torch, sklearn, statsmodels, lightgbm" 2>/dev/null || {
  echo "缺少依赖，请先执行: $PY -m pip install -r requirements.txt"; exit 1; }
echo "启动中… 打开 http://127.0.0.1:${PORT:-8000}"
exec "$PY" app.py
