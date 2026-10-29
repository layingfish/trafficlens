#!/usr/bin/env bash
# 生成一个公网临时链接，让别人通过浏览器访问 TrafficLens（只读演示模式）。
# 你自己在本机打开 http://127.0.0.1:8000 仍然是完整功能，可以训练模型。
# 按 Ctrl+C 结束分享；这台电脑休眠或关机后链接也会失效，重新运行会得到一个新链接。
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
command -v cloudflared >/dev/null || { echo "未找到 cloudflared，请先执行: brew install cloudflared"; exit 1; }

if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/api/mode"; then
  echo "启动本地服务…"
  PORT="$PORT" ./run.sh > results/server.log 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null' EXIT
  for i in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:$PORT/api/mode" && break; sleep 1; done
fi

echo "正在创建公网链接…"
LOG=results/tunnel.log; : > "$LOG"
cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" --logfile "$LOG" > /dev/null 2>&1 &
TUNNEL_PID=$!
trap 'kill $TUNNEL_PID $SERVER_PID 2>/dev/null' EXIT
URL=""
for i in $(seq 1 60); do
  URL=$(grep -oE 'https://[a-z0-9-]+[.]trycloudflare[.]com' "$LOG" | head -1)
  [ -n "$URL" ] && break
  kill -0 $TUNNEL_PID 2>/dev/null || break
  sleep 1
done
if [ -z "$URL" ]; then echo "没能创建链接，cloudflared 日志如下："; tail -20 "$LOG"; exit 1; fi
printf '\n  公网链接（只读演示，发给别人）：%s\n  本机完整版（你自己训练用）：http://127.0.0.1:%s\n\n  按 Ctrl+C 结束分享。\n' "$URL" "$PORT"
wait $TUNNEL_PID
