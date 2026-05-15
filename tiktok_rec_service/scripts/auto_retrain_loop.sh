#!/bin/bash
# 自动重训守护：每隔 INTERVAL 秒（默认 1 小时）跑一次 train+/reload。
# 失败会吞掉错误继续下一轮，不会退出。
#
# 启动：
#   nohup bash scripts/auto_retrain_loop.sh > /tmp/ctr_auto_retrain.log 2>&1 &
# 停止：
#   pkill -f auto_retrain_loop.sh
#
# 环境变量：
#   INTERVAL  : 间隔秒数（默认 3600）
#   DAYS      : 每次训练用最近多少天数据（默认 30）
#   MODEL     : deepfm / din / mmoe（默认 deepfm）

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INTERVAL="${INTERVAL:-3600}"
export DAYS="${DAYS:-30}"
export MODEL="${MODEL:-deepfm}"
export MYSQL_HOST="${MYSQL_HOST:-127.0.0.1}"
export MYSQL_PORT="${MYSQL_PORT:-3307}"
export REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
export REDIS_PORT="${REDIS_PORT:-6379}"
export CTR_SERVICE_PORT="${CTR_SERVICE_PORT:-8000}"

cd "$PROJECT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto retrain loop started, interval=${INTERVAL}s model=${MODEL} days=${DAYS}"

while true; do
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] tick"
    echo "============================================================"
    if bash "$SCRIPT_DIR/retrain_and_reload.sh"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED, will retry next cycle"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] sleeping ${INTERVAL}s ..."
    sleep "$INTERVAL"
done
