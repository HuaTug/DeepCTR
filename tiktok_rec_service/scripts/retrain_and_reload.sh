#!/bin/bash
# 增量训练 DeepFM + 热加载到运行中的服务
#
# 用法：
#   bash scripts/retrain_and_reload.sh           # 默认用 30 天数据
#   DAYS=7 bash scripts/retrain_and_reload.sh    # 用 7 天数据
#
# 依赖运行中的服务 http://localhost:${PORT:-8000}/reload
# 失败时不会影响在线服务（旧模型仍然在用）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PORT="${CTR_SERVICE_PORT:-8000}"
DAYS="${DAYS:-30}"
MODEL="${MODEL:-deepfm}"

cd "$PROJECT_DIR"

echo "=========================================="
echo "Step 1: 训练 ${MODEL}（最近 ${DAYS} 天数据）"
echo "=========================================="
MYSQL_HOST="${MYSQL_HOST:-127.0.0.1}" \
MYSQL_PORT="${MYSQL_PORT:-3307}" \
REDIS_HOST="${REDIS_HOST:-127.0.0.1}" \
REDIS_PORT="${REDIS_PORT:-6379}" \
python3 train.py --model "$MODEL" --days "$DAYS" 2>&1 | tail -20
TRAIN_EXIT=${PIPESTATUS[0]}

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "[!] 训练失败 (exit=$TRAIN_EXIT)，跳过 reload，旧模型保持在线"
    exit "$TRAIN_EXIT"
fi

echo ""
echo "=========================================="
echo "Step 2: 调 /reload 让运行中的服务加载新权重"
echo "=========================================="
RESP=$(curl -s -m 30 -X POST "http://localhost:${PORT}/reload")
echo "服务响应: $RESP"
if echo "$RESP" | grep -q '"status":"ok"'; then
    echo "[✓] 模型热更新完成"
    exit 0
else
    echo "[!] /reload 调用失败"
    exit 1
fi
