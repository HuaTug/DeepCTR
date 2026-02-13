#!/bin/bash
# ========== 配置 ==========
API_HOST="http://localhost:8888"
USERNAME="testuser"
PASSWORD="test123456"

# ========== Step 0: 注册用户（如果还没注册过）==========
echo "=== Step 0: Register User ==="
REGISTER_RESP=$(curl -s -X POST "${API_HOST}/v1/user/create/" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"${USERNAME}\",\"password\":\"${PASSWORD}\"}")

echo "Register Response: ${REGISTER_RESP}"
echo ""

# ========== Step 1: 登录获取 Token ==========
echo "=== Step 1: Login ==="
LOGIN_RESP=$(curl -s -X POST "${API_HOST}/v1/user/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"${USERNAME}\",\"password\":\"${PASSWORD}\"}")

echo "Login Response: ${LOGIN_RESP}"

# 提取 token (需要 jq 工具: brew install jq)
ACCESS_TOKEN=$(echo "${LOGIN_RESP}" | jq -r '.data.token')
REFRESH_TOKEN=$(echo "${LOGIN_RESP}" | jq -r '.data.RefreshToken')

if [ "${ACCESS_TOKEN}" == "null" ] || [ -z "${ACCESS_TOKEN}" ]; then
  echo "❌ Login failed, cannot extract token"
  exit 1
fi

echo "✅ Access Token: ${ACCESS_TOKEN:0:50}..."
echo ""

# ========== Step 2: 请求推荐视频 ==========
echo "=== Step 2: Get Recommended Videos ==="
RECOMMEND_RESP=$(curl -s -X GET "${API_HOST}/v1/recommend/video" \
  -H "Access-Token: ${ACCESS_TOKEN}" \
  -H "Refresh-Token: ${REFRESH_TOKEN}")

echo "Recommend Response:"
echo "${RECOMMEND_RESP}" | jq '.'

# ========== Step 3: 统计结果 ==========
VIDEO_COUNT=$(echo "${RECOMMEND_RESP}" | jq '.data.video_list | length')
ALGORITHM=$(echo "${RECOMMEND_RESP}" | jq -r '.data.algorithm_used')
REC_ID=$(echo "${RECOMMEND_RESP}" | jq -r '.data.recommendation_id')

echo ""
echo "========== Summary =========="
echo "📹 Recommended videos: ${VIDEO_COUNT}"
echo "🧠 Algorithm used:     ${ALGORITHM}"
echo "🆔 Recommendation ID:  ${REC_ID}"
