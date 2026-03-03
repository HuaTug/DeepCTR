
# DeepCTR 推荐模型训练与部署指南

> **项目**：Refactored-TikTok 推荐系统  
> **模型库**：DeepCTR (v0.9.3)  
> **框架**：TensorFlow 2.x + Keras  
> **支持模型**：DeepFM / DIN / MMoE（多任务）

---

## 目录

- [1. 整体架构概览](#1-整体架构概览)
- [2. 环境准备](#2-环境准备)
  - [2.1 Python 环境](#21-python-环境)
  - [2.2 依赖安装](#22-依赖安装)
  - [2.3 基础设施依赖](#23-基础设施依赖)
- [3. 数据库初始化](#3-数据库初始化)
  - [3.1 创建推荐系统表](#31-创建推荐系统表)
  - [3.2 生成 Mock 训练数据](#32-生成-mock-训练数据)
- [4. 数据流水线](#4-数据流水线)
  - [4.1 训练数据来源](#41-训练数据来源)
  - [4.2 特征体系](#42-特征体系)
  - [4.3 特征工程](#43-特征工程)
- [5. 模型训练](#5-模型训练)
  - [5.1 训练 DeepFM（CTR 预估）](#51-训练-deepfmctr-预估)
  - [5.2 训练 DIN（序列推荐）](#52-训练-din序列推荐)
  - [5.3 训练 MMoE（多任务学习）](#53-训练-mmoe多任务学习)
  - [5.4 一键训练全部模型](#54-一键训练全部模型)
  - [5.5 训练参数调优](#55-训练参数调优)
- [6. 模型部署与在线服务](#6-模型部署与在线服务)
  - [6.1 启动 CTR 预测服务](#61-启动-ctr-预测服务)
  - [6.2 API 接口说明](#62-api-接口说明)
  - [6.3 与 Go 后端对接](#63-与-go-后端对接)
- [7. 模型评估与监控](#7-模型评估与监控)
- [8. 高级用法](#8-高级用法)
  - [8.1 DeepCTR 原生 Examples](#81-deepctr-原生-examples)
  - [8.2 可用模型清单](#82-可用模型清单)
  - [8.3 多 GPU 训练](#83-多-gpu-训练)
- [9. 常见问题排查](#9-常见问题排查)

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                   训练 & 部署全流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  MySQL   │───▶│ DataGenerator│───▶│ FeatureProcessor │   │
│  │ (数据源) │    │ (数据提取)    │    │ (特征工程)        │   │
│  └──────────┘    └──────────────┘    └────────┬─────────┘   │
│                                               │             │
│                                               ▼             │
│                                     ┌──────────────────┐    │
│                                     │   train.py       │    │
│                                     │ (模型训练)        │    │
│                                     │ DeepFM/DIN/MMoE  │    │
│                                     └────────┬─────────┘    │
│                                              │              │
│                        ┌─────────────────────┼──────┐       │
│                        ▼                     ▼      ▼       │
│                  ┌──────────┐         ┌──────────────────┐  │
│                  │ .h5 权重 │         │feature_processor │  │
│                  │ 模型文件  │         │     .pkl         │  │
│                  └─────┬────┘         └────────┬─────────┘  │
│                        │                       │            │
│                        ▼                       ▼            │
│                   ┌────────────────────────────────┐        │
│                   │       serve.py (FastAPI)        │        │
│                   │  POST /predict                  │        │
│                   │  POST /predict/ensemble         │        │
│                   │  GET  /health                   │        │
│                   └───────────────┬────────────────┘        │
│                                   │                         │
│                                   ▼                         │
│                   ┌────────────────────────────────┐        │
│                   │  Go ctr_client.go (HTTP)       │        │
│                   │  recommendation_agent.go       │        │
│                   │  (Refactored-TikTok 后端)       │        │
│                   └────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 环境准备

### 2.1 Python 环境

推荐使用 **Python 3.8 ~ 3.10**（TensorFlow 2.x 兼容性最佳）：

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 或使用 conda
conda create -n tiktok-rec python=3.9
conda activate tiktok-rec
```

### 2.2 依赖安装

```bash
cd /data/workspace/TikTok/DeepCTR

# 1. 安装 DeepCTR 库本身（从本地源码）
pip install -e .

# 2. 安装推荐服务专用依赖
cd tiktok_rec_service
pip install -r requirements.txt
```

**requirements.txt 依赖清单：**

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| tensorflow | >=2.6.0, <2.16.0 | 深度学习框架 |
| deepctr[cpu] | >=0.9.0 | CTR 模型库 |
| numpy | >=1.19.0, <2.0.0 | 数值计算 |
| pandas | >=1.3.0, <2.1.0 | 数据处理 |
| scikit-learn | >=1.0.0, <1.4.0 | 特征编码/评估指标 |
| pymysql | >=1.0.0 | MySQL 连接 |
| redis | >=4.0.0 | Redis 连接 |
| fastapi | >=0.95.0, <0.110.0 | HTTP 服务框架 |
| uvicorn | >=0.20.0, <0.30.0 | ASGI 服务器 |
| pydantic | >=1.10.0, <3.0.0 | 数据验证 |

> ⚠️ 如果使用 GPU，将 `tensorflow` 替换为 `tensorflow-gpu`，或安装 `deepctr[gpu]`。

### 2.3 基础设施依赖

训练和服务需要以下基础设施正常运行：

| 服务 | 默认地址 | 用途 |
|------|---------|------|
| MySQL | 127.0.0.1:3307 | 训练数据存储 |
| Redis | 127.0.0.1:6379 | 缓存（可选） |

可通过**环境变量**覆盖默认配置：

```bash
export MYSQL_HOST=127.0.0.1
export MYSQL_PORT=3307
export MYSQL_USER=root
export MYSQL_PASSWORD="TikTok@MySQL#2025!Secure"
export MYSQL_DB=TikTok
```

或直接修改 `config.py` 中的 `MYSQL_CONFIG`。

---

## 3. 数据库初始化

### 3.1 创建推荐系统表

确保 Go 后端已通过 GORM AutoMigrate 创建好核心表（`users`、`videos`、`user_behaviors`），然后执行推荐系统专用建表脚本：

```bash
mysql -h 127.0.0.1 -P 3307 -u root -p TikTok < \
  /data/workspace/TikTok/Refactored-TikTok/config/mysql/recommendation_init.sql
```

该脚本会创建以下推荐表：

| 表名 | 用途 |
|------|------|
| `user_profiles` | 用户画像（兴趣标签、行为统计） |
| `video_features` | 视频特征（质量分、CTR、完播率） |
| `author_scores` | 作者评分（质量、影响力） |
| `user_video_interactions` | 用户-视频交互详情 |
| `recommendation_exposures` | 推荐曝光记录 **（核心训练数据）** |
| `video_hot_scores` | 视频实时热度分 |
| `tag_video_mappings` | 标签-视频映射 |
| `category_video_stats` | 分类级统计 |

### 3.2 生成 Mock 训练数据

如果数据库中还没有足够的交互数据，使用 Mock 数据生成脚本：

```bash
cd /data/workspace/TikTok/DeepCTR/tiktok_rec_service

# 默认：100 用户 × 500 视频 × 80 曝光/用户
python seed_mock_data.py

# 自定义规模
python seed_mock_data.py --users 200 --videos 1000 --exposures-per-user 100

# 清空旧数据后重新生成
python seed_mock_data.py --clean --users 100 --videos 500

# 只更新推荐表（保留已有 users/videos）
python seed_mock_data.py --skip-core
```

**命令行参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--users` | 100 | 用户数量 |
| `--videos` | 500 | 视频数量 |
| `--behaviors-per-user` | 50 | 每用户平均行为数 |
| `--exposures-per-user` | 80 | 每用户平均曝光数 |
| `--skip-core` | false | 跳过 users/videos/user_behaviors 生成 |
| `--clean` | false | 清空推荐表后重新生成 |
| `--clean-all` | false | 清空所有表后重新生成 |
| `--seed` | 42 | 随机种子 |

生成完成后会打印数据摘要：

```
DATA SUMMARY
============================================================
  users                                           100 rows
  videos                                          500 rows
  user_behaviors                               12,345 rows
  user_profiles                                   100 rows
  video_features                                  500 rows
  author_scores                                    20 rows
  user_video_interactions                       5,000 rows
  recommendation_exposures                      8,000 rows    ← 核心训练数据
  video_hot_scores                              2,000 rows

  RECOMMENDATION EXPOSURE STATS:
    Total exposures:      8,000
    Click rate:            0.0800
    Avg completion (click): 0.6543
```

> **关键说明：** Mock 数据中的点击率通过**特征依赖的逻辑回归模型**生成，而非随机噪声。这意味着点击概率与视频质量、用户活跃度、分类匹配度、时段等特征正相关，使模型能学习到有意义的模式。

---

## 4. 数据流水线

### 4.1 训练数据来源

训练数据的核心来源是 `recommendation_exposures` 表，它记录了每次推荐曝光及用户反馈：

```sql
SELECT
    user_id, video_id,
    is_clicked AS is_click,       -- 主标签：是否点击
    completion_rate >= 0.8 AS is_finish,  -- 是否完播
    is_liked AS is_like,          -- 是否点赞
    is_shared AS is_share,        -- 是否分享
    recall_source, position, exposure_time
FROM recommendation_exposures
WHERE exposure_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
```

如果 `recommendation_exposures` 表为空，系统会**自动降级**从 `user_video_interactions` 和 `user_behaviors` 表构建伪曝光数据（正样本 + 负采样，负正比 4:1）。

### 4.2 特征体系

训练特征通过 JOIN 多张 MySQL 表获得：

```
recommendation_exposures (基表)
    ├── LEFT JOIN users            → user_sex, following/follower_count
    ├── LEFT JOIN user_profiles    → avg_watch_duration, like_rate, user_level ...
    ├── LEFT JOIN videos           → category, duration, visit_count, likes_count ...
    ├── LEFT JOIN video_features   → quality_score, popularity_score, ctr ...
    ├── LEFT JOIN author_scores    → author_quality_score, influence_score ...
    └── LEFT JOIN video_hot_scores → video_hot_score (24h 窗口)
```

完整特征列表（在 `config.py` 中定义）：

**稀疏特征（Sparse Features）—— 需要 Embedding：**

| 特征名 | 来源 | 编码方式 |
|--------|------|---------|
| `category` | videos.category | LabelEncoder |
| `user_sex` | users.sex | LabelEncoder |
| `user_level` | user_profiles.user_level | LabelEncoder |
| `video_duration_bucket` | 派生：videos.duration 分桶 | LabelEncoder |
| `hour_of_day` | 派生：exposure_time.hour | LabelEncoder |
| `day_of_week` | 派生：exposure_time.dayofweek | LabelEncoder |
| `device_type` | 上下文 | LabelEncoder |
| `user_id` | users.user_id | Hash Bucket (500) |
| `video_id` | videos.video_id | Hash Bucket (500) |
| `author_id` | videos.user_id | Hash Bucket (200) |

**稠密特征（Dense Features）—— MinMaxScaler 归一化：**

| 特征名 | 来源 | 说明 |
|--------|------|------|
| `user_avg_watch_duration` | user_profiles | 用户平均观看时长 |
| `user_avg_completion_rate` | user_profiles | 用户平均完播率 |
| `user_like_rate` | user_profiles | 用户点赞率 |
| `user_comment_rate` | user_profiles | 用户评论率 |
| `user_share_rate` | user_profiles | 用户分享率 |
| `user_total_view_count` | user_profiles | 用户总观看数 |
| `user_following_count` | users | 关注数 |
| `user_follower_count` | users | 粉丝数 |
| `video_quality_score` | video_features | 视频质量分 (0-10) |
| `video_popularity_score` | video_features | 视频热度分 |
| `video_avg_watch_duration` | video_features | 视频平均观看时长 |
| `video_duration` | videos | 视频时长(秒) |
| `video_visit_count` | videos | 播放量 |
| `video_likes_count` | videos | 点赞数 |
| `video_comment_count` | videos | 评论数 |
| `video_share_count` | videos | 分享数 |
| `video_favorites_count` | videos | 收藏数 |
| `author_quality_score` | author_scores | 作者质量分 |
| `author_influence_score` | author_scores | 作者影响力 |
| `author_overall_score` | author_scores | 作者综合分 |
| `author_avg_engagement_rate` | author_scores | 作者平均互动率 |
| `video_freshness_hours` | 派生 | 视频新鲜度(小时) |
| `video_hot_score` | video_hot_scores | 实时热度分 |
| `user_video_duration_ratio` | 派生 | 用户偏好时长/视频时长 |
| `user_activity_log` | 派生 | log(用户总观看数+1) |
| `video_popularity_log` | 派生 | log(视频播放量+1) |

### 4.3 特征工程

特征工程由 `feature_engineering.py` 中的 `FeatureProcessor` 类完成：

```python
# 1. 派生特征
video_duration_bucket  ← pd.cut(duration, bins=[0,15,30,60,120,300,600,∞])
hour_of_day           ← exposure_time.hour
day_of_week           ← exposure_time.dayofweek
video_freshness_hours ← (exposure_time - video_created_at).hours
user_video_duration_ratio ← user_avg_watch_duration / video_duration
user_activity_log     ← log1p(user_total_view_count)
video_popularity_log  ← log1p(video_visit_count)

# 2. 稀疏特征编码
LabelEncoder → 将类别字符串映射为整数索引
Hash Bucket  → 高基数 ID (user_id/video_id/author_id) 通过 hash(x) % bucket_size 降维

# 3. 稠密特征归一化
MinMaxScaler → 将数值特征缩放到 [0, 1] 区间

# 4. 构建 DeepCTR FeatureColumn
SparseFeat(name, vocabulary_size, embedding_dim=8)  → 稀疏特征
DenseFeat(name, dimension=1)                        → 稠密特征
VarLenSparseFeat(...)                               → 变长序列特征 (DIN)
```

---

## 5. 模型训练

### 5.1 训练 DeepFM（CTR 预估）

**DeepFM = FM（二阶特征交叉） + DNN（高阶特征交叉）**，是 CTR 预估的经典基线模型。

```bash
cd /data/workspace/TikTok/DeepCTR/tiktok_rec_service

python train.py --model deepfm
```

**训练流程：**

```
Step 1: Loading training data from MySQL...
  └── DataGenerator.generate_training_data(days=30)
  └── JOIN exposures + users + user_profiles + videos + video_features + author_scores + hot_scores

Step 2: Feature engineering...
  └── FeatureProcessor.fit_transform(df_train)  ← 仅在训练集上 fit，避免数据泄漏
  └── FeatureProcessor.transform(df_val)

Step 3: Train/Validation split (80/20)...
  └── Shuffle + split

Step 4: Training DeepFM model...
  └── model = DeepFM(linear_feature_columns, dnn_feature_columns,
                      dnn_hidden_units=(128, 64), dnn_dropout=0.2, task='binary')
  └── model.compile(Adam(lr=1e-3), loss='binary_crossentropy', metrics=['AUC'])
  └── model.fit(epochs=50, batch_size=256, EarlyStopping(patience=8, monitor='val_auc'))
```

**输出文件：**

```
models/
├── deepfm_best.h5           # 最优模型 checkpoint (by val_auc)
├── deepfm_weights.h5        # 最终权重文件
├── feature_processor.pkl    # 特征处理器（Serving 需要）
└── training_metrics.json    # 训练指标
```

**模型参数（`config.py` → `MODEL_CONFIG["deepfm"]`）：**

```python
{
    "embedding_dim": 8,          # 稀疏特征 Embedding 维度
    "dnn_hidden_units": (128, 64),  # DNN 隐藏层
    "dnn_dropout": 0.2,          # Dropout 率
    "l2_reg_embedding": 1e-5,    # Embedding L2 正则
    "l2_reg_dnn": 1e-5,          # DNN L2 正则
}
```

### 5.2 训练 DIN（序列推荐）

**DIN（Deep Interest Network）** 使用 Attention 机制从用户历史行为序列中提取兴趣表达，适合有丰富交互历史的用户。

```bash
python train.py --model din
```

**DIN 特有配置：**

```python
{
    "att_hidden_size": (64, 16),  # Attention 网络隐藏层
    "att_activation": "Dice",     # Attention 激活函数（DIN 论文提出）
}
```

**注意：** DIN 需要用户行为序列特征 (`hist_video_ids`)，如果数据中缺少该特征，会自动降级为 DeepFM。

### 5.3 训练 MMoE（多任务学习）

**MMoE（Multi-gate Mixture-of-Experts）** 同时优化多个任务，利用共享 Expert 实现任务间知识迁移。

```bash
python train.py --model mmoe
```

**四个训练任务及权重：**

| 任务 | 标签 | Loss 权重 | 说明 |
|------|------|-----------|------|
| is_click | 是否点击 | 1.0 | 主任务 |
| is_finish | 是否完播 | 0.8 | 内容质量指标 |
| is_like | 是否点赞 | 0.5 | 用户喜好指标 |
| is_share | 是否分享 | 0.3 | 传播价值指标 |

**MMoE 特有配置：**

```python
{
    "num_experts": 3,            # Expert 网络数量
    "expert_dim": 64,            # Expert 隐藏层维度
    "task_names": ["is_click", "is_finish", "is_like", "is_share"],
    "task_types": ["binary", "binary", "binary", "binary"],
}
```

### 5.4 一键训练全部模型

```bash
# 训练 DeepFM + DIN + MMoE
python train.py --model all

# 自定义训练天数和输出目录
python train.py --model all --days 60 --output ./my_models
```

### 5.5 训练参数调优

所有训练超参在 `config.py` → `TRAINING_CONFIG` 中配置：

```python
TRAINING_CONFIG = {
    "batch_size": 256,              # 批大小
    "epochs": 50,                   # 最大训练轮数
    "validation_split": 0.2,        # 验证集比例
    "learning_rate": 1e-3,          # 学习率
    "early_stopping_patience": 8,   # 早停耐心值
    "min_interactions": 5,          # 用户最少交互数
    "train_days": 30,               # 训练数据天数
    "label_name": "is_click",       # 主标签
}
```

**调优建议：**

| 场景 | 调整建议 |
|------|---------|
| 数据量 < 1万 | 减小 `dnn_hidden_units` 为 (64, 32)，增大 `dnn_dropout` 为 0.3 |
| 数据量 > 100万 | 增大 `batch_size` 为 1024，`dnn_hidden_units` 为 (256, 128, 64) |
| 过拟合 | 增大 `l2_reg_embedding`/`l2_reg_dnn`，增大 `dnn_dropout` |
| 收敛慢 | 减小 `learning_rate` 为 5e-4，增大 `epochs` |
| 样本不均衡 | 在 train.py 中增加 `class_weight` 或做欠采样 |

---

## 6. 模型部署与在线服务

### 6.1 启动 CTR 预测服务

```bash
cd /data/workspace/TikTok/DeepCTR/tiktok_rec_service

# 直接启动（开发模式）
python serve.py

# 或使用 uvicorn 启动（生产模式）
uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 1

# 自定义端口
CTR_SERVICE_PORT=9000 python serve.py
```

> ⚠️ **workers 必须设为 1**，因为 TensorFlow 模型不支持 fork 多进程。如需扩展，请使用多实例 + 负载均衡。

服务启动时会：
1. 加载 `feature_processor.pkl`（特征处理器）
2. 依次尝试加载 `deepfm_weights.h5`、`din_weights.h5`、`mmoe_weights.h5`
3. 构建模型图并加载权重
4. 打印 "CTR prediction service ready!"

### 6.2 API 接口说明

#### `POST /predict` — 单模型预测

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "video_ids": [101, 102, 103, 104, 105],
    "model": "deepfm"
  }'
```

**响应：**

```json
{
  "predictions": [
    {"video_id": 103, "score": 0.82, "ctr": 0.82, "is_finish": 0.0, "is_like": 0.0, "is_share": 0.0},
    {"video_id": 101, "score": 0.65, "ctr": 0.65, "is_finish": 0.0, "is_like": 0.0, "is_share": 0.0},
    {"video_id": 105, "score": 0.58, "ctr": 0.58, "is_finish": 0.0, "is_like": 0.0, "is_share": 0.0},
    {"video_id": 102, "score": 0.43, "ctr": 0.43, "is_finish": 0.0, "is_like": 0.0, "is_share": 0.0},
    {"video_id": 104, "score": 0.21, "ctr": 0.21, "is_finish": 0.0, "is_like": 0.0, "is_share": 0.0}
  ],
  "latency_ms": 12.34,
  "model": "deepfm"
}
```

#### `POST /predict/ensemble` — 集成预测

```bash
curl -X POST http://localhost:8000/predict/ensemble \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "video_ids": [101, 102, 103]
  }'
```

融合所有已加载模型的预测分数（取平均），返回综合排序。

#### `GET /health` — 健康检查

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "models_loaded": ["deepfm", "mmoe"],
  "uptime_seconds": 3600.5
}
```

#### `GET /metrics` — 服务指标

```bash
curl http://localhost:8000/metrics
```

### 6.3 与 Go 后端对接

Go 后端通过 `ctr_client.go` 中的 `CTRServiceClient` 与 Python CTR 服务通信：

```go
// 初始化（在 Go 服务启动时）
recommendation.InitCTRClient(&recommendation.CTRServiceConfig{
    ServiceURL:   "http://localhost:8000",
    Timeout:      200 * time.Millisecond,
    MaxRetries:   2,
    DefaultModel: "deepfm",
})

// 调用预测
client := recommendation.GetCTRClient()
predictions, err := client.Predict(ctx, userID, videoIDs, contextInfo)

// 使用指定模型
predictions, err := client.PredictWithModel(ctx, userID, videoIDs, "mmoe")
```

**容错机制：**
- 10 秒一次健康检查（`/health`）
- 服务不可用时自动降级为 fallback 评分（统一返回 0.5）
- 支持最多 2 次重试，间隔 50ms

---

## 7. 模型评估与监控

训练完成后，评估指标自动保存到 `models/training_metrics.json`：

```json
{
  "timestamp": "2026-02-13T16:55:36.701405",
  "total_samples": 29843,
  "train_samples": 23874,
  "val_samples": 5969,
  "positive_ratio": 0.0800,
  "models": {
    "deepfm": {
      "auc": 0.6399,
      "logloss": 0.2656
    },
    "mmoe": {
      "is_click_auc": 0.6512,
      "is_finish_auc": 0.7234,
      "is_like_auc": 0.6891,
      "is_share_auc": 0.6543
    }
  }
}
```

**评估指标说明：**

| 指标 | 含义 | 目标 |
|------|------|------|
| AUC | ROC 曲线下面积 | 越高越好，> 0.65 可用 |
| LogLoss | 对数损失 | 越低越好 |
| Positive Ratio | 正样本比例 | 通常 5%~15% |

---

## 8. 高级用法

### 8.1 DeepCTR 原生 Examples

DeepCTR 库自带丰富的训练示例，位于 `examples/` 目录：

```bash
cd /data/workspace/TikTok/DeepCTR/examples

# Criteo 数据集 CTR 分类（DeepFM）
python run_classification_criteo.py

# MovieLens 评分回归
python run_regression_movielens.py

# DIN 序列推荐
python run_din.py

# DIEN (DIN 增强版)
python run_dien.py

# MMoE 多任务学习
python run_mtl.py

# DSIN 会话推荐
python run_dsin.py

# 多值特征 (一个 field 对应多个值)
python run_multivalue_movielens.py

# Hash 特征 (高基数稀疏特征)
python run_classification_criteo_hash.py

# Estimator API (TF 分布式训练)
python run_estimator_pandas_classification.py

# 多 GPU 训练
python run_classification_criteo_multi_gpu.py
```

### 8.2 可用模型清单

DeepCTR 提供 **25+** 种模型，均可在本项目中使用：

| 模型 | 类型 | 论文 | 特点 |
|------|------|------|------|
| **DeepFM** | CTR | IJCAI 2017 | FM + DNN，低阶+高阶交叉 |
| **DIN** | 序列 | KDD 2018 | Attention 提取兴趣 |
| **DIEN** | 序列 | AAAI 2019 | 兴趣演化网络 |
| **DSIN** | 序列 | IJCAI 2019 | 会话级兴趣 |
| **BST** | 序列 | DLP-KDD 2019 | Transformer 序列建模 |
| **MMoE** | 多任务 | KDD 2018 | 多门专家混合 |
| **PLE** | 多任务 | RecSys 2020 | 渐进分层专家提取 |
| **ESMM** | 多任务 | SIGIR 2018 | 全空间多任务 |
| **SharedBottom** | 多任务 | - | 共享底座多任务 |
| **DCN** | CTR | ADKDD 2017 | 交叉网络 |
| **DCN-Mix** | CTR | WWW 2021 | 混合交叉网络 |
| **xDeepFM** | CTR | KDD 2018 | 压缩交互网络 |
| **AutoInt** | CTR | CIKM 2019 | 自注意力交互 |
| **FiBiNET** | CTR | RecSys 2019 | 双线性特征交互 |
| WDL | CTR | DLRS 2016 | Wide & Deep |
| FNN | CTR | ECIR 2016 | FM 预训练 + DNN |
| PNN | CTR | ICDM 2016 | 乘积网络 |
| NFM | CTR | SIGIR 2017 | 神经 FM |
| AFM | CTR | IJCAI 2017 | 注意力 FM |
| FGCNN | CTR | WWW 2019 | 特征生成 CNN |

### 8.3 多 GPU 训练

```python
# 参考 examples/run_classification_criteo_multi_gpu.py
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy", metrics=['AUC'])

model.fit(train_input, train_label, batch_size=1024, epochs=10)
```

---

## 9. 常见问题排查

### Q1: 训练时报 "Insufficient training data: N samples. Need at least 100."

**原因：** MySQL 中的曝光/交互数据不足。

**解决：**
```bash
# 生成 Mock 数据
python seed_mock_data.py --users 100 --videos 500 --exposures-per-user 100
```

### Q2: 训练 AUC 很低 (< 0.55)，接近随机

**可能原因：**
- 数据量太少（< 1000 条）
- 正负样本严重不均衡
- 特征没有区分度

**解决：**
- 增加训练数据量：`python train.py --days 60`
- 检查 `positive_ratio`，如果太低 (< 0.01)，考虑欠采样负样本
- 检查 `config.py` 中 DENSE_FEATURES 注释提到的标签泄漏问题

### Q3: serve.py 启动报 "No trained models found"

**原因：** `models/` 目录下没有模型权重文件。

**解决：** 先运行训练再启动服务：
```bash
python train.py --model deepfm
python serve.py
```

### Q4: Go 端报 "CTR Service unhealthy, using fallback scoring"

**原因：** Python CTR 服务未启动或不可达。

**解决：**
```bash
# 确认 CTR 服务是否运行
curl http://localhost:8000/health

# 检查端口是否正确
# Go 端默认连接 http://localhost:8000
# 在 Go 代码中修改: recommendation.InitCTRClient(&CTRServiceConfig{ServiceURL: "http://your-host:8000"})
```

### Q5: TensorFlow 版本冲突

**解决：**
```bash
# 检查 TF 版本
python -c "import tensorflow; print(tensorflow.__version__)"

# 推荐版本：2.10.x ~ 2.15.x
pip install tensorflow==2.13.0
```

### Q6: 训练时 OOM (Out of Memory)

**解决：**
- 减小 `batch_size`（256 → 128 → 64）
- 减小 `embedding_dim`（8 → 4）
- 减小 `dnn_hidden_units`（(128,64) → (64,32)）
- 限制 GPU 显存：
```python
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
```

---

## 完整执行流程速查

```bash
# === 1. 环境准备 ===
cd /data/workspace/TikTok/DeepCTR
pip install -e .
cd tiktok_rec_service
pip install -r requirements.txt

# === 2. 数据库初始化 ===
mysql -h 127.0.0.1 -P 3307 -u root -p TikTok < \
  /data/workspace/TikTok/Refactored-TikTok/config/mysql/recommendation_init.sql

# === 3. 生成训练数据 ===
python seed_mock_data.py --users 100 --videos 500

# === 4. 训练模型 ===
python train.py --model deepfm       # 单模型
python train.py --model mmoe         # 多任务
python train.py --model all          # 全部模型

# === 5. 启动 CTR 服务 ===
python serve.py

# === 6. 测试接口 ===
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "video_ids": [1,2,3,4,5]}'

# === 7. 启动 Go 后端（Go 会自动连接 CTR 服务）===
cd /data/workspace/TikTok/Refactored-TikTok
go run cmd/api/main.go
```
