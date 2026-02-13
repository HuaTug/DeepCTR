"""
Configuration for TikTok Video Recommendation Service (DeepCTR).
Maps to Refactored-TikTok MySQL table schemas.
"""

import os

# =====================================================
# Database Configuration
# =====================================================
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", 3307)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "TikTok@MySQL#2025!Secure"),
    "database": os.getenv("MYSQL_DB", "TikTok"),
    "charset": "utf8mb4",
}

REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "127.0.0.1"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0)),
    "password": os.getenv("REDIS_PASSWORD", ""),
}

# =====================================================
# Table Names (matching Refactored-TikTok schemas)
# =====================================================
TABLE_USERS = "users"
TABLE_VIDEOS = "videos"
TABLE_VIDEO_COUNTERS = "video_counters"
TABLE_USER_BEHAVIORS = "user_behaviors"
TABLE_VIDEO_LIKES = "video_likes"
TABLE_USER_VIDEO_WATCH_HISTORY = "user_video_watch_histories"
TABLE_USER_VIDEO_INTERACTIONS = "user_video_interactions"
TABLE_VIDEO_FEATURES = "video_features"
TABLE_USER_PROFILES = "user_profiles"
TABLE_AUTHOR_SCORES = "author_scores"
TABLE_TAG_VIDEO_MAPPINGS = "tag_video_mappings"
TABLE_CATEGORY_VIDEO_STATS = "category_video_stats"
TABLE_RECOMMENDATION_EXPOSURES = "recommendation_exposures"
TABLE_VIDEO_HOT_SCORES = "video_hot_scores"
TABLE_FAVORITES_VIDEOS = "favorites_videos"
TABLE_VIDEO_SHARES = "video_shares"
TABLE_VIDEO_TOPICS = "video_topics"
TABLE_TOPICS = "topics"

# =====================================================
# Feature Configuration
# =====================================================
# Sparse features: categorical features that need embedding
SPARSE_FEATURES = [
    "user_id",
    "video_id",
    "author_id",
    "category",
    "user_sex",
    "user_level",
    "video_duration_bucket",
    "hour_of_day",
    "day_of_week",
    "device_type",
]

# Dense features: continuous numeric features
DENSE_FEATURES = [
    # User profile features (from user_profiles table)
    "user_avg_watch_duration",
    "user_avg_completion_rate",
    "user_like_rate",
    "user_comment_rate",
    "user_share_rate",
    "user_total_view_count",
    "user_following_count",
    "user_follower_count",
    # Video features (from video_features + videos table)
    "video_quality_score",
    "video_popularity_score",
    "video_ctr",
    "video_finish_rate",
    "video_like_rate",
    "video_comment_rate",
    "video_share_rate",
    "video_favorite_rate",
    "video_avg_watch_duration",
    "video_duration",
    "video_visit_count",
    "video_likes_count",
    "video_comment_count",
    "video_share_count",
    "video_favorites_count",
    # Author features (from author_scores table)
    "author_quality_score",
    "author_influence_score",
    "author_overall_score",
    "author_avg_engagement_rate",
    # Context features
    "video_freshness_hours",
    "video_hot_score",
]

# Sequence features for DIN model (user behavior history)
SEQUENCE_FEATURE = "hist_video_ids"
SEQUENCE_MAX_LEN = 50

# =====================================================
# Model Configuration
# =====================================================
MODEL_CONFIG = {
    "deepfm": {
        "embedding_dim": 8,
        "dnn_hidden_units": (256, 128, 64),
        "dnn_dropout": 0.3,
        "l2_reg_embedding": 1e-5,
        "l2_reg_dnn": 0,
    },
    "din": {
        "embedding_dim": 8,
        "dnn_hidden_units": (256, 128, 64),
        "att_hidden_size": (64, 16),
        "att_activation": "Dice",
        "dnn_dropout": 0.3,
        "l2_reg_embedding": 1e-5,
    },
    "mmoe": {
        "embedding_dim": 8,
        "dnn_hidden_units": (256, 128, 64),
        "num_experts": 4,
        "expert_dim": 128,
        "dnn_dropout": 0.3,
        "task_names": ["is_click", "is_finish", "is_like", "is_share"],
        "task_types": ["binary", "binary", "binary", "binary"],
    },
}

# =====================================================
# Training Configuration
# =====================================================
TRAINING_CONFIG = {
    "batch_size": 1024,
    "epochs": 10,
    "validation_split": 0.2,
    "learning_rate": 1e-3,
    "early_stopping_patience": 3,
    "min_interactions": 5,  # minimum interactions for a user to be in training data
    "train_days": 30,  # how many days of data to use for training
    "label_name": "is_click",  # primary label
}

# =====================================================
# Serving Configuration
# =====================================================
SERVING_CONFIG = {
    "host": os.getenv("CTR_SERVICE_HOST", "0.0.0.0"),
    "port": int(os.getenv("CTR_SERVICE_PORT", 8000)),
    "model_dir": os.getenv("MODEL_DIR", "./models"),
    "default_model": "deepfm",
    "max_batch_size": 500,
    "cache_ttl_seconds": 300,
}

# =====================================================
# Feature Buckets
# =====================================================
DURATION_BUCKETS = [0, 15, 30, 60, 120, 300, 600, float("inf")]
DURATION_LABELS = ["0-15s", "15-30s", "30-60s", "1-2m", "2-5m", "5-10m", "10m+"]
