"""
Data Generator for TikTok Video Recommendation.
Extracts training data from MySQL tables defined in Refactored-TikTok.

Training sample schema:
  - Label: is_click (from recommendation_exposures.is_clicked)
  - Multi-task labels: is_finish, is_like, is_share
  - Features: user profile + video features + author scores + context
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine, text
from config import (
    MYSQL_CONFIG,
    TABLE_AUTHOR_SCORES,
    TABLE_RECOMMENDATION_EXPOSURES,
    TABLE_USER_BEHAVIORS,
    TABLE_USER_PROFILES,
    TABLE_USER_VIDEO_INTERACTIONS,
    TABLE_USERS,
    TABLE_VIDEO_FEATURES,
    TABLE_VIDEO_HOT_SCORES,
    TABLE_VIDEOS,
    TRAINING_CONFIG,
)

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generates training data by joining multiple MySQL tables."""

    def __init__(self, mysql_config: Optional[dict] = None):
        self.mysql_config = mysql_config or MYSQL_CONFIG
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            cfg = self.mysql_config
            password = quote_plus(cfg["password"])
            url = f"mysql+pymysql://{cfg['user']}:{password}@{cfg['host']}:{cfg['port']}/{cfg['database']}?charset={cfg.get('charset', 'utf8mb4')}"
            self._engine = create_engine(url)
        return self._engine

    def _get_connection(self):
        return pymysql.connect(**self.mysql_config, cursorclass=pymysql.cursors.DictCursor)

    def _read_sql(self, sql, params=None):
        """Helper to execute pd.read_sql with SQLAlchemy engine, converting %s to :p0, :p1, ..."""
        if params:
            # Replace %s placeholders with named params :p0, :p1, ...
            new_sql = sql
            named_params = {}
            for i, val in enumerate(params):
                new_sql = new_sql.replace("%s", f":p{i}", 1)
                named_params[f"p{i}"] = val
            return pd.read_sql(text(new_sql), self._get_engine(), params=named_params)
        else:
            return pd.read_sql(text(sql), self._get_engine())

    # =====================================================
    # Main entry: generate training DataFrame
    # =====================================================

    def generate_training_data(self, days: int = None) -> pd.DataFrame:
        """
        Generate training data by joining:
          - recommendation_exposures (labels: is_clicked, is_liked, etc.)
          - users (user_sex, following_count, follower_count)
          - user_profiles (user behavior stats)
          - videos (category, duration, label_names)
          - video_features (ctr, finish_rate, quality_score, etc.)
          - author_scores (author quality, influence)
          - video_hot_scores (real-time hot score)

        Returns a flat DataFrame ready for FeatureProcessor.
        """
        if days is None:
            days = TRAINING_CONFIG["train_days"]

        conn = self._get_connection()
        try:
            logger.info(f"Generating training data for last {days} days...")

            # Step 1: Get exposure records as base (these are our labeled samples)
            df_exposure = self._load_exposures(conn, days)
            if df_exposure.empty:
                logger.warning("No exposure records found. Trying to build from user_behaviors...")
                df_exposure = self._build_exposures_from_behaviors(conn, days)

            if df_exposure.empty:
                logger.warning("No training data available.")
                return pd.DataFrame()

            logger.info(f"Loaded {len(df_exposure)} exposure samples")

            # Step 2: Join user features
            user_ids = df_exposure["user_id"].unique().tolist()
            df_users = self._load_user_features(conn, user_ids)
            df_exposure = df_exposure.merge(df_users, on="user_id", how="left")

            # Step 3: Join video features
            video_ids = df_exposure["video_id"].unique().tolist()
            df_videos = self._load_video_features(conn, video_ids)
            df_exposure = df_exposure.merge(df_videos, on="video_id", how="left")

            # Step 4: Join author scores
            author_ids = df_exposure["author_id"].dropna().unique().tolist()
            if author_ids:
                df_authors = self._load_author_features(conn, author_ids)
                df_exposure = df_exposure.merge(df_authors, on="author_id", how="left")

            # Step 5: Join video hot scores
            df_hot = self._load_video_hot_scores(conn, video_ids)
            if not df_hot.empty:
                df_exposure = df_exposure.merge(df_hot, on="video_id", how="left")

            logger.info(f"Training data shape: {df_exposure.shape}")
            # Ensure label columns are numeric
            for col in ["is_click", "is_finish", "is_like", "is_share"]:
                if col in df_exposure.columns:
                    df_exposure[col] = pd.to_numeric(df_exposure[col], errors="coerce").fillna(0).astype(int)
            logger.info(f"Positive ratio (is_click): {df_exposure['is_click'].mean():.4f}")

            return df_exposure

        finally:
            conn.close()

    # =====================================================
    # Load exposure records (labeled data)
    # =====================================================

    def _load_exposures(self, conn, days: int) -> pd.DataFrame:
        """Load labeled samples from recommendation_exposures table."""
        cutoff = datetime.now() - timedelta(days=days)
        sql = f"""
            SELECT
                re.user_id,
                re.video_id,
                re.is_clicked AS is_click,
                COALESCE(re.completion_rate >= 0.8, 0) AS is_finish,
                COALESCE(re.is_liked, 0) AS is_like,
                COALESCE(re.is_shared, 0) AS is_share,
                re.watch_duration,
                re.completion_rate,
                re.recall_source,
                re.position,
                re.exposure_time
            FROM {TABLE_RECOMMENDATION_EXPOSURES} re
            WHERE re.exposure_time >= %s
            ORDER BY re.exposure_time ASC
        """
        df = self._read_sql(sql, params=[cutoff.strftime("%Y-%m-%d %H:%M:%S")])
        return df

    def _build_exposures_from_behaviors(self, conn, days: int) -> pd.DataFrame:
        """
        Fallback: build pseudo-exposure data from user_behaviors and
        user_video_interactions when recommendation_exposures is empty.
        This creates positive samples from actual interactions and
        generates negative samples by random sampling.
        """
        cutoff = datetime.now() - timedelta(days=days)

        # Positive samples from user_video_interactions
        sql_positive = f"""
            SELECT
                uvi.user_id,
                uvi.video_id,
                1 AS is_click,
                CASE WHEN uvi.max_watch_progress >= 0.8 THEN 1 ELSE 0 END AS is_finish,
                CAST(uvi.is_liked AS SIGNED) AS is_like,
                CAST(uvi.is_shared AS SIGNED) AS is_share,
                uvi.total_watch_time AS watch_duration,
                uvi.max_watch_progress AS completion_rate,
                'behavior' AS recall_source,
                0 AS position,
                uvi.last_interact_at AS exposure_time
            FROM {TABLE_USER_VIDEO_INTERACTIONS} uvi
            WHERE uvi.last_interact_at >= %s
                AND uvi.click_count > 0
        """
        df_pos = self._read_sql(sql_positive, params=[cutoff.strftime("%Y-%m-%d %H:%M:%S")])

        if df_pos.empty:
            # Try user_behaviors table
            sql_behaviors = f"""
                SELECT
                    ub.user_id,
                    ub.video_id,
                    1 AS is_click,
                    CASE WHEN ub.behavior_type = 'view' THEN 0 ELSE 0 END AS is_finish,
                    CASE WHEN ub.behavior_type = 'like' THEN 1 ELSE 0 END AS is_like,
                    CASE WHEN ub.behavior_type = 'share' THEN 1 ELSE 0 END AS is_share,
                    0 AS watch_duration,
                    0.0 AS completion_rate,
                    'behavior' AS recall_source,
                    0 AS position,
                    ub.behavior_time AS exposure_time
                FROM {TABLE_USER_BEHAVIORS} ub
                WHERE ub.behavior_time >= %s
            """
            df_pos = self._read_sql(sql_behaviors, params=[cutoff.strftime("%Y-%m-%d %H:%M:%S")])

        if df_pos.empty:
            return pd.DataFrame()

        # Deduplicate positive samples (keep the latest interaction)
        df_pos = df_pos.sort_values("exposure_time", ascending=False)
        df_pos = df_pos.drop_duplicates(subset=["user_id", "video_id"], keep="first")

        # Generate negative samples (random user-video pairs that did NOT interact)
        n_neg = min(len(df_pos) * 4, 100000)  # 4:1 negative ratio, max 100k
        df_neg = self._generate_negative_samples(conn, df_pos, n_neg)

        df = pd.concat([df_pos, df_neg], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

        logger.info(f"Built {len(df_pos)} positive + {len(df_neg)} negative samples from behaviors")
        return df

    def _generate_negative_samples(self, conn, df_pos: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Generate negative samples by random pairing users with videos they haven't interacted with."""
        user_ids = df_pos["user_id"].unique()
        
        # Get all public video IDs
        sql = f"SELECT video_id FROM {TABLE_VIDEOS} WHERE open = 1 AND audit_status = 1 AND deleted_at IS NULL"
        df_videos = self._read_sql(sql)
        all_video_ids = df_videos["video_id"].values

        if len(all_video_ids) == 0:
            return pd.DataFrame()

        # Build positive set for fast lookup
        pos_set = set(zip(df_pos["user_id"], df_pos["video_id"]))

        neg_records = []
        rng = np.random.default_rng(42)

        attempts = 0
        max_attempts = n_samples * 10
        while len(neg_records) < n_samples and attempts < max_attempts:
            uid = rng.choice(user_ids)
            vid = rng.choice(all_video_ids)
            if (uid, vid) not in pos_set:
                neg_records.append({
                    "user_id": int(uid),
                    "video_id": int(vid),
                    "is_click": 0,
                    "is_finish": 0,
                    "is_like": 0,
                    "is_share": 0,
                    "watch_duration": 0,
                    "completion_rate": 0.0,
                    "recall_source": "negative_sample",
                    "position": 0,
                    "exposure_time": df_pos["exposure_time"].iloc[rng.integers(0, len(df_pos))],
                })
                pos_set.add((uid, vid))
            attempts += 1

        return pd.DataFrame(neg_records)

    # =====================================================
    # Load user features
    # =====================================================

    def _load_user_features(self, conn, user_ids: list) -> pd.DataFrame:
        """
        Join users + user_profiles tables to get user features.
        Maps to:
          - users: user_id, sex, following_count, follower_count
          - user_profiles: avg_watch_duration, like_rate, comment_rate, etc.
        """
        if not user_ids:
            return pd.DataFrame(columns=["user_id"])

        placeholders = ",".join(["%s"] * len(user_ids))
        sql = f"""
            SELECT
                u.user_id,
                u.sex AS user_sex,
                u.following_count AS user_following_count,
                u.follower_count AS user_follower_count,
                COALESCE(up.avg_watch_duration, 0) AS user_avg_watch_duration,
                COALESCE(up.avg_completion_rate, 0) AS user_avg_completion_rate,
                COALESCE(up.like_rate, 0) AS user_like_rate,
                COALESCE(up.comment_rate, 0) AS user_comment_rate,
                COALESCE(up.share_rate, 0) AS user_share_rate,
                COALESCE(up.total_view_count, 0) AS user_total_view_count,
                COALESCE(up.user_level, 1) AS user_level
            FROM {TABLE_USERS} u
            LEFT JOIN {TABLE_USER_PROFILES} up ON u.user_id = up.user_id
            WHERE u.user_id IN ({placeholders})
        """
        df = self._read_sql(sql, params=user_ids)
        return df

    # =====================================================
    # Load video features
    # =====================================================

    def _load_video_features(self, conn, video_ids: list) -> pd.DataFrame:
        """
        Join videos + video_features tables.
        Maps to:
          - videos: user_id(author), category, duration, visit_count, likes_count, etc.
          - video_features: quality_score, ctr, finish_rate, etc.
        """
        if not video_ids:
            return pd.DataFrame(columns=["video_id"])

        placeholders = ",".join(["%s"] * len(video_ids))
        sql = f"""
            SELECT
                v.video_id,
                v.user_id AS author_id,
                COALESCE(v.category, '') AS category,
                COALESCE(v.duration, 0) AS video_duration,
                COALESCE(v.visit_count, 0) AS video_visit_count,
                COALESCE(v.likes_count, 0) AS video_likes_count,
                COALESCE(v.comment_count, 0) AS video_comment_count,
                COALESCE(v.share_count, 0) AS video_share_count,
                COALESCE(v.favorites_count, 0) AS video_favorites_count,
                v.created_at AS video_created_at,
                COALESCE(vf.quality_score, 0) AS video_quality_score,
                COALESCE(vf.popularity_score, 0) AS video_popularity_score,
                COALESCE(vf.ctr, 0) AS video_ctr,
                COALESCE(vf.finish_rate, 0) AS video_finish_rate,
                COALESCE(vf.like_rate, 0) AS video_like_rate,
                COALESCE(vf.comment_rate, 0) AS video_comment_rate,
                COALESCE(vf.share_rate, 0) AS video_share_rate,
                COALESCE(vf.favorite_rate, 0) AS video_favorite_rate,
                COALESCE(vf.avg_watch_duration, 0) AS video_avg_watch_duration
            FROM {TABLE_VIDEOS} v
            LEFT JOIN {TABLE_VIDEO_FEATURES} vf ON v.video_id = vf.video_id
            WHERE v.video_id IN ({placeholders})
        """
        df = self._read_sql(sql, params=video_ids)
        return df

    # =====================================================
    # Load author features
    # =====================================================

    def _load_author_features(self, conn, author_ids: list) -> pd.DataFrame:
        """Load features from author_scores table."""
        if not author_ids:
            return pd.DataFrame(columns=["author_id"])

        placeholders = ",".join(["%s"] * len(author_ids))
        sql = f"""
            SELECT
                author_id,
                COALESCE(quality_score, 0) AS author_quality_score,
                COALESCE(influence_score, 0) AS author_influence_score,
                COALESCE(overall_score, 0) AS author_overall_score,
                COALESCE(avg_engagement_rate, 0) AS author_avg_engagement_rate
            FROM {TABLE_AUTHOR_SCORES}
            WHERE author_id IN ({placeholders})
        """
        df = self._read_sql(sql, params=author_ids)
        return df

    # =====================================================
    # Load video hot scores
    # =====================================================

    def _load_video_hot_scores(self, conn, video_ids: list) -> pd.DataFrame:
        """Load the latest 24h hot score for videos."""
        if not video_ids:
            return pd.DataFrame(columns=["video_id"])

        placeholders = ",".join(["%s"] * len(video_ids))
        sql = f"""
            SELECT
                video_id,
                COALESCE(hot_score, 0) AS video_hot_score
            FROM {TABLE_VIDEO_HOT_SCORES}
            WHERE video_id IN ({placeholders})
                AND time_window = '24h'
        """
        df = self._read_sql(sql, params=video_ids)
        # Deduplicate (keep highest score)
        df = df.sort_values("video_hot_score", ascending=False).drop_duplicates("video_id", keep="first")
        return df

    # =====================================================
    # Load user behavior sequences (for DIN model)
    # =====================================================

    def load_user_behavior_sequences(self, conn=None, user_ids: list = None,
                                      max_len: int = 50) -> pd.DataFrame:
        """
        Load user's recent video interaction history for DIN sequence feature.
        Returns DataFrame with columns: [user_id, hist_video_ids]
        """
        should_close = False
        if conn is None:
            conn = self._get_connection()
            should_close = True

        try:
            if not user_ids:
                return pd.DataFrame(columns=["user_id", "hist_video_ids"])

            records = []
            for uid in user_ids:
                sql = f"""
                    SELECT video_id
                    FROM {TABLE_USER_BEHAVIORS}
                    WHERE user_id = %s AND behavior_type IN ('view', 'like', 'share')
                    ORDER BY behavior_time DESC
                    LIMIT %s
                """
                df = self._read_sql(sql, params=[uid, max_len])
                hist_ids = df["video_id"].tolist()
                records.append({"user_id": uid, "hist_video_ids": hist_ids})

            return pd.DataFrame(records)
        finally:
            if should_close:
                conn.close()

    # =====================================================
    # Generate prediction data (for online serving)
    # =====================================================

    def generate_prediction_data(self, user_id: int, video_ids: list) -> pd.DataFrame:
        """
        Generate feature data for online CTR prediction.
        Used by the FastAPI serving endpoint.
        """
        conn = self._get_connection()
        try:
            # Create base DataFrame
            df = pd.DataFrame({
                "user_id": [user_id] * len(video_ids),
                "video_id": video_ids,
                "is_click": 0,  # placeholder
                "exposure_time": datetime.now(),
            })

            # Join user features
            df_users = self._load_user_features(conn, [user_id])
            df = df.merge(df_users, on="user_id", how="left")

            # Join video features
            df_videos = self._load_video_features(conn, video_ids)
            df = df.merge(df_videos, on="video_id", how="left")

            # Join author features
            author_ids = df["author_id"].dropna().unique().tolist()
            if author_ids:
                df_authors = self._load_author_features(conn, author_ids)
                df = df.merge(df_authors, on="author_id", how="left")

            # Join hot scores
            df_hot = self._load_video_hot_scores(conn, video_ids)
            if not df_hot.empty:
                df = df.merge(df_hot, on="video_id", how="left")

            return df
        finally:
            conn.close()
