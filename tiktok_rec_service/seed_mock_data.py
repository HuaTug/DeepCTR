#!/usr/bin/env python3
"""
TikTok Recommendation System - Mock Data Generator
===================================================

This script generates realistic mock data for the DeepCTR recommendation model training.
It populates the following MySQL tables (matching Refactored-TikTok schemas):

  Core tables (must already exist, created by Go GORM AutoMigrate):
    - users                        (user profiles)
    - videos                       (video metadata)
    - user_behaviors               (user behavior logs: view/like/share/comment)

  Recommendation tables (created by recommendation_init.sql):
    - user_profiles                (user feature profiles for recommendation)
    - video_features               (video quality/interaction features)
    - author_scores                (author quality scores)
    - user_video_interactions      (detailed user-video interaction records)
    - recommendation_exposures     (labeled exposure data for CTR training)
    - video_hot_scores             (real-time video hot scores)
    - tag_video_mappings           (tag-video associations)
    - category_video_stats         (category-level statistics)

Usage:
    python seed_mock_data.py                          # Full mock data (default)
    python seed_mock_data.py --users 50 --videos 200  # Custom counts
    python seed_mock_data.py --skip-core              # Only fill recommendation tables
    python seed_mock_data.py --clean                  # Clean all recommendation data first

Environment variables (or edit config.py):
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta

import pymysql

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =====================================================
# Configuration
# =====================================================

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", 3307)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "TikTok@MySQL#2025!Secure"),
    "database": os.getenv("MYSQL_DB", "TikTok"),
    "charset": "utf8mb4",
}

# Categories matching the category_video_stats defaults
CATEGORIES = [
    "娱乐", "搞笑", "美食", "音乐", "舞蹈", "游戏", "知识",
    "科技", "生活", "时尚", "运动", "旅行", "萌宠", "二次元", "校园",
]

TAGS = [
    "搞笑", "日常", "美食", "手工", "科普", "Vlog", "开箱", "测评",
    "教程", "挑战", "热门", "推荐", "新人", "校园", "舞蹈", "翻唱",
    "旅行", "健身", "穿搭", "化妆", "萌宠", "游戏", "动漫", "摄影",
    "情感", "职场", "编程", "创业", "音乐", "电影",
]

SEXES = [0, 1, 2]  # 0: unknown, 1: male, 2: female
BEHAVIOR_TYPES = ["view", "like", "share", "comment"]
RECALL_SOURCES = ["hot", "cf", "content", "social", "new", "trending"]

# Weights for behavior probabilities (higher = more likely after view)
BEHAVIOR_WEIGHTS = {
    "view": 1.0,
    "like": 0.15,
    "comment": 0.05,
    "share": 0.03,
}

NOW = datetime.now()


# =====================================================
# Utility Functions
# =====================================================


def get_connection():
    """Get MySQL connection."""
    return pymysql.connect(**MYSQL_CONFIG, cursorclass=pymysql.cursors.DictCursor)


def random_datetime(start_days_ago=30, end_days_ago=0):
    """Generate a random datetime within a range."""
    start = NOW - timedelta(days=start_days_ago)
    end = NOW - timedelta(days=end_days_ago)
    delta = (end - start).total_seconds()
    random_seconds = random.uniform(0, delta)
    return start + timedelta(seconds=random_seconds)


def random_time_window():
    """Return a random recent datetime string for SQL."""
    dt = random_datetime(30, 0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def weighted_choice(items, weights):
    """Weighted random choice from a list."""
    total = sum(weights)
    r = random.uniform(0, total)
    cumulative = 0
    for item, weight in zip(items, weights):
        cumulative += weight
        if r <= cumulative:
            return item
    return items[-1]


# =====================================================
# Data Generators
# =====================================================


def generate_users(conn, n_users=100):
    """
    Generate mock user records in the `users` table.
    Matches Go model: User struct in cmd/model/user.go
    """
    logger.info(f"Generating {n_users} users...")

    # Check how many users already exist
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM users")
        existing = cur.fetchone()["cnt"]

    if existing >= n_users:
        logger.info(f"  Already have {existing} users, skipping user generation")
        with conn.cursor() as cur:
            cur.execute(f"SELECT user_id FROM users ORDER BY user_id LIMIT {n_users}")
            return [row["user_id"] for row in cur.fetchall()]

    start_id = existing + 1
    to_create = n_users - existing

    first_names = ["小明", "小红", "小刚", "小丽", "小华", "阿杰", "阿美", "大壮", "小凡", "小婷",
                   "天天", "星星", "乐乐", "果果", "豆豆", "文文", "武武", "秀秀", "强强", "慧慧"]
    last_names = ["张", "李", "王", "刘", "陈", "杨", "赵", "黄", "周", "吴",
                  "徐", "孙", "马", "朱", "胡", "林", "何", "高", "罗", "郑"]

    users = []
    for i in range(to_create):
        uid = start_id + i
        name = random.choice(last_names) + random.choice(first_names)
        username = f"user_{uid}_{name}"
        # Use a simple hashed password
        password = hashlib.md5(f"password_{uid}".encode()).hexdigest()
        sex = random.choice(SEXES)
        email = f"user{uid}@tiktok.mock.com"

        following = random.randint(0, 500)
        follower = random.randint(0, 10000)
        like_count = random.randint(0, 50000)
        video_count = random.randint(0, 200)

        created_at = random_datetime(180, 30)
        updated_at = random_datetime(30, 0)

        users.append((
            uid, username, password, email, sex,
            f"https://avatar.mock/{uid}.jpg", "",
            following, follower, like_count, video_count,
            1,  # status: normal
            created_at.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        ))

    if users:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT IGNORE INTO users
                (user_id, user_name, password, email, sex, avatar_url, bio,
                 following_count, follower_count, like_count, video_count,
                 status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, users)
        conn.commit()
        logger.info(f"  Inserted {len(users)} new users")

    with conn.cursor() as cur:
        cur.execute(f"SELECT user_id FROM users ORDER BY user_id LIMIT {n_users}")
        return [row["user_id"] for row in cur.fetchall()]


def generate_videos(conn, user_ids, n_videos=500):
    """
    Generate mock video records in the `videos` table.
    Matches Go model: Video struct in cmd/model/video.go
    """
    logger.info(f"Generating {n_videos} videos...")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM videos")
        existing = cur.fetchone()["cnt"]

    if existing >= n_videos:
        logger.info(f"  Already have {existing} videos, skipping video generation")
        with conn.cursor() as cur:
            cur.execute(f"SELECT video_id FROM videos WHERE deleted_at IS NULL ORDER BY video_id LIMIT {n_videos}")
            return [row["video_id"] for row in cur.fetchall()]

    start_id = existing + 1
    to_create = n_videos - existing

    # Use top users as active creators (Pareto: 20% of users create 80% of content)
    creator_ids = random.sample(user_ids, max(1, len(user_ids) // 5))

    title_prefixes = [
        "今天来做", "教你", "挑战", "开箱", "测评", "日常", "分享",
        "终于", "震惊", "超级", "最新", "必看", "盘点", "3分钟学会",
    ]
    title_suffixes = [
        "超好吃的蛋糕", "这个很有趣", "太厉害了", "你学会了吗",
        "不看后悔", "真的可以", "绝了", "笑死我了",
        "最强攻略", "新手必看", "高手进阶", "生活技巧",
    ]

    videos = []
    for i in range(to_create):
        vid = start_id + i
        author_id = random.choice(creator_ids)
        category = random.choice(CATEGORIES)
        duration = random.choices(
            [random.randint(5, 15), random.randint(15, 60),
             random.randint(60, 180), random.randint(180, 600)],
            weights=[0.3, 0.4, 0.2, 0.1]
        )[0]

        title = random.choice(title_prefixes) + " " + random.choice(title_suffixes)
        labels = ",".join(random.sample(TAGS, random.randint(1, 4)))

        visit_count = int(random.paretovariate(1.5) * 100)
        likes_count = int(visit_count * random.uniform(0.01, 0.15))
        comment_count = int(visit_count * random.uniform(0.001, 0.03))
        share_count = int(visit_count * random.uniform(0.001, 0.02))
        favorites_count = int(visit_count * random.uniform(0.005, 0.05))

        created_at = random_datetime(60, 1)

        videos.append((
            vid, author_id,
            f"https://video.mock/{vid}.mp4",
            f"https://cover.mock/{vid}.jpg",
            title, title,
            duration, 1080, 1920, duration * 500000,
            visit_count, share_count, likes_count, favorites_count,
            comment_count, visit_count,
            1,  # open: public
            1,  # audit_status: approved
            labels, category,
            created_at.strftime("%Y-%m-%d %H:%M:%S"),
            created_at.strftime("%Y-%m-%d %H:%M:%S"),
        ))

    if videos:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT IGNORE INTO videos
                (video_id, user_id, video_url, cover_url, title, description,
                 duration, width, height, file_size,
                 visit_count, share_count, likes_count, favorites_count,
                 comment_count, history_count,
                 open, audit_status, label_names, category,
                 created_at, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, videos)
        conn.commit()
        logger.info(f"  Inserted {len(videos)} new videos")

    with conn.cursor() as cur:
        cur.execute(f"SELECT video_id FROM videos WHERE deleted_at IS NULL ORDER BY video_id LIMIT {n_videos}")
        return [row["video_id"] for row in cur.fetchall()]


def generate_user_behaviors(conn, user_ids, video_ids, n_behaviors_per_user=50):
    """
    Generate user behavior records in `user_behaviors` table.
    Matches Go model: UserBehavior struct in cmd/model/user.go

    Each user has ~N behaviors (view/like/share/comment).
    """
    logger.info(f"Generating user behaviors ({n_behaviors_per_user} per user avg)...")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM user_behaviors")
        existing = cur.fetchone()["cnt"]

    if existing > len(user_ids) * 10:
        logger.info(f"  Already have {existing} behaviors, skipping")
        return

    behaviors = []
    batch_size = 5000

    for uid in user_ids:
        # Each user views a random subset of videos
        n_views = random.randint(n_behaviors_per_user // 2, n_behaviors_per_user * 2)
        viewed_videos = random.sample(video_ids, min(n_views, len(video_ids)))

        for vid in viewed_videos:
            behavior_time = random_datetime(30, 0)
            # Always a "view" event
            behaviors.append((
                uid, vid, "view",
                behavior_time.strftime("%Y-%m-%d %H:%M:%S"),
                behavior_time.strftime("%Y-%m-%d %H:%M:%S"),
            ))

            # Probabilistic follow-up behaviors
            if random.random() < BEHAVIOR_WEIGHTS["like"]:
                like_time = behavior_time + timedelta(seconds=random.randint(5, 60))
                behaviors.append((uid, vid, "like",
                                  like_time.strftime("%Y-%m-%d %H:%M:%S"),
                                  like_time.strftime("%Y-%m-%d %H:%M:%S")))

            if random.random() < BEHAVIOR_WEIGHTS["comment"]:
                comment_time = behavior_time + timedelta(seconds=random.randint(10, 120))
                behaviors.append((uid, vid, "comment",
                                  comment_time.strftime("%Y-%m-%d %H:%M:%S"),
                                  comment_time.strftime("%Y-%m-%d %H:%M:%S")))

            if random.random() < BEHAVIOR_WEIGHTS["share"]:
                share_time = behavior_time + timedelta(seconds=random.randint(10, 180))
                behaviors.append((uid, vid, "share",
                                  share_time.strftime("%Y-%m-%d %H:%M:%S"),
                                  share_time.strftime("%Y-%m-%d %H:%M:%S")))

        # Insert in batches
        if len(behaviors) >= batch_size:
            _insert_behaviors_batch(conn, behaviors)
            behaviors = []

    if behaviors:
        _insert_behaviors_batch(conn, behaviors)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM user_behaviors")
        total = cur.fetchone()["cnt"]
    logger.info(f"  Total user behaviors: {total}")


def _insert_behaviors_batch(conn, behaviors):
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT IGNORE INTO user_behaviors
            (user_id, video_id, behavior_type, behavior_time, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, behaviors)
    conn.commit()


# =====================================================
# Recommendation Tables Data Generators
# =====================================================


def generate_user_profiles(conn, user_ids):
    """
    Fill `user_profiles` table with computed user features.
    Maps to: UserProfile struct in cmd/model/recommendation.go
    """
    logger.info(f"Generating user profiles for {len(user_ids)} users...")

    profiles = []
    for uid in user_ids:
        avg_watch = random.uniform(10, 120)
        avg_completion = random.uniform(0.2, 0.95)
        like_rate = random.uniform(0.02, 0.25)
        comment_rate = random.uniform(0.005, 0.08)
        share_rate = random.uniform(0.003, 0.05)
        total_views = random.randint(50, 10000)
        total_likes = int(total_views * like_rate)
        total_comments = int(total_views * comment_rate)
        total_shares = int(total_views * share_rate)
        user_level = random.choices([1, 2, 3, 4, 5], weights=[0.3, 0.3, 0.2, 0.15, 0.05])[0]
        content_pref = random.randint(1, 5)
        duration_pref = random.randint(1, 3)

        # Interest tags: pick 3-6 random categories with weights
        n_interests = random.randint(3, 6)
        interest_cats = random.sample(CATEGORIES, n_interests)
        interest_tags = {cat: round(random.uniform(0.3, 1.0), 2) for cat in interest_cats}

        # Category preference
        cat_pref = {cat: round(random.uniform(0.1, 1.0), 2) for cat in interest_cats}

        # Active time slots
        active_slots = sorted(random.sample(range(6, 24), random.randint(3, 8)))

        last_active = random_datetime(3, 0)

        profiles.append((
            uid,
            json.dumps(interest_tags, ensure_ascii=False),
            json.dumps(cat_pref, ensure_ascii=False),
            json.dumps({}),  # author_preference
            json.dumps({}),  # topic_preference
            json.dumps(active_slots),
            round(avg_watch, 2),
            round(avg_completion, 4),
            round(like_rate, 4),
            round(comment_rate, 4),
            round(share_rate, 4),
            total_views, total_likes, total_comments, total_shares,
            user_level, content_pref, duration_pref,
            last_active.strftime("%Y-%m-%d %H:%M:%S"),
        ))

    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO user_profiles
            (user_id, interest_tags, category_preference, author_preference,
             topic_preference, active_time_slots,
             avg_watch_duration, avg_completion_rate, like_rate, comment_rate, share_rate,
             total_view_count, total_like_count, total_comment_count, total_share_count,
             user_level, content_quality_pref, video_duration_pref, last_active_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                interest_tags=VALUES(interest_tags),
                avg_watch_duration=VALUES(avg_watch_duration),
                avg_completion_rate=VALUES(avg_completion_rate),
                like_rate=VALUES(like_rate),
                comment_rate=VALUES(comment_rate),
                share_rate=VALUES(share_rate),
                total_view_count=VALUES(total_view_count),
                user_level=VALUES(user_level),
                last_active_at=VALUES(last_active_at)
        """, profiles)
    conn.commit()
    logger.info(f"  Upserted {len(profiles)} user profiles")


def generate_video_features(conn, video_ids):
    """
    Fill `video_features` table with computed video quality metrics.
    Maps to: VideoFeature struct in cmd/model/recommendation.go
    """
    logger.info(f"Generating video features for {len(video_ids)} videos...")

    features = []
    for vid in video_ids:
        quality = round(random.uniform(2.0, 9.5), 2)
        popularity = round(random.paretovariate(1.5) * 10, 2)
        freshness = round(random.uniform(1.0, 10.0), 2)
        ctr = round(random.uniform(0.01, 0.25), 6)
        finish_rate = round(random.uniform(0.1, 0.95), 4)
        like_rate = round(random.uniform(0.01, 0.20), 4)
        comment_rate = round(random.uniform(0.001, 0.05), 4)
        share_rate = round(random.uniform(0.001, 0.03), 4)
        favorite_rate = round(random.uniform(0.005, 0.08), 4)
        interact_score = round(quality * 0.3 + popularity * 0.3 + ctr * 100 * 0.2 + finish_rate * 10 * 0.2, 2)
        avg_watch = round(random.uniform(5, 120), 2)
        exposure_count = random.randint(100, 100000)
        click_count = int(exposure_count * ctr)
        author_score = round(random.uniform(3.0, 9.0), 2)
        is_high_quality = 1 if quality > 7.0 else 0

        features.append((
            vid, quality, popularity, freshness, ctr, finish_rate,
            like_rate, comment_rate, share_rate, favorite_rate,
            interact_score, avg_watch, exposure_count, click_count,
            author_score, is_high_quality,
        ))

    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO video_features
            (video_id, quality_score, popularity_score, freshness_score, ctr, finish_rate,
             like_rate, comment_rate, share_rate, favorite_rate,
             interact_score, avg_watch_duration, exposure_count, click_count,
             author_score, is_high_quality)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                quality_score=VALUES(quality_score),
                popularity_score=VALUES(popularity_score),
                ctr=VALUES(ctr),
                finish_rate=VALUES(finish_rate),
                like_rate=VALUES(like_rate),
                exposure_count=VALUES(exposure_count),
                click_count=VALUES(click_count)
        """, features)
    conn.commit()
    logger.info(f"  Upserted {len(features)} video features")


def generate_author_scores(conn, user_ids, video_ids):
    """
    Fill `author_scores` table for video creators.
    Maps to: AuthorScore struct in cmd/model/recommendation.go
    """
    logger.info("Generating author scores...")

    # Get distinct authors from videos
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT user_id FROM videos WHERE deleted_at IS NULL")
        author_ids = [row["user_id"] for row in cur.fetchall()]

    if not author_ids:
        author_ids = random.sample(user_ids, max(1, len(user_ids) // 5))

    scores = []
    for aid in author_ids:
        quality = round(random.uniform(3.0, 9.5), 2)
        activity = round(random.uniform(2.0, 9.0), 2)
        influence = round(random.uniform(1.0, 9.0), 2)
        growth = round(random.uniform(1.0, 8.0), 2)
        overall = round(quality * 0.3 + activity * 0.2 + influence * 0.3 + growth * 0.2, 2)
        total_videos = random.randint(1, 200)
        avg_quality = round(random.uniform(3.0, 8.5), 2)
        avg_views = round(random.paretovariate(1.5) * 500, 2)
        avg_engagement = round(random.uniform(0.01, 0.15), 4)
        level = min(10, max(1, int(overall)))
        is_verified = 1 if overall > 7.0 and random.random() > 0.5 else 0

        last_publish = random_datetime(15, 0)

        scores.append((
            aid, quality, activity, influence, growth, overall,
            total_videos, avg_quality, avg_views, avg_engagement,
            last_publish.strftime("%Y-%m-%d %H:%M:%S"),
            level, is_verified,
        ))

    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO author_scores
            (author_id, quality_score, activity_score, influence_score, growth_score, overall_score,
             total_videos, avg_video_quality, avg_video_views, avg_engagement_rate,
             last_publish_at, level, is_verified)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                quality_score=VALUES(quality_score),
                influence_score=VALUES(influence_score),
                overall_score=VALUES(overall_score),
                avg_engagement_rate=VALUES(avg_engagement_rate),
                level=VALUES(level)
        """, scores)
    conn.commit()
    logger.info(f"  Upserted {len(scores)} author scores")


def generate_user_video_interactions(conn, user_ids, video_ids):
    """
    Fill `user_video_interactions` table with detailed interaction records.
    Maps to: UserVideoInteraction struct in cmd/model/recommendation.go
    """
    logger.info("Generating user_video_interactions...")

    interactions = []
    batch_size = 5000

    for uid in user_ids:
        # Each user interacts with N random videos
        n_interact = random.randint(10, min(80, len(video_ids)))
        interacted_videos = random.sample(video_ids, n_interact)

        for vid in interacted_videos:
            impression = random.randint(1, 5)
            click = random.randint(0, impression)
            watch_time = random.randint(3, 300) if click > 0 else 0
            max_progress = round(random.uniform(0.05, 1.0), 4) if click > 0 else 0
            replay = random.randint(0, 3) if max_progress > 0.8 else 0
            is_liked = 1 if random.random() < 0.15 else 0
            is_favorited = 1 if random.random() < 0.05 else 0
            is_shared = 1 if random.random() < 0.03 else 0
            comment_count = 1 if random.random() < 0.05 else 0

            # Engagement score: weighted sum
            engagement = round(
                click * 1.0 + watch_time * 0.01 + max_progress * 3 +
                replay * 2 + is_liked * 3 + is_favorited * 4 +
                is_shared * 5 + comment_count * 4,
                4
            )

            first_interact = random_datetime(30, 1)
            last_interact = first_interact + timedelta(hours=random.randint(0, 72))

            interactions.append((
                uid, vid, impression, click, watch_time, max_progress,
                watch_time if click > 0 else 0,  # last_watch_position
                replay, is_liked, is_favorited, is_shared, comment_count,
                engagement,
                first_interact.strftime("%Y-%m-%d %H:%M:%S"),
                last_interact.strftime("%Y-%m-%d %H:%M:%S"),
            ))

            if len(interactions) >= batch_size:
                _insert_interactions_batch(conn, interactions)
                interactions = []

    if interactions:
        _insert_interactions_batch(conn, interactions)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM user_video_interactions")
        total = cur.fetchone()["cnt"]
    logger.info(f"  Total user_video_interactions: {total}")


def _insert_interactions_batch(conn, interactions):
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO user_video_interactions
            (user_id, video_id, impression_count, click_count, total_watch_time,
             max_watch_progress, last_watch_position, replay_count,
             is_liked, is_favorited, is_shared, comment_count,
             engagement_score, first_interact_at, last_interact_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                impression_count = impression_count + VALUES(impression_count),
                click_count = click_count + VALUES(click_count),
                total_watch_time = VALUES(total_watch_time),
                max_watch_progress = GREATEST(max_watch_progress, VALUES(max_watch_progress)),
                is_liked = VALUES(is_liked),
                is_favorited = VALUES(is_favorited),
                is_shared = VALUES(is_shared),
                engagement_score = VALUES(engagement_score),
                last_interact_at = VALUES(last_interact_at)
        """, interactions)
    conn.commit()


def generate_recommendation_exposures(conn, user_ids, video_ids, n_per_user=30):
    """
    Fill `recommendation_exposures` table with labeled exposure data.
    THIS IS THE CORE TRAINING DATA for DeepCTR models.

    Click probability is computed from observable features so the model
    can learn meaningful patterns:
      - video quality & popularity → higher quality videos get more clicks
      - user activity level → active users click more
      - category match → users click more on preferred categories
      - time context → certain hours have higher engagement
      - position bias → earlier positions get more clicks
    """
    logger.info(f"Generating recommendation exposures ({n_per_user} per user)...")

    # ---- Load feature tables into memory for feature-dependent click model ----
    # Video features
    video_feat = {}
    with conn.cursor() as cur:
        cur.execute("SELECT video_id, quality_score, popularity_score FROM video_features")
        for row in cur.fetchall():
            video_feat[row["video_id"]] = row

    # Video metadata (category, duration, author)
    video_meta = {}
    with conn.cursor() as cur:
        cur.execute("SELECT video_id, user_id AS author_id, category, duration FROM videos WHERE deleted_at IS NULL")
        for row in cur.fetchall():
            video_meta[row["video_id"]] = row

    # User profiles
    user_prof = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT user_id, avg_watch_duration, avg_completion_rate, like_rate,
                   total_view_count, user_level, interest_tags
            FROM user_profiles
        """)
        for row in cur.fetchall():
            user_prof[row["user_id"]] = row

    # Author scores
    author_sc = {}
    with conn.cursor() as cur:
        cur.execute("SELECT author_id, overall_score FROM author_scores")
        for row in cur.fetchall():
            author_sc[row["author_id"]] = row

    # User following/follower counts
    user_base = {}
    with conn.cursor() as cur:
        cur.execute("SELECT user_id, following_count, follower_count FROM users")
        for row in cur.fetchall():
            user_base[row["user_id"]] = row

    logger.info("  Feature tables loaded into memory for click model")

    # ---- Helper: compute click probability from features ----
    def _compute_click_prob(uid, vid, position, hour):
        # Use logistic model: logit = sum of weighted signals, then sigmoid
        logit = -2.5  # base logit (sigmoid(-2.5) ≈ 0.075, base CTR ~7.5%)

        vf = video_feat.get(vid, {})
        vm = video_meta.get(vid, {})
        up = user_prof.get(uid, {})

        # 1. Video quality: strongest signal (quality_score: 2-9.5)
        quality = float(vf.get("quality_score", 5.0))
        logit += (quality - 5.0) / 2.5 * 0.8  # range: -0.96 to +1.44

        # 2. Video popularity (log-transformed)
        pop = float(vf.get("popularity_score", 10.0))
        logit += min(math.log1p(pop) / 5.0, 1.0) * 0.5

        # 3. Category match: very strong signal
        video_cat = vm.get("category", "")
        interest_str = up.get("interest_tags", "{}")
        try:
            interests = json.loads(interest_str) if isinstance(interest_str, str) else interest_str
        except (json.JSONDecodeError, TypeError):
            interests = {}
        if video_cat and video_cat in interests:
            cat_weight = float(interests[video_cat])
            logit += 0.6 + cat_weight * 0.8  # match: +0.6 to +1.4
        else:
            logit -= 0.5  # no match: penalty

        # 4. User activity level
        views = int(up.get("total_view_count", 100))
        logit += min(math.log1p(views) / math.log1p(10000), 1.0) * 0.4

        # 5. User level (1-5)
        level = int(up.get("user_level", 1))
        logit += (level - 3) / 2.0 * 0.3

        # 6. Author quality
        author_id = vm.get("author_id")
        if author_id:
            a_score = float(author_sc.get(author_id, {}).get("overall_score", 5.0))
            logit += (a_score - 5.0) / 3.0 * 0.4

        # 7. Position bias (moderate decay)
        logit -= 0.02 * position

        # 8. Time-of-day
        if 18 <= hour <= 23:
            logit += 0.2
        elif 0 <= hour <= 6:
            logit -= 0.3

        # 9. Short video preference
        vid_dur = int(vm.get("duration", 30))
        if vid_dur <= 15:
            logit += 0.2
        elif vid_dur >= 180:
            logit -= 0.2

        # Sigmoid with moderate noise
        noise = random.gauss(0, 0.3)
        prob = 1.0 / (1.0 + math.exp(-(logit + noise)))

        return max(0.01, min(prob, 0.75))

    # ---- Generate exposures ----
    exposures = []
    batch_size = 5000
    total_inserted = 0

    for uid in user_ids:
        n_expose = random.randint(n_per_user // 2, n_per_user * 2)
        exposed_videos = random.choices(video_ids, k=n_expose)

        for pos, vid in enumerate(exposed_videos):
            recall_source = random.choice(RECALL_SOURCES)
            exposure_time = random_datetime(30, 0)
            hour = exposure_time.hour

            # Feature-dependent click probability
            click_prob = _compute_click_prob(uid, vid, pos, hour)
            score = round(click_prob, 6)
            is_clicked = 1 if random.random() < click_prob else 0

            # Post-click behaviors (probability also depends on video quality)
            if is_clicked:
                vf = video_feat.get(vid, {})
                q = float(vf.get("quality_score", 5.0)) / 10.0
                watch_duration = random.randint(5, 300)
                completion_rate = round(random.uniform(0.1, min(1.0, 0.3 + q * 0.7)), 4)
                is_liked = 1 if random.random() < (0.1 + q * 0.15) else 0
                is_commented = 1 if random.random() < (0.02 + q * 0.05) else 0
                is_shared = 1 if random.random() < (0.01 + q * 0.04) else 0
                is_favorited = 1 if random.random() < (0.02 + q * 0.05) else 0
            else:
                watch_duration = 0
                completion_rate = 0.0
                is_liked = 0
                is_commented = 0
                is_shared = 0
                is_favorited = 0

            request_id = f"req_{uid}_{int(exposure_time.timestamp())}"

            exposures.append((
                uid, vid, recall_source, pos, score,
                is_clicked, is_liked, is_commented, is_shared, is_favorited,
                watch_duration, completion_rate,
                exposure_time.strftime("%Y-%m-%d %H:%M:%S"),
                request_id,
            ))

            if len(exposures) >= batch_size:
                _insert_exposures_batch(conn, exposures)
                total_inserted += len(exposures)
                exposures = []

    if exposures:
        _insert_exposures_batch(conn, exposures)
        total_inserted += len(exposures)

    # Log statistics
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS cnt, AVG(is_clicked) AS click_rate FROM recommendation_exposures")
        stats = cur.fetchone()
    logger.info(f"  Total exposures: {stats['cnt']}, Click rate: {stats['click_rate']:.4f}")


def _insert_exposures_batch(conn, exposures):
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO recommendation_exposures
            (user_id, video_id, recall_source, position, score,
             is_clicked, is_liked, is_commented, is_shared, is_favorited,
             watch_duration, completion_rate, exposure_time, request_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, exposures)
    conn.commit()


def generate_video_hot_scores(conn, video_ids):
    """
    Fill `video_hot_scores` table with hot score data across time windows.
    Maps to: VideoHotScore struct in cmd/model/recommendation.go
    """
    logger.info(f"Generating video hot scores for {len(video_ids)} videos...")

    time_windows = ["1h", "6h", "24h", "7d"]
    records = []

    for vid in video_ids:
        for tw in time_windows:
            view_count = random.randint(10, 50000)
            like_count = int(view_count * random.uniform(0.02, 0.15))
            comment_count = int(view_count * random.uniform(0.005, 0.03))
            share_count = int(view_count * random.uniform(0.002, 0.02))
            favorite_count = int(view_count * random.uniform(0.005, 0.05))

            # Hot score formula
            hot_score = round(
                view_count * 1.0 + like_count * 3.0 + comment_count * 5.0 +
                share_count * 7.0 + favorite_count * 4.0,
                2
            )

            window_end = NOW
            if tw == "1h":
                window_start = NOW - timedelta(hours=1)
            elif tw == "6h":
                window_start = NOW - timedelta(hours=6)
            elif tw == "24h":
                window_start = NOW - timedelta(hours=24)
            else:
                window_start = NOW - timedelta(days=7)

            records.append((
                vid, tw, view_count, like_count, comment_count,
                share_count, favorite_count, hot_score, 0,
                window_start.strftime("%Y-%m-%d %H:%M:%S"),
                window_end.strftime("%Y-%m-%d %H:%M:%S"),
            ))

    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO video_hot_scores
            (video_id, time_window, view_count, like_count, comment_count,
             share_count, favorite_count, hot_score, `rank`,
             window_start, window_end)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                view_count=VALUES(view_count),
                like_count=VALUES(like_count),
                hot_score=VALUES(hot_score)
        """, records)
    conn.commit()

    # Update ranks within each time window
    for tw in time_windows:
        with conn.cursor() as cur:
            cur.execute("SET @rank := 0")
            cur.execute("""
                UPDATE video_hot_scores
                SET `rank` = (@rank := @rank + 1)
                WHERE time_window = %s
                ORDER BY hot_score DESC
            """, (tw,))
    conn.commit()

    logger.info(f"  Generated hot scores for {len(video_ids)} videos × {len(time_windows)} windows")


def generate_tag_video_mappings(conn, video_ids):
    """
    Fill `tag_video_mappings` table.
    Maps to: TagVideoMapping struct in cmd/model/recommendation.go
    """
    logger.info("Generating tag-video mappings...")

    mappings = []
    for vid in video_ids:
        n_tags = random.randint(1, 5)
        chosen_tags = random.sample(TAGS, n_tags)
        for tag in chosen_tags:
            weight = round(random.uniform(0.5, 1.0), 4)
            source = random.choice(["manual", "ai", "user"])
            mappings.append((tag, vid, weight, source))

    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO tag_video_mappings (tag_name, video_id, weight, source)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE weight=VALUES(weight)
        """, mappings)
    conn.commit()
    logger.info(f"  Inserted {len(mappings)} tag-video mappings")


def update_category_video_stats(conn):
    """
    Update `category_video_stats` table by aggregating from videos.
    """
    logger.info("Updating category video stats...")

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE category_video_stats cvs
            JOIN (
                SELECT
                    category,
                    COUNT(*) AS total_videos,
                    SUM(visit_count) AS total_views,
                    SUM(likes_count) AS total_likes
                FROM videos
                WHERE deleted_at IS NULL AND open = 1 AND category != ''
                GROUP BY category
            ) v ON cvs.category = v.category COLLATE utf8mb4_unicode_ci
            SET
                cvs.total_videos = v.total_videos,
                cvs.total_views = v.total_views,
                cvs.total_likes = v.total_likes,
                cvs.hot_score = v.total_views * 1 + v.total_likes * 3,
                cvs.daily_new_videos = FLOOR(v.total_videos / 30)
        """)
    conn.commit()
    logger.info("  Category stats updated")


# =====================================================
# Clean functions
# =====================================================


def clean_recommendation_data(conn):
    """Clean all recommendation-specific tables (keeps core users/videos)."""
    logger.info("Cleaning recommendation data tables...")
    tables = [
        "recommendation_exposures",
        "user_video_interactions",
        "user_profiles",
        "video_features",
        "author_scores",
        "video_hot_scores",
        "tag_video_mappings",
        "video_similarities",
        "video_embeddings",
        "user_embeddings",
        "negative_feedbacks",
        "recommendation_request_logs",
    ]
    with conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"DELETE FROM `{table}`")
                logger.info(f"  Cleaned {table}")
            except Exception as e:
                logger.warning(f"  Failed to clean {table}: {e}")
    conn.commit()


def clean_all_data(conn):
    """Clean all data including core tables."""
    logger.info("Cleaning ALL data tables...")
    clean_recommendation_data(conn)
    with conn.cursor() as cur:
        for table in ["user_behaviors", "videos", "users"]:
            try:
                cur.execute(f"DELETE FROM `{table}`")
                logger.info(f"  Cleaned {table}")
            except Exception as e:
                logger.warning(f"  Failed to clean {table}: {e}")
    conn.commit()


# =====================================================
# Main
# =====================================================


def print_summary(conn):
    """Print a summary of all table counts."""
    tables = [
        "users", "videos", "user_behaviors",
        "user_profiles", "video_features", "author_scores",
        "user_video_interactions", "recommendation_exposures",
        "video_hot_scores", "tag_video_mappings", "category_video_stats",
    ]
    logger.info("=" * 60)
    logger.info("DATA SUMMARY")
    logger.info("=" * 60)
    with conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) AS cnt FROM `{table}`")
                count = cur.fetchone()["cnt"]
                logger.info(f"  {table:40s} {count:>10,} rows")
            except Exception:
                logger.info(f"  {table:40s} {'N/A':>10s}")

    # Recommendation exposures stats
    with conn.cursor() as cur:
        try:
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(is_clicked) AS clicks,
                    AVG(is_clicked) AS click_rate,
                    AVG(CASE WHEN is_clicked=1 THEN completion_rate END) AS avg_completion
                FROM recommendation_exposures
            """)
            stats = cur.fetchone()
            logger.info("")
            logger.info("  RECOMMENDATION EXPOSURE STATS:")
            logger.info(f"    Total exposures:      {stats['total']:,}")
            logger.info(f"    Total clicks:          {stats['clicks']:,}")
            logger.info(f"    Click rate:            {stats['click_rate']:.4f}")
            logger.info(f"    Avg completion (click): {stats['avg_completion']:.4f}")
        except Exception:
            pass
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate mock data for TikTok DeepCTR recommendation system"
    )
    parser.add_argument("--users", type=int, default=100, help="Number of users (default: 100)")
    parser.add_argument("--videos", type=int, default=500, help="Number of videos (default: 500)")
    parser.add_argument("--behaviors-per-user", type=int, default=50,
                        help="Avg behaviors per user (default: 50)")
    parser.add_argument("--exposures-per-user", type=int, default=80,
                        help="Avg exposures per user (default: 80)")
    parser.add_argument("--skip-core", action="store_true",
                        help="Skip core table generation (users/videos/behaviors)")
    parser.add_argument("--clean", action="store_true",
                        help="Clean recommendation tables before generating")
    parser.add_argument("--clean-all", action="store_true",
                        help="Clean ALL tables before generating (including users/videos)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("TikTok DeepCTR Mock Data Generator")
    logger.info("=" * 60)
    logger.info(f"Config: {args.users} users, {args.videos} videos")
    logger.info(f"MySQL: {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")
    logger.info("")

    conn = get_connection()
    try:
        start_time = time.time()

        # Clean if requested
        if args.clean_all:
            clean_all_data(conn)
        elif args.clean:
            clean_recommendation_data(conn)

        # Step 1: Core tables (users, videos, user_behaviors)
        if not args.skip_core:
            logger.info("=" * 60)
            logger.info("STEP 1: Core tables (users, videos, user_behaviors)")
            logger.info("=" * 60)
            user_ids = generate_users(conn, args.users)
            video_ids = generate_videos(conn, user_ids, args.videos)
            generate_user_behaviors(conn, user_ids, video_ids, args.behaviors_per_user)
        else:
            logger.info("Skipping core table generation, reading existing data...")
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM users ORDER BY user_id")
                user_ids = [row["user_id"] for row in cur.fetchall()]
                cur.execute("SELECT video_id FROM videos WHERE deleted_at IS NULL ORDER BY video_id")
                video_ids = [row["video_id"] for row in cur.fetchall()]
            if not user_ids or not video_ids:
                logger.error("No existing users/videos found! Run without --skip-core first.")
                sys.exit(1)
            logger.info(f"  Found {len(user_ids)} users, {len(video_ids)} videos")

        # Step 2: Recommendation tables
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: Recommendation tables")
        logger.info("=" * 60)
        generate_user_profiles(conn, user_ids)
        generate_video_features(conn, video_ids)
        generate_author_scores(conn, user_ids, video_ids)
        generate_user_video_interactions(conn, user_ids, video_ids)
        generate_recommendation_exposures(conn, user_ids, video_ids, args.exposures_per_user)
        generate_video_hot_scores(conn, video_ids)
        generate_tag_video_mappings(conn, video_ids)
        update_category_video_stats(conn)

        elapsed = time.time() - start_time
        logger.info("")
        logger.info(f"Data generation completed in {elapsed:.1f} seconds")
        logger.info("")

        # Print summary
        print_summary(conn)

        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. cd /data/workspace/TikTok/DeepCTR/tiktok_rec_service")
        logger.info("  2. python train.py --model deepfm   # Train DeepFM model")
        logger.info("  3. python serve.py                  # Start CTR prediction service")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
