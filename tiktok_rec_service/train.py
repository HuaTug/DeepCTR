"""
Training script for TikTok Video Recommendation models.
Supports: DeepFM, DIN, MMoE (multi-task).
Uses DeepCTR library with features extracted from Refactored-TikTok tables.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Add parent directory to path to import deepctr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DENSE_FEATURES,
    MODEL_CONFIG,
    SERVING_CONFIG,
    SPARSE_FEATURES,
    TRAINING_CONFIG,
)
from data_generator import DataGenerator
from deepctr.models import DeepFM
from deepctr.models.multitask.mmoe import MMoE
from deepctr.models.sequence.din import DIN
from feature_engineering import FeatureProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_deepfm(df_train, df_val, feature_processor: FeatureProcessor, model_dir: str):
    """
    Train DeepFM model for CTR prediction.

    DeepFM = FM (2nd-order feature interactions) + DNN (high-order feature interactions)
    Good for: capturing both low-order and high-order feature interactions.

    Input features mapped to DB tables:
      - Sparse: user_id, video_id, author_id, category, user_sex, user_level, etc.
      - Dense: user_avg_watch_duration, video_quality_score, video_ctr, etc.
    """
    cfg = MODEL_CONFIG["deepfm"]

    feature_columns = feature_processor.get_feature_columns()
    feature_names = feature_processor.get_feature_names()

    # Build model
    model = DeepFM(
        linear_feature_columns=feature_columns,
        dnn_feature_columns=feature_columns,
        dnn_hidden_units=cfg["dnn_hidden_units"],
        dnn_dropout=cfg["dnn_dropout"],
        l2_reg_embedding=cfg["l2_reg_embedding"],
        l2_reg_dnn=cfg["l2_reg_dnn"],
        task="binary",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TRAINING_CONFIG["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["AUC", "binary_crossentropy"],
    )

    # Prepare input
    train_input = feature_processor.get_model_input(df_train)
    val_input = feature_processor.get_model_input(df_val)
    train_label = df_train["is_click"].values
    val_label = df_val["is_click"].values

    # Callbacks
    model_path = os.path.join(model_dir, "deepfm_best.h5")
    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=TRAINING_CONFIG["early_stopping_patience"],
            mode="max",
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    # Train
    logger.info("Training DeepFM model...")
    history = model.fit(
        train_input,
        train_label,
        batch_size=TRAINING_CONFIG["batch_size"],
        epochs=TRAINING_CONFIG["epochs"],
        validation_data=(val_input, val_label),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    val_pred = model.predict(val_input, batch_size=TRAINING_CONFIG["batch_size"])
    auc = roc_auc_score(val_label, val_pred)
    logloss = log_loss(val_label, val_pred)
    logger.info(f"DeepFM Validation AUC: {auc:.4f}, LogLoss: {logloss:.4f}")

    # Save
    model.save_weights(os.path.join(model_dir, "deepfm_weights.h5"))
    logger.info(f"DeepFM model saved to {model_dir}")

    return model, {"auc": auc, "logloss": logloss}


def train_din(df_train, df_val, feature_processor: FeatureProcessor, model_dir: str):
    """
    Train DIN (Deep Interest Network) model.

    DIN uses attention mechanism to capture user's diverse interests from
    behavior sequence. Especially good when users have rich interaction history.

    Extra input: hist_video_ids (user's recent watched/liked video IDs)
    Mapped from: user_behaviors table (behavior_type IN ('view', 'like', 'share'))
    """
    cfg = MODEL_CONFIG["din"]

    feature_columns = feature_processor.get_feature_columns()
    feature_names = feature_processor.get_feature_names()

    # Find the behavior feature column and target feature
    behavior_feature_list = [
        col for col in feature_columns
        if hasattr(col, "sparsefeat") and col.sparsefeat.name == "hist_video_ids"
    ]

    if not behavior_feature_list:
        logger.warning("No behavior sequence feature found. Building DIN with empty behavior list.")
        # Fallback: treat as DeepFM
        return train_deepfm(df_train, df_val, feature_processor, model_dir)

    model = DIN(
        dnn_feature_columns=feature_columns,
        history_feature_list=["video_id"],
        dnn_hidden_units=cfg["dnn_hidden_units"],
        att_hidden_size=cfg["att_hidden_size"],
        att_activation=cfg["att_activation"],
        dnn_dropout=cfg["dnn_dropout"],
        l2_reg_embedding=cfg["l2_reg_embedding"],
        task="binary",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TRAINING_CONFIG["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )

    train_input = feature_processor.get_model_input(df_train)
    val_input = feature_processor.get_model_input(df_val)
    train_label = df_train["is_click"].values
    val_label = df_val["is_click"].values

    model_path = os.path.join(model_dir, "din_best.h5")
    callbacks = [
        EarlyStopping(monitor="val_auc", patience=TRAINING_CONFIG["early_stopping_patience"],
                       mode="max", restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_auc", mode="max", save_best_only=True),
    ]

    logger.info("Training DIN model...")
    model.fit(
        train_input,
        train_label,
        batch_size=TRAINING_CONFIG["batch_size"],
        epochs=TRAINING_CONFIG["epochs"],
        validation_data=(val_input, val_label),
        callbacks=callbacks,
        verbose=1,
    )

    val_pred = model.predict(val_input, batch_size=TRAINING_CONFIG["batch_size"])
    auc = roc_auc_score(val_label, val_pred)
    logloss = log_loss(val_label, val_pred)
    logger.info(f"DIN Validation AUC: {auc:.4f}, LogLoss: {logloss:.4f}")

    model.save_weights(os.path.join(model_dir, "din_weights.h5"))
    logger.info(f"DIN model saved to {model_dir}")

    return model, {"auc": auc, "logloss": logloss}


def train_mmoe(df_train, df_val, feature_processor: FeatureProcessor, model_dir: str):
    """
    Train MMoE (Multi-gate Mixture-of-Experts) for multi-task learning.

    Tasks (from recommendation_exposures / user_video_interactions):
      - is_click: whether user clicked the video
      - is_finish: whether user completed watching (completion_rate >= 0.8)
      - is_like: whether user liked the video
      - is_share: whether user shared the video

    MMoE learns shared expert networks with task-specific gating,
    allowing knowledge transfer between related tasks.
    """
    cfg = MODEL_CONFIG["mmoe"]

    feature_columns = feature_processor.get_feature_columns()

    model = MMoE(
        dnn_feature_columns=feature_columns,
        tower_dnn_hidden_units=[64, 32],
        num_experts=cfg["num_experts"],
        expert_dnn_hidden_units=[cfg["expert_dim"], cfg["expert_dim"] // 2],
        task_types=cfg["task_types"],
        tasks=cfg["task_names"],
        l2_reg_embedding=1e-5,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TRAINING_CONFIG["learning_rate"]),
        loss={
            "is_click": "binary_crossentropy",
            "is_finish": "binary_crossentropy",
            "is_like": "binary_crossentropy",
            "is_share": "binary_crossentropy",
        },
        loss_weights={
            "is_click": 1.0,
            "is_finish": 0.8,
            "is_like": 0.5,
            "is_share": 0.3,
        },
        metrics=["AUC"],
    )

    train_input = feature_processor.get_model_input(df_train)
    val_input = feature_processor.get_model_input(df_val)

    train_labels = {
        "is_click": df_train["is_click"].values,
        "is_finish": df_train["is_finish"].values,
        "is_like": df_train["is_like"].values,
        "is_share": df_train["is_share"].values,
    }
    val_labels = {
        "is_click": df_val["is_click"].values,
        "is_finish": df_val["is_finish"].values,
        "is_like": df_val["is_like"].values,
        "is_share": df_val["is_share"].values,
    }

    model_path = os.path.join(model_dir, "mmoe_best.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=TRAINING_CONFIG["early_stopping_patience"],
                       restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    logger.info("Training MMoE model (multi-task: click/finish/like/share)...")
    model.fit(
        train_input,
        train_labels,
        batch_size=TRAINING_CONFIG["batch_size"],
        epochs=TRAINING_CONFIG["epochs"],
        validation_data=(val_input, val_labels),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate each task
    val_pred = model.predict(val_input, batch_size=TRAINING_CONFIG["batch_size"])
    metrics = {}
    for i, task_name in enumerate(cfg["task_names"]):
        pred = val_pred[i] if isinstance(val_pred, list) else val_pred
        true = val_labels[task_name]
        try:
            auc = roc_auc_score(true, pred)
            metrics[f"{task_name}_auc"] = auc
            logger.info(f"  {task_name} AUC: {auc:.4f}")
        except ValueError:
            logger.warning(f"  {task_name}: only one class in labels, skip AUC")
            metrics[f"{task_name}_auc"] = 0.0

    model.save_weights(os.path.join(model_dir, "mmoe_weights.h5"))
    logger.info(f"MMoE model saved to {model_dir}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train DeepCTR recommendation models")
    parser.add_argument("--model", type=str, default="deepfm",
                        choices=["deepfm", "din", "mmoe", "all"],
                        help="Model to train (default: deepfm)")
    parser.add_argument("--days", type=int, default=TRAINING_CONFIG["train_days"],
                        help="Days of training data to use")
    parser.add_argument("--output", type=str, default=SERVING_CONFIG["model_dir"],
                        help="Output directory for trained models")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Step 1: Generate training data from MySQL
    logger.info("=" * 60)
    logger.info("Step 1: Loading training data from MySQL...")
    logger.info("=" * 60)
    data_gen = DataGenerator()
    df = data_gen.generate_training_data(days=args.days)

    if df.empty or len(df) < 100:
        logger.error(f"Insufficient training data: {len(df)} samples. Need at least 100.")
        logger.info("Please ensure there are interaction records in the database.")
        sys.exit(1)

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Feature columns: {list(df.columns)}")

    # Step 2: Feature engineering
    logger.info("=" * 60)
    logger.info("Step 2: Feature engineering...")
    logger.info("=" * 60)
    fp = FeatureProcessor()
    df = fp.fit_transform(df)

    # Save feature processor for serving
    fp_path = os.path.join(args.output, "feature_processor.pkl")
    with open(fp_path, "wb") as f:
        pickle.dump(fp, f)
    logger.info(f"Feature processor saved to {fp_path}")

    # Step 3: Train/Val split (time-based to avoid leakage)
    logger.info("=" * 60)
    logger.info("Step 3: Train/Validation split...")
    logger.info("=" * 60)
    split_ratio = 1 - TRAINING_CONFIG["validation_split"]
    split_idx = int(len(df) * split_ratio)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]
    logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}")

    # Step 4: Train models
    all_metrics = {}

    models_to_train = [args.model] if args.model != "all" else ["deepfm", "din", "mmoe"]

    for model_name in models_to_train:
        logger.info("=" * 60)
        logger.info(f"Step 4: Training {model_name.upper()} model...")
        logger.info("=" * 60)

        try:
            if model_name == "deepfm":
                model, metrics = train_deepfm(df_train, df_val, fp, args.output)
            elif model_name == "din":
                model, metrics = train_din(df_train, df_val, fp, args.output)
            elif model_name == "mmoe":
                model, metrics = train_mmoe(df_train, df_val, fp, args.output)
            else:
                logger.error(f"Unknown model: {model_name}")
                continue

            all_metrics[model_name] = metrics
            logger.info(f"{model_name.upper()} metrics: {json.dumps(metrics, indent=2)}")

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}", exc_info=True)

    # Save metrics
    metrics_path = os.path.join(args.output, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(df),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "positive_ratio": float(df["is_click"].mean()),
            "models": all_metrics,
        }, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
