"""
Feature engineering module for TikTok Video Recommendation.
Defines DeepCTR feature columns mapped to Refactored-TikTok DB tables.
"""

import numpy as np
import pandas as pd
from config import (
    DENSE_FEATURES,
    DURATION_BUCKETS,
    DURATION_LABELS,
    SEQUENCE_FEATURE,
    SEQUENCE_MAX_LEN,
    SPARSE_FEATURES,
)
from deepctr.feature_column import DenseFeat, SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class FeatureProcessor:
    """Processes raw data from MySQL tables into DeepCTR-compatible features."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit encoders and transform the data."""
        df = self._engineer_features(df)
        df = self._encode_sparse_features(df, fit=True)
        df = self._scale_dense_features(df, fit=True)
        self._build_feature_columns(df)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted encoders."""
        if not self._fitted:
            raise RuntimeError("FeatureProcessor not fitted. Call fit_transform first.")
        df = self._engineer_features(df)
        df = self._encode_sparse_features(df, fit=False)
        df = self._scale_dense_features(df, fit=False)
        return df

    def get_feature_columns(self):
        """Return DeepCTR feature column definitions."""
        return self.feature_columns

    def get_feature_names(self):
        """Return feature names list for model input."""
        from deepctr.feature_column import get_feature_names
        return get_feature_names(self.feature_columns)

    # =====================================================
    # Feature Engineering
    # =====================================================

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from raw columns."""
        df = df.copy()

        # --- Video duration bucketing ---
        if "video_duration" in df.columns:
            df["video_duration_bucket"] = pd.cut(
                df["video_duration"],
                bins=DURATION_BUCKETS,
                labels=DURATION_LABELS,
                right=False,
            ).astype(str)
        else:
            df["video_duration_bucket"] = "unknown"

        # --- Time context features ---
        if "exposure_time" in df.columns:
            df["exposure_time"] = pd.to_datetime(df["exposure_time"])
            df["hour_of_day"] = df["exposure_time"].dt.hour.astype(str)
            df["day_of_week"] = df["exposure_time"].dt.dayofweek.astype(str)
        else:
            df["hour_of_day"] = "0"
            df["day_of_week"] = "0"

        # --- Video freshness (hours since creation) ---
        if "video_created_at" in df.columns and "exposure_time" in df.columns:
            df["video_created_at"] = pd.to_datetime(df["video_created_at"])
            delta = (df["exposure_time"] - df["video_created_at"]).dt.total_seconds() / 3600.0
            df["video_freshness_hours"] = delta.clip(lower=0).fillna(0)
        else:
            df["video_freshness_hours"] = 0.0

        # --- Default device type (can be enriched later) ---
        if "device_type" not in df.columns:
            df["device_type"] = "mobile"

        # --- Fill NaN for sparse/dense features ---
        for col in SPARSE_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna("unknown").astype(str)
            else:
                df[col] = "unknown"

        for col in DENSE_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0

        return df

    # =====================================================
    # Sparse Feature Encoding
    # =====================================================

    def _encode_sparse_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode sparse categorical features."""
        for feat in SPARSE_FEATURES:
            if feat not in df.columns:
                df[feat] = "unknown"

            if fit:
                le = LabelEncoder()
                # Add "unknown" to handle unseen values
                unique_vals = list(df[feat].unique()) + ["unknown"]
                le.fit(unique_vals)
                self.label_encoders[feat] = le
            else:
                le = self.label_encoders.get(feat)
                if le is None:
                    continue
                # Handle unseen values
                known = set(le.classes_)
                df[feat] = df[feat].apply(lambda x: x if x in known else "unknown")

            df[feat] = le.transform(df[feat])

        return df

    # =====================================================
    # Dense Feature Scaling
    # =====================================================

    def _scale_dense_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Min-Max scale dense numeric features."""
        available_dense = [f for f in DENSE_FEATURES if f in df.columns]
        if not available_dense:
            return df

        if fit:
            self.scaler.fit(df[available_dense])

        df[available_dense] = self.scaler.transform(df[available_dense])
        return df

    # =====================================================
    # Build DeepCTR Feature Columns
    # =====================================================

    def _build_feature_columns(self, df: pd.DataFrame):
        """Build DeepCTR feature column definitions."""
        sparse_feature_columns = []
        dense_feature_columns = []

        # Sparse features
        for feat in SPARSE_FEATURES:
            vocab_size = df[feat].nunique() + 1  # +1 for unknown
            # Use larger embedding for high-cardinality features
            if feat in ("user_id", "video_id", "author_id"):
                emb_dim = 16
            else:
                emb_dim = 8
            sparse_feature_columns.append(
                SparseFeat(feat, vocabulary_size=vocab_size, embedding_dim=emb_dim)
            )

        # Dense features
        for feat in DENSE_FEATURES:
            if feat in df.columns:
                dense_feature_columns.append(DenseFeat(feat, dimension=1))

        self.feature_columns = sparse_feature_columns + dense_feature_columns

    def build_sequence_feature(self, df: pd.DataFrame, hist_col: str = "hist_video_ids"):
        """Build VarLenSparseFeat for DIN model (user behavior sequence)."""
        if hist_col not in df.columns:
            return

        # Get video_id vocabulary size from label encoder
        vid_le = self.label_encoders.get("video_id")
        if vid_le is None:
            return

        vocab_size = len(vid_le.classes_) + 1

        seq_feature = VarLenSparseFeat(
            SparseFeat(hist_col, vocabulary_size=vocab_size, embedding_dim=16,
                       embedding_name="video_id"),  # share embedding with video_id
            maxlen=SEQUENCE_MAX_LEN,
            combiner="mean",
        )

        self.feature_columns.append(seq_feature)

    # =====================================================
    # Prepare Model Input
    # =====================================================

    def get_model_input(self, df: pd.DataFrame) -> dict:
        """Convert DataFrame to dict of numpy arrays for DeepCTR model input."""
        model_input = {}

        for feat in SPARSE_FEATURES:
            if feat in df.columns:
                model_input[feat] = df[feat].values

        for feat in DENSE_FEATURES:
            if feat in df.columns:
                model_input[feat] = df[feat].values.astype("float32")

        # Handle sequence feature (for DIN)
        if SEQUENCE_FEATURE in df.columns:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            model_input[SEQUENCE_FEATURE] = pad_sequences(
                df[SEQUENCE_FEATURE].values,
                maxlen=SEQUENCE_MAX_LEN,
                padding="post",
                truncating="post",
                value=0,
            )

        return model_input
