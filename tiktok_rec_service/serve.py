"""
FastAPI HTTP Serving for TikTok Video CTR Prediction.
Exposes endpoints that match the Go CTRServiceClient (ctr_client.go).

Endpoints:
  POST /predict          - Single model CTR prediction
  POST /predict/ensemble - Ensemble prediction (multiple models)
  GET  /health           - Health check
  GET  /metrics          - Model metrics
"""

import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add parent directory to path for deepctr imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DENSE_FEATURES, MODEL_CONFIG, SERVING_CONFIG, SPARSE_FEATURES
from data_generator import DataGenerator
from feature_engineering import FeatureProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================
# Pydantic models (matching Go CTRPredictRequest/Response)
# =====================================================


class PredictRequest(BaseModel):
    """Matches Go CTRPredictRequest in ctr_client.go"""
    user_id: int
    video_ids: List[int]
    context: Optional[Dict[str, str]] = None
    model: Optional[str] = None  # deepfm/din/mmoe


class PredictionResult(BaseModel):
    """Matches Go CTRPrediction in ctr_client.go"""
    video_id: int
    score: float
    ctr: float
    is_finish: float = 0.0
    is_like: float = 0.0
    is_share: float = 0.0


class PredictResponse(BaseModel):
    """Matches Go CTRPredictResponse in ctr_client.go"""
    predictions: List[PredictionResult]
    latency_ms: float
    model: str
    models_used: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    models: Dict


# =====================================================
# Model Manager
# =====================================================


class ModelManager:
    """Manages loading and inference of DeepCTR models."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = {}
        self.feature_processor: Optional[FeatureProcessor] = None
        self.data_generator = DataGenerator()
        self.start_time = time.time()
        self.request_count = 0
        self.total_latency = 0.0

        self._load_models()

    def _load_models(self):
        """Load all available trained models and the feature processor."""
        # Load feature processor
        fp_path = os.path.join(self.model_dir, "feature_processor.pkl")
        if os.path.exists(fp_path):
            with open(fp_path, "rb") as f:
                self.feature_processor = pickle.load(f)
            logger.info("Feature processor loaded successfully")
        else:
            logger.warning(f"Feature processor not found at {fp_path}")
            logger.info("Will use a new FeatureProcessor (not ideal for production)")
            self.feature_processor = FeatureProcessor()

        # Load DeepFM
        self._try_load_model("deepfm", "deepfm_weights.h5")

        # Load DIN
        self._try_load_model("din", "din_weights.h5")

        # Load MMoE
        self._try_load_model("mmoe", "mmoe_weights.h5")

        if not self.models:
            logger.warning("No trained models found! Will use fallback scoring.")
        else:
            logger.info(f"Loaded models: {list(self.models.keys())}")

    def _try_load_model(self, model_name: str, weights_file: str):
        """Try to load a model, skip if not available."""
        weights_path = os.path.join(self.model_dir, weights_file)
        if not os.path.exists(weights_path):
            logger.info(f"Model weights not found: {weights_path}, skipping {model_name}")
            return

        try:
            import tensorflow as tf
            feature_columns = self.feature_processor.get_feature_columns()

            if model_name == "deepfm":
                from deepctr.models import DeepFM
                cfg = MODEL_CONFIG["deepfm"]
                model = DeepFM(
                    linear_feature_columns=feature_columns,
                    dnn_feature_columns=feature_columns,
                    dnn_hidden_units=cfg["dnn_hidden_units"],
                    task="binary",
                )
            elif model_name == "din":
                from deepctr.models.sequence.din import DIN
                cfg = MODEL_CONFIG["din"]
                model = DIN(
                    dnn_feature_columns=feature_columns,
                    history_feature_list=["video_id"],
                    dnn_hidden_units=cfg["dnn_hidden_units"],
                    task="binary",
                )
            elif model_name == "mmoe":
                from deepctr.models.multitask.mmoe import MMOE
                cfg = MODEL_CONFIG["mmoe"]
                model = MMOE(
                    dnn_feature_columns=feature_columns,
                    tower_dnn_hidden_units=[64, 32],
                    num_experts=cfg["num_experts"],
                    task_types=cfg["task_types"],
                    tasks=cfg["task_names"],
                )
            else:
                return

            # Build model by running a dummy prediction
            dummy_input = self._create_dummy_input(feature_columns)
            model.predict(dummy_input, batch_size=1)

            # Load weights
            model.load_weights(weights_path)
            self.models[model_name] = model
            logger.info(f"Model '{model_name}' loaded successfully from {weights_path}")

        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}", exc_info=True)

    def _create_dummy_input(self, feature_columns) -> dict:
        """Create minimal dummy input to build the model graph."""
        dummy = {}
        for col in feature_columns:
            name = col.name if hasattr(col, "name") else str(col)
            if hasattr(col, "vocabulary_size"):
                dummy[name] = np.array([0])
            elif hasattr(col, "maxlen"):
                dummy[name] = np.zeros((1, col.maxlen), dtype="int32")
            else:
                dummy[name] = np.array([0.0], dtype="float32")
        return dummy

    # =====================================================
    # Prediction
    # =====================================================

    def predict(self, user_id: int, video_ids: List[int],
                model_name: Optional[str] = None) -> List[PredictionResult]:
        """Run CTR prediction for given user and video candidates."""
        start = time.time()

        model_name = model_name or SERVING_CONFIG["default_model"]

        # Fetch features from DB
        try:
            df = self.data_generator.generate_prediction_data(user_id, video_ids)
        except Exception as e:
            logger.error(f"Failed to generate prediction data: {e}")
            return self._fallback_predictions(video_ids)

        if df.empty:
            return self._fallback_predictions(video_ids)

        # Transform features (preserve original video_id for matching)
        try:
            df["_orig_video_id"] = df["video_id"].values.copy()
            df = self.feature_processor.transform(df)
        except Exception as e:
            logger.error(f"Feature transform failed: {e}")
            return self._fallback_predictions(video_ids)

        # Run model inference
        model = self.models.get(model_name)
        if model is None:
            logger.warning(f"Model '{model_name}' not loaded, using fallback")
            return self._fallback_predictions(video_ids)

        try:
            model_input = self.feature_processor.get_model_input(df)

            if model_name == "mmoe":
                preds = model.predict(model_input, batch_size=len(video_ids))
                # MMoE returns [click_pred, finish_pred, like_pred, share_pred]
                results = []
                for i, vid in enumerate(video_ids):
                    idx = df.index[df["_orig_video_id"] == vid]
                    if len(idx) == 0:
                        results.append(self._default_prediction(vid))
                        continue
                    j = idx[0]
                    click_score = float(preds[0][j]) if isinstance(preds, list) else float(preds[j])
                    finish_score = float(preds[1][j]) if isinstance(preds, list) and len(preds) > 1 else 0.0
                    like_score = float(preds[2][j]) if isinstance(preds, list) and len(preds) > 2 else 0.0
                    share_score = float(preds[3][j]) if isinstance(preds, list) and len(preds) > 3 else 0.0

                    # Combined score: weighted sum of all task predictions
                    combined = 0.4 * click_score + 0.3 * finish_score + 0.2 * like_score + 0.1 * share_score

                    results.append(PredictionResult(
                        video_id=vid,
                        score=combined,
                        ctr=click_score,
                        is_finish=finish_score,
                        is_like=like_score,
                        is_share=share_score,
                    ))
            else:
                preds = model.predict(model_input, batch_size=len(video_ids))
                results = []
                for i, vid in enumerate(video_ids):
                    idx = df.index[df["_orig_video_id"] == vid]
                    if len(idx) == 0:
                        results.append(self._default_prediction(vid))
                        continue
                    j = idx[0]
                    score = float(preds[j])
                    results.append(PredictionResult(
                        video_id=vid,
                        score=score,
                        ctr=score,
                    ))

            # Sort by score descending
            results.sort(key=lambda x: x.score, reverse=True)

            latency = (time.time() - start) * 1000
            self.request_count += 1
            self.total_latency += latency

            return results

        except Exception as e:
            logger.error(f"Model prediction failed: {e}", exc_info=True)
            return self._fallback_predictions(video_ids)

    def predict_ensemble(self, user_id: int, video_ids: List[int]) -> (List[PredictionResult], List[str]):
        """Ensemble prediction: average scores from all loaded models."""
        if not self.models:
            return self._fallback_predictions(video_ids), []

        all_scores = {}  # video_id -> list of scores
        models_used = []

        for model_name in self.models:
            try:
                preds = self.predict(user_id, video_ids, model_name)
                models_used.append(model_name)
                for p in preds:
                    if p.video_id not in all_scores:
                        all_scores[p.video_id] = {"scores": [], "ctr": [], "finish": [], "like": [], "share": []}
                    all_scores[p.video_id]["scores"].append(p.score)
                    all_scores[p.video_id]["ctr"].append(p.ctr)
                    all_scores[p.video_id]["finish"].append(p.is_finish)
                    all_scores[p.video_id]["like"].append(p.is_like)
                    all_scores[p.video_id]["share"].append(p.is_share)
            except Exception as e:
                logger.error(f"Ensemble prediction failed for {model_name}: {e}")

        results = []
        for vid in video_ids:
            if vid in all_scores:
                s = all_scores[vid]
                results.append(PredictionResult(
                    video_id=vid,
                    score=np.mean(s["scores"]),
                    ctr=np.mean(s["ctr"]),
                    is_finish=np.mean(s["finish"]),
                    is_like=np.mean(s["like"]),
                    is_share=np.mean(s["share"]),
                ))
            else:
                results.append(self._default_prediction(vid))

        results.sort(key=lambda x: x.score, reverse=True)
        return results, models_used

    def _fallback_predictions(self, video_ids: List[int]) -> List[PredictionResult]:
        """Fallback scoring when model is unavailable."""
        return [self._default_prediction(vid) for vid in video_ids]

    def _default_prediction(self, video_id: int) -> PredictionResult:
        return PredictionResult(video_id=video_id, score=0.5, ctr=0.5)


# =====================================================
# FastAPI Application
# =====================================================

app = FastAPI(
    title="TikTok Video CTR Prediction Service",
    description="DeepCTR-based CTR prediction service for Refactored-TikTok recommendation system",
    version="1.0.0",
)

# Global model manager (lazy initialization)
model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(SERVING_CONFIG["model_dir"])
    return model_manager


# =====================================================
# API Endpoints (matching Go ctr_client.go)
# =====================================================


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Single model CTR prediction.
    Called by Go CTRServiceClient.Predict() in ctr_client.go
    """
    start = time.time()
    mgr = get_model_manager()

    if not req.video_ids:
        return PredictResponse(predictions=[], latency_ms=0, model="none")

    model_name = req.model or SERVING_CONFIG["default_model"]
    predictions = mgr.predict(req.user_id, req.video_ids, model_name)

    latency = (time.time() - start) * 1000
    return PredictResponse(
        predictions=predictions,
        latency_ms=round(latency, 2),
        model=model_name,
    )


@app.post("/predict/ensemble", response_model=PredictResponse)
async def predict_ensemble(req: PredictRequest):
    """
    Ensemble CTR prediction (all loaded models).
    Called by Go CTRServiceClient when EnableEnsemble=true.
    """
    start = time.time()
    mgr = get_model_manager()

    if not req.video_ids:
        return PredictResponse(predictions=[], latency_ms=0, model="ensemble")

    predictions, models_used = mgr.predict_ensemble(req.user_id, req.video_ids)

    latency = (time.time() - start) * 1000
    return PredictResponse(
        predictions=predictions,
        latency_ms=round(latency, 2),
        model="ensemble",
        models_used=models_used,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Called by Go CTRServiceClient.checkHealth() every 10 seconds.
    """
    mgr = get_model_manager()
    return HealthResponse(
        status="healthy",
        models_loaded=list(mgr.models.keys()),
        uptime_seconds=round(time.time() - mgr.start_time, 1),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Service metrics."""
    mgr = get_model_manager()

    # Load training metrics if available
    import json
    metrics_path = os.path.join(mgr.model_dir, "training_metrics.json")
    model_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            model_metrics = json.load(f)

    avg_latency = mgr.total_latency / max(mgr.request_count, 1)
    return MetricsResponse(
        total_requests=mgr.request_count,
        avg_latency_ms=round(avg_latency, 2),
        models=model_metrics,
    )


@app.on_event("startup")
async def startup_event():
    """Pre-load models on startup."""
    logger.info("Starting CTR prediction service...")
    get_model_manager()
    logger.info("CTR prediction service ready!")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host=SERVING_CONFIG["host"],
        port=SERVING_CONFIG["port"],
        reload=False,
        workers=1,  # TensorFlow models are not fork-safe
        log_level="info",
    )
