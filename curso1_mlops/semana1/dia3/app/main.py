# FastAPI + modelo sklearn + caché Redis

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import redis
import os

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
MODEL_PATH = os.getenv("MODEL_PATH", "modelo.pkl")
CACHE_TTL  = int(os.getenv("CACHE_TTL", "3600"))

redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
model = None


class PredictRequest(BaseModel):
    x: float


class PredictResponse(BaseModel):
    input: float
    prediction: float
    cache: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="ML Model Serving API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "redis": _redis_ok()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    cache_key = f"predict:{request.x}"
    cached = redis_client.get(cache_key)
    if cached is not None:
        return PredictResponse(input=request.x, prediction=float(cached), cache=True)

    X = np.array([[request.x]])
    prediction = float(model.predict(X)[0])
    redis_client.setex(cache_key, CACHE_TTL, prediction)
    return PredictResponse(input=request.x, prediction=prediction, cache=False)


def _redis_ok() -> bool:
    try:
        return redis_client.ping()
    except Exception:
        return False
