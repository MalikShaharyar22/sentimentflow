from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os

# Do NOT instantiate heavy models at import time. We'll lazy-load them in get_model().

# Try to import local SentimentModel wrapper
try:
    from sentiment_model import SentimentModel
except Exception:
    SentimentModel = None

# Pydantic schemas
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

# Create app and enable CORS (allow all origins for dev; restrict in prod)
app = FastAPI(title="SentimentFlow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# Initialize model lazily
_model = None

def get_model():
    global _model
    if _model is None:
        # prefer user-provided SentimentModel class in sentiment_model.py
        if SentimentModel is not None:
            _model = SentimentModel()
        else:
            # fallback: lightweight pipeline
            from transformers import pipeline
            model_name = os.environ.get("HF_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
            use_auth = os.environ.get("HF_TOKEN", None)
            kwargs = {}
            if use_auth:
                kwargs["use_auth_token"] = use_auth
            cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/tmp/transformers_cache")
            os.makedirs(cache_dir, exist_ok=True)
            kwargs["cache_dir"] = cache_dir
            _model = pipeline("sentiment-analysis", model=model_name, **kwargs)
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(input: TextInput):
    try:
        model = get_model()
        if hasattr(model, "predict"):
            out = model.predict(input.text)
            return out
        out = model(input.text[:512])[0]
        label = out.get("label", "").upper()
        score = float(out.get("score", 0.0))
        prob_pos = score if label == "POSITIVE" else (1.0 - score)
        return {"label": label, "score": score, "prob_pos": prob_pos}
    except Exception as e:
        logger.exception("Error in /analyze")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
def batch(input: BatchInput):
    try:
        model = get_model()
        results = []
        for t in input.texts:
            try:
                if hasattr(model, "predict"):
                    r = model.predict(t)
                else:
                    out = model(t[:512])[0]
                    label = out.get("label", "").upper()
                    score = float(out.get("score", 0.0))
                    r = {"label": label, "score": score, "prob_pos": score if label == "POSITIVE" else (1.0 - score)}
            except Exception:
                r = {"label": None, "score": 0.0, "prob_pos": 0.0}
            results.append(r)
        return {"results": results}
    except Exception as e:
        logger.exception("Error in /batch")
        raise HTTPException(status_code=500, detail=str(e))