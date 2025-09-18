# backend/app.py
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try to import a sentiment model wrapper from sentiment_model.py if present.
# If not present, fallback to a simple built-in wrapper using HuggingFace pipeline.
try:
    from sentiment_model import \
        SentimentModel  # user-provided wrapper (preferred)
except Exception:
    from transformers import pipeline

    class SentimentModel:
        def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
            # initialize HF pipeline
            self.pipeline = pipeline("sentiment-analysis", model=model_name)

        def predict(self, text: str) -> Dict[str, Any]:
            if not text:
                return {"label": None, "score": 0.0, "prob_pos": 0.0}
            # Hugging Face pipeline returns best-label + score (max class probability)
            out = self.pipeline(text[:512])[0]
            label = out.get("label", "").upper()
            score = float(out.get("score", 0.0))
            # Approximate prob_pos: if label is POSITIVE use score, else 1-score
            prob_pos = score if label == "POSITIVE" else (1.0 - score)
            return {"label": label, "score": score, "prob_pos": prob_pos}


# Pydantic request models
class TextInput(BaseModel):
    text: str


class BatchInput(BaseModel):
    texts: List[str]


# Create app and enable CORS (allow all origins for dev; restrict in prod)
app = FastAPI(title="SentimentFlow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
logger = logging.getLogger("uvicorn.error")
model = SentimentModel()

# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(input: TextInput):
    try:
        out = model.predict(input.text)
        return out
    except Exception as e:
        logger.exception("Error in /analyze")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
def batch(input: BatchInput):
    try:
        results = []
        for t in input.texts:
            try:
                res = model.predict(t)
            except Exception:
                # ensure we return a consistent shape even if one item fails
                res = {"label": None, "score": 0.0, "prob_pos": 0.0}
            results.append(res)
        return {"results": results}
    except Exception as e:
        logger.exception("Error in /batch")
        raise HTTPException(status_code=500, detail=str(e))
