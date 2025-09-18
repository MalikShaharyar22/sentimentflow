import os
from typing import Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class SentimentModel:
    """
    Wrapper around Hugging Face transformers pipeline.
    Uses TRANSFORMERS_CACHE (defaults to /tmp/transformers_cache) to avoid root /.cache permission issues.
    """
    def __init__(self, model_name: str = None):
        model_name = model_name or os.environ.get(
            "HF_MODEL",
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        hf_token = os.environ.get("HF_TOKEN", None)

        # ensure cache dir is writable
        cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/tmp/transformers_cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            os.chmod(cache_dir, 0o777)
        except Exception:
            pass

        # load tokenizer & model with cache_dir
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_auth_token=hf_token
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_auth_token=hf_token
        )

        # build pipeline (no cache_dir here!)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
        )

    def predict(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"label": None, "score": 0.0, "prob_pos": 0.0}
        out = self.pipe(text[:512])[0]
        label = out.get("label", "").upper()
        score = float(out.get("score", 0.0))
        prob_pos = score if label == "POSITIVE" else (1.0 - score)
        return {"label": label, "score": score, "prob_pos": prob_pos}