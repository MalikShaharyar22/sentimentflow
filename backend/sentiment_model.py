from transformers import pipeline


class SentimentModel:
    def __init__(self):
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def predict(self, text: str):
        result = self.pipeline(text[:512])[0]  # truncate long texts
        return {"label": result['label'], "score": float(result['score'])}
