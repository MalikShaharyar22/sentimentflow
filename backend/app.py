from fastapi import FastAPI
from pydantic import BaseModel
from sentiment_model import SentimentModel

app = FastAPI(title="SentimentFlow API")
model = SentimentModel()

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]

@app.post("/analyze")
def analyze(input: TextInput):
    return model.predict(input.text)

@app.post("/batch")
def batch(input: BatchInput):
    results = [model.predict(t) for t in input.texts]
    return {"results": results}
