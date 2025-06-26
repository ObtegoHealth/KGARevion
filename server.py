from fastapi import FastAPI
from typing import Any
import triplet_agent  # whatever runs your model

app = FastAPI()

@app.get("/predict")
async def predict(q: str) -> Any:
    """
    Simple GET endpoint:
      GET /predict?q=some+question
      → {"answer": "..."}
    """
    answer = triplet_agent.get_triplets(q)
    return {"answer": answer}