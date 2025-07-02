import argparse
from transformers import set_seed
import json
# import logging
# from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import uvicorn
from action.inference_review import ReviewInfer
import secrets
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global model instance for API
model_instance = None

# Pydantic models for API
class TripleInput(BaseModel):
    head_entity: str
    relation: str  
    tail_entity: str

class ScoreResponse(BaseModel):
    result: str
    confidence: float
    triple: TripleInput

# FastAPI app
app = FastAPI(
    title="KGARevion Scoring API",
    description="API for scoring knowledge graph triples using KGARevion model",
    version="1.0.0"
)
security = HTTPBasic()

USERNAME = os.environ.get("SERVICE_USERNAME")
PASSWORD = os.environ.get("SERVICE_PASSWORD")

# Check if credentials are set
if USERNAME is None or PASSWORD is None:
    raise ValueError(
        "Authentication credentials not found. Please set SERVICE_USERNAME and SERVICE_PASSWORD environment variables."
    )

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts"""
    global model_instance
    set_seed(42)
    # Use default weights path - you can modify this as needed
    weights_path = "fine_tuned_model/"
    print("Loading KGARevion model...")
    model_instance = ReviewInfer(model_weights=weights_path, model_name='llama3.1')
    print("Model loaded successfully!")

@app.get("/score")
async def score_triple(
    user: str = Depends(authenticate),
    query: str = Query(..., description="JSON string containing the triple to score, e.g., '{\"head_entity\": \"ADH1B\", \"relation\": \"protein_protein\", \"tail_entity\": \"KIF15\"}'")
) -> ScoreResponse:
    """
    Score a knowledge graph triple for factual correctness.
    
    Args:
        query: JSON string with head_entity, relation, and tail_entity
        
    Returns:
        ScoreResponse with result (True/False), confidence score, and the input triple
    """
    global model_instance
    
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Parse the query JSON
        triple_data = json.loads(query)
        triple_input = TripleInput(**triple_data)
        
        # Create the input format expected by the score function
        score_input = [triple_input.head_entity, triple_input.relation, triple_input.tail_entity]
        
        # Get the score
        result, confidence = model_instance.score(score_input)
        
        return ScoreResponse(
            result=result,
            confidence=confidence,
            triple=triple_input
        )
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, 
            detail="Invalid JSON format in query parameter. Expected format: '{\"head_entity\": \"entity1\", \"relation\": \"relation_type\", \"tail_entity\": \"entity2\"}'"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/score")
async def score_triple_post(triple: TripleInput) -> ScoreResponse:
    """
    Score a knowledge graph triple for factual correctness (POST version).
    
    Args:
        triple: TripleInput object with head_entity, relation, and tail_entity
        
    Returns:
        ScoreResponse with result (True/False), confidence score, and the input triple
    """
    global model_instance
    
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Create the input format expected by the score function
        score_input = [triple.head_entity, triple.relation, triple.tail_entity]
        
        # Get the score
        result, confidence = model_instance.score(score_input)
        
        return ScoreResponse(
            result=result,
            confidence=confidence,
            triple=triple
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_instance is not None}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "KGARevion Scoring API",
        "version": "1.0.0",
        "endpoints": {
            "GET /score": "Score a triple using query parameter",
            "POST /score": "Score a triple using request body",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation"
        },
        "example_usage": {
            "GET": "/score?query={\"head_entity\": \"ADH1B\", \"relation\": \"protein_protein\", \"tail_entity\": \"KIF15\"}",
            "POST": {
                "url": "/score",
                "body": {
                    "head_entity": "ADH1B",
                    "relation": "protein_protein", 
                    "tail_entity": "KIF15"
                }
            }
        }
    }

# def score(args):
#     """Original score function for command-line usage"""
#     model = ReviewInfer(model_weights = args.weights_path, model_name = 'llama3.1')
#     print("=== SCORES ===\n\n")
#     print(model.score(['ADH1B', 'protein_protein', 'KIF15']))
#     print(model.score(['Clathrin', 'interacts with', 'FAT3 protein']))
#     print(model.score(['AHR', 'target', 'TG']))

def run_api(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    # set_seed(42)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--key", type=str)
    # parser.add_argument("--query", type=str)
    # parser.add_argument("--type", type=str, default='MCQ', choices=['MCQ', 'SAQ'])
    # parser.add_argument("--max_round", type=int, default=1)
    # parser.add_argument("--is_revise", type=bool, default=True)
    # parser.add_argument("--KG_name", default='primeKG', choices=['UMLS', 'primeKG', 'ogb-biokg'], type=str)
    # parser.add_argument("--llm_name", default='llama3.1', choices=['llama3.1', 'llama3', 'gpt-4-turbo', 'llama3.1-70'], type=str)
    # parser.add_argument("--weights_path", type=str, default='fine_tuned_model/')
    parser.add_argument("--api", action="store_true", help="Run as API server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    args = parser.parse_args()
    
    if args.api:
        print(f"Starting FastAPI server on {args.host}:{args.port}")
        run_api(host=args.host, port=args.port)

