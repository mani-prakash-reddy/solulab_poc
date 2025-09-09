from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from typing import Dict, List
import numpy as np

app = FastAPI(title="Toxic Comment Classification API")

# Load models and vectorizer
try:
    models = {}
    model_dir = "models"
    vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    
    # Load all models
    for model_file in os.listdir(model_dir):
        if model_file.startswith("logreg_") and model_file.endswith(".joblib"):
            label = model_file[7:-7]
            model_path = os.path.join(model_dir, model_file)
            models[label] = joblib.load(model_path)
    
except Exception as e:
    raise RuntimeError(f"Failed to load models: {str(e)}")

class TextRequest(BaseModel):
    text: str

class TextBatchRequest(BaseModel):
    texts: List[str]

@app.get("/")
async def root():
    return {
        "message": "Toxic Comment Classification API",
        "available_models": list(models.keys())
    }

@app.post("/predict")
async def predict_single(request: TextRequest) -> Dict[str, float]:
    try:
        # Transform the text using the vectorizer
        X = vectorizer.transform([request.text])
        
        # Get predictions from all models
        results = {}
        for label, model in models.items():
            # Get probability of toxic class
            prob = model.predict_proba(X)[0][1]
            results[label] = float(prob)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: TextBatchRequest) -> List[Dict[str, float]]:
    try:
        # Transform the texts using the vectorizer
        X = vectorizer.transform(request.texts)
        
        # Get predictions from all models
        results = []
        for i in range(len(request.texts)):
            text_results = {}
            for label, model in models.items():
                # Get probability of toxic class
                prob = model.predict_proba(X[i:i+1])[0][1]
                text_results[label] = float(prob)
            results.append(text_results)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
