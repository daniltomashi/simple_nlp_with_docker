from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError, validator
from typing import List
import joblib
from utils.lifecycle_manager import ModelLifecycleManager
from utils.text_processing import TextProcessing
import pandas as pd



app = FastAPI()


text_processor = TextProcessing()

# initialize and load model, vectorizer and label encoder
model_path, vectorizer_path, label_encoder_path = "tools/model.pkl", "tools/vectorizer.pkl", "tools/label_encoder.pkl"
model_manager = ModelLifecycleManager(model_path, vectorizer_path, label_encoder_path)

# Lazy load the model when needed
model, vectorizer, label_encoder = model_manager.get_tools()


class TextRequest(BaseModel):
    text: str

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Text must not be empty")
        return v

class TextPrediction(BaseModel):
    prediction: str
    confidence: float


@app.post('/predict', response_model=TextPrediction)
async def predict(request: TextRequest):
    try:
        input_text = request.text
        processed_text = text_processor.preprocess(pd.Series(input_text))
        vectorized_text = vectorizer.transform(processed_text)
        prediction = model.predict(vectorized_text)[0]
        predict_proba = model.predict_proba(vectorized_text)[0].max() # [prediction]

        prediction = label_encoder.inverse_transform([prediction])[0]

        return TextPrediction(prediction=prediction, confidence=predict_proba)


    except Exception as e :
        raise HTTPException(status_code=400, detail='Error in prediction: ' + str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)