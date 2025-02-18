from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("fine_tuned_random_forest.pkl")

app = FastAPI()

# Define the expected JSON structure
class WicketPredictionInput(BaseModel):
    venue: str
    batting_team: str
    batter: str
    bowler: str
    non_striker: str
    runs_batter: int
    runs_total: int
    extras: int
    over_number: int
    delivery_number: int

@app.post("/predict_wicket")
def predict_wicket(input_data: WicketPredictionInput):
    try:
        print("Received Data:", input_data.dict())  # Print incoming data for debugging

        df = pd.DataFrame([input_data.dict()])

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability of a wicket

        return {"wicket_prediction": int(prediction), "wicket_probability": round(probability * 100, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

