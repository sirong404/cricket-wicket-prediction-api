from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model and label encoders
model = joblib.load("fine_tuned_random_forest.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # Load encoders used during training

app = FastAPI()

# Define the expected input structure
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
        df = pd.DataFrame([input_data.dict()])

        # Convert categorical variables using label encoders
        for col in ["venue", "batting_team", "batter", "bowler", "non_striker"]:
            if col in label_encoders:
                df[col] = df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability of a wicket

        return {"wicket_prediction": int(prediction), "wicket_probability": round(probability * 100, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")
