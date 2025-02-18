from fastapi import FastAPI
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("fine_tuned_random_forest.pkl")

# Create FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def home():
    return {"message": "Cricket Wicket Prediction API is live!"}

# Define the prediction endpoint
@app.post("/predict_wicket")
def predict_wicket(
    venue: str,
    batting_team: str,
    batter: str,
    bowler: str,
    non_striker: str,
    runs_batter: int,
    runs_total: int,
    extras: int,
    over_number: int,
    delivery_number: int
):
    # Convert input into a DataFrame
    input_data = pd.DataFrame([[
        venue, batting_team, batter, bowler, non_striker,
        runs_batter, runs_total, extras, over_number, delivery_number
    ]], columns=["venue", "batting_team", "batter", "bowler", "non_striker",
                 "runs_batter", "runs_total", "extras", "over_number", "delivery_number"])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of a wicket

    return {"wicket_prediction": int(prediction), "wicket_probability": round(probability * 100, 2)}
