from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "CardioPredict AI API Running"}

@app.post("/predict")
def predict(data: list):
    arr = np.array(data).reshape(1, -1)
    result = model.predict(arr)
    return {"prediction": int(result[0])}
