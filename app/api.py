from pathlib import Path
import numpy as np
from fastapi import FastAPI, Response
from joblib import load
# from pypmml import Model
from .schemas import Wine, Rating, feature_names
# from .schemas import Iris, Rating, feature_names
# from .monitoring import instrumentator

ROOT_DIR = Path(__file__).parent.parent

app = FastAPI()
scaler = load(ROOT_DIR / "artifacts/scaler.joblib")
model = load(ROOT_DIR / "artifacts/model.joblib")
# model = Model.fromFile("DecisionTreeIris.pmml")
# model = Model.load("DecisionTreeIris.pmml")

@app.get("/")
def root():
    return "Wine Quality Ratings"
    # return "Iris Species"

@app.post("/predict", response_model=Rating)
def predict(response: Response, sample: Wine):
    sample_dict = sample.dict()
    features = np.array([sample_dict[f] for f in feature_names]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    response.headers["X-model-score"] = str(prediction)
    return Rating(quality=prediction)

# @app.post("/predict", response_model=Rating)
# def predict(response: Response, sample: Iris):
#     sample_dict = sample.dict()
#     features = np.array([sample_dict[f] for f in feature_names]).reshape(1, -1)
#     # features_scaled = scaler.transform(features)
#     features_scaled = features
#     prediction = model.predict(features_scaled)[0]
#     response.headers["X-model-score"] = str(prediction)
#     return Rating(Species=prediction)

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}