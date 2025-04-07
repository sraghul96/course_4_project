# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.model import inference
from starter.ml.data import process_data

import pandas as pd
import joblib

app = FastAPI()


class DataModel(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI model inference API!"}


@app.post("/predict")
def predict(data: DataModel):
    """
    Predict the salary based on the input data.
    """
    # Load the model and encoder
    model = joblib.load("model/model.pkl")
    encoder = joblib.load("model/encoder.pkl")
    lb = joblib.load("model/label_binarizer.pkl")

    # Convert the input data to a DataFrame
    data_dict = data.model_dump(by_alias=True)
    df = pd.DataFrame([data_dict])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Process the data
    X, _, _, _ = process_data(df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb)

    # Make predictions
    preds = inference(model, X)

    # Convert predictions back to original labels
    preds = lb.inverse_transform(preds)

    return {"predictions": preds.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.2", port=1000, reload=True)
