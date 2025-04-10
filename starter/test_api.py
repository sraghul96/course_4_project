from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the FastAPI model inference API!"}


def test_predict_less_than_50k():
    response = client.post(
        "/predict",
        json={
            "age": 25,
            "workclass": "Private",
            "fnlwgt": 226802,
            "education": "11th",
            "education-num": 7,
            "marital-status": "Never-married",
            "occupation": "Machine-op-inspct",
            "relationship": "Own-child",
            "race": "Black",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"predictions": [" <=50K"]}


def test_predict_greater_than_50k():
    response = client.post(
        "/predict",
        json={
            "age": 45,
            "workclass": "Private",
            "fnlwgt": 234721,
            "education": "Doctorate",
            "education-num": 16,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 14084,
            "capital-loss": 0,
            "hours-per-week": 60,
            "native-country": "United-States",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"predictions": [" >50K"]}
