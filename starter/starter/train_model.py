# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
import os
import joblib
from ml.model import train_model, inference, compute_model_metrics
import pandas
from datetime import datetime


def load_data():
    """
    Load the data from the CSV file.
    """
    data = pandas.read_csv(
        "/data/home/corp.evolenthealth.com/rsrinivas/Udacity/MLOps/course_4_project/starter/data/census_udacity.csv"
    )
    data.columns = [col.strip() for col in data.columns]
    return data


def training_pipeline():
    print("Starting script...", datetime.now())
    # Add code to load in the data.
    data = load_data()
    print("Data loaded successfully.", datetime.now())

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(test, cat_features, label="salary", training=False, encoder=encoder, lb=lb)

    # Train and save a model.
    model = train_model(X_train, y_train)
    base_path = "model"
    os.makedirs(base_path, exist_ok=True)
    joblib.dump(model, os.path.join(base_path, "model.pkl"))
    joblib.dump(encoder, os.path.join(base_path, "encoder.pkl"))
    joblib.dump(lb, os.path.join(base_path, "label_binarizer.pkl"))

    pred = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, pred)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Fbeta: {fbeta}")


if __name__ == "__main__":
    training_pipeline()
    print("Training Script finished.", datetime.now())
