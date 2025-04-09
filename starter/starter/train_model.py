# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.slices import data_slices
import os
import joblib
from ml.model import train_model, inference, compute_model_metrics
import pandas as pd
from datetime import datetime


test_processed_dataset = None


def load_data():
    """
    Load the data from the CSV file.
    """
    data = pd.read_csv(
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
    train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data["salary"])

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
    model = train_model(X_train, y_train, fine_tuning=True)
    base_path = "model"
    os.makedirs(base_path, exist_ok=True)
    joblib.dump(model, os.path.join(base_path, "model.pkl"))
    joblib.dump(encoder, os.path.join(base_path, "encoder.pkl"))
    joblib.dump(lb, os.path.join(base_path, "label_binarizer.pkl"))

    pred = inference(model, X_test)
    copy_df = pd.DataFrame(test).copy()
    copy_df["pred"] = pd.Series(pred, index=copy_df.index)
    copy_df["salary_binary"] = copy_df["salary"].apply(lambda x: 1 if x == ">50K" else 0)
    global test_processed_dataset
    test_processed_dataset = copy_df

    precision, recall, fbeta = compute_model_metrics(y_test, pred)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Fbeta: {fbeta}")


def slices_pipeline():
    """
    Function to create slices of the data for each categorical feature.
    """
    slices = data_slices(test_processed_dataset, "salary")
    # Compute metrics for each slice
    for slice_name, slice_data in slices.items():
        print(f"Metrics for slice: {slice_name}")
        for slice_key, slice_value in slice_data.items():
            print(f"\tSlice Bucket: {slice_key}")
            # Process the slice data
            y_slice = slice_value["salary_binary"]
            pred_slice = slice_value["pred"]
            # Compute metrics for the slice
            precision, recall, fbeta = compute_model_metrics(y_slice, pred_slice)
            print(f"\t\tTotal: {slice_value.shape[0]}\n")
            print(f"\t\t<=50K counts: {slice_value[slice_value['salary_binary'] == 0].shape[0]}")
            print(f"\t\t>50K counts: {slice_value[slice_value['salary_binary'] == 1].shape[0]}\n")
            print(f"\t\tPrecision: {precision}")
            print(f"\t\tRecall: {recall}")
            print(f"\t\tFbeta: {fbeta}")
            print()


if __name__ == "__main__":
    training_pipeline()
    print("Training Script finished.", datetime.now())
    slices_pipeline()
    print("Slices Script finished.", datetime.now())
