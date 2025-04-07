from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from datetime import datetime


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)

    param_dist = {
        "n_estimators": range(50, 200, 20),
        "max_depth": range(3, 20),
        "min_samples_split": range(2, 50, 5),
        "min_samples_leaf": range(5, 50, 5),
        "bootstrap": [True, False],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"],
    }

    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, random_state=42)
    random_search.fit(X_train, y_train)

    # Use the best estimator from the random search
    best_params = random_search.best_params_

    param_grid = {
        "n_estimators": [
            best_params["n_estimators"] - 5,
            best_params["n_estimators"],
            best_params["n_estimators"] + 5,
        ],
        "max_depth": [best_params["max_depth"] - 2, best_params["max_depth"], best_params["max_depth"] + 2],
        "min_samples_split": [
            best_params["min_samples_split"] - 1,
            best_params["min_samples_split"],
            best_params["min_samples_split"] + 1,
        ],
        "min_samples_leaf": [
            best_params["min_samples_leaf"] - 1,
            best_params["min_samples_leaf"],
            best_params["min_samples_leaf"] + 1,
        ],
        "bootstrap": [best_params["bootstrap"]],
        "max_features": [best_params["max_features"]],
        "criterion": [best_params["criterion"]],
    }

    # Perform Grid Search
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameters from Grid Search
    final_best_params = grid_search.best_params_

    # Train the final model with the best parameters
    model = RandomForestClassifier(
        n_estimators=final_best_params["n_estimators"],
        max_depth=final_best_params["max_depth"],
        min_samples_split=final_best_params["min_samples_split"],
        min_samples_leaf=final_best_params["min_samples_leaf"],
        bootstrap=final_best_params["bootstrap"],
        max_features=final_best_params["max_features"],
        criterion=final_best_params["criterion"],
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds
