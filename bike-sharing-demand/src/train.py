
import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_score,
    RandomizedSearchCV
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

from src.preprocessing import (
    load_data,
    prepare_features,
    chronological_split
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/hour.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 3


# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

def get_data():

    df = load_data(DATA_PATH)

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = chronological_split(
        X, y
    )

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

def get_models():

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=25,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        )
    }

    # Include XGBoost if installed
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )

        print("XGBoost available. Added to comparison.")

    except ImportError:
        print("XGBoost not installed. Skipping it.")

    return models


# --------------------------------------------------
# CROSS VALIDATION MODEL COMPARISON
# --------------------------------------------------

def compare_models(X_train, y_train, models):

    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    results = []
    trained_models = {}

    print("\n========== MODEL COMPARISON ==========")

    for name, model in models.items():

        print(f"\nEvaluating: {name}")

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=1
        )

        mean_mae = -scores.mean()
        std_mae = scores.std()

        print(f"CV MAE: {mean_mae:.2f}")
        print(f"CV Std: {std_mae:.2f}")

        results.append({
            "Model": name,
            "CV_MAE": mean_mae,
            "CV_Std": std_mae
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="CV_MAE"
    ).reset_index(drop=True)

    print("\nMODEL RANKING:")
    print(results_df.to_string(index=False))

    return results_df, trained_models


# --------------------------------------------------
# HYPERPARAMETER OPTIMISATION
# --------------------------------------------------

def tune_model(model_name, model, X_train, y_train):

    print(f"\n========== OPTIMISING {model_name} ==========")

    param_distributions = {

        "Random Forest": {
            "n_estimators": [150, 200, 250],
            "max_depth": [15, 20, 25, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [0.7, 1.0]
        },

        "Hist Gradient Boosting": {
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "max_iter": [100, 150, 200, 250],
            "max_leaf_nodes": [15, 20, 25, 31],
            "l2_regularization": [0.0, 0.5, 1.0, 2.0]
        },

        "XGBoost": {
            "n_estimators": [150, 200, 250, 300],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
        }
    }

    if model_name not in param_distributions:
        print("No tuning required for this model.")
        return model, None

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions[model_name],
        n_iter=8,
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)

    print(
        "\nBest CV MAE:",
        round(-search.best_score_, 2)
    )

    return search.best_estimator_, search.best_params_


# --------------------------------------------------
# FINAL TEST EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(
        y_test, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test, predictions
    )

    metrics = {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4)
    }

    print("\n========== FINAL TEST RESULTS ==========")

    print("MAE:", metrics["MAE"])
    print("RMSE:", metrics["RMSE"])
    print("R2:", metrics["R2"])

    return metrics, predictions


# --------------------------------------------------
# SAVE MODEL AND RESULTS
# --------------------------------------------------

def save_artifacts(model, metrics, comparison, feature_names):

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, "bike_demand_model.pkl")
    )

    comparison.to_csv(
        os.path.join(MODEL_DIR, "model_comparison.csv"),
        index=False
    )

    with open(
        os.path.join(MODEL_DIR, "metrics.json"),
        "w"
    ) as file:

        json.dump(metrics, file, indent=4)

    with open(
        os.path.join(MODEL_DIR, "feature_names.json"),
        "w"
    ) as file:

        json.dump(feature_names, file, indent=4)

    print("\nModel and evaluation results saved successfully.")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("\nLoading dataset...")

    X_train, X_test, y_train, y_test = get_data()

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    models = get_models()

    # Step 1: Compare models using chronological CV
    comparison, trained_models = compare_models(
        X_train,
        y_train,
        models
    )

    # Step 2: Select best model using CV MAE
    best_name = comparison.iloc[0]["Model"]
    best_model = trained_models[best_name]

    print("\nSelected model:", best_name)

    # Step 3: Tune selected model
    final_model, best_params = tune_model(
        best_name,
        best_model,
        X_train,
        y_train
    )

    # Step 4: Fit final model on training data
    final_model.fit(X_train, y_train)

    # Step 5: Evaluate once on untouched test data
    metrics, predictions = evaluate_model(
        final_model,
        X_test,
        y_test
    )

    metrics["Selected_Model"] = best_name
    metrics["Best_Parameters"] = best_params

    # Step 6: Save artifacts
    save_artifacts(
        final_model,
        metrics,
        comparison,
        X_train.columns.tolist()
    )

    print("\n========== SAMPLE PREDICTIONS ==========")

    sample = pd.DataFrame({
        "Actual": y_test.iloc[:10].values,
        "Predicted": np.round(predictions[:10], 0)
    })

    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
