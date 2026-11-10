
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocessing import load_data, prepare_features


MODEL_PATH = "models/bike_demand_model.pkl"
DATA_PATH = "data/hour.csv"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)


def load_model():
    """Load the saved model."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found.")

    return joblib.load(MODEL_PATH)


def get_test_data():
    """Recreate the same chronological test split."""

    df = load_data(DATA_PATH)

    X, y = prepare_features(df)

    split_index = int(len(X) * 0.8)

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    # Retain original columns for business-level analysis
    raw_test = df.iloc[split_index:].copy()

    return X_test, y_test, raw_test


def evaluate_predictions(y_test, predictions):
    """Calculate final regression metrics."""

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    print("\n========== EVALUATION METRICS ==========")

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    return mae, rmse, r2


def feature_importance_analysis(model, feature_names):
    """Identify important features."""

    if not hasattr(model, "feature_importances_"):
        print("Feature importance unavailable for this model.")
        return

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========== FEATURE IMPORTANCE ==========")

    print(importance.head(10).to_string(index=False))

    importance.to_csv(
        f"{REPORT_DIR}/feature_importance.csv",
        index=False
    )

    # Visualisation
    top_features = importance.head(12).sort_values(
        "Importance"
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title("Top Features Influencing Bike Demand")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/feature_importance.png",
        dpi=150
    )

    plt.close()


def prediction_analysis(y_test, predictions):
    """Analyse actual versus predicted demand."""

    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    results["Residual"] = (
        results["Actual"] - results["Predicted"]
    )

    results["Absolute_Error"] = (
        results["Residual"].abs()
    )

    results.to_csv(
        f"{REPORT_DIR}/prediction_results.csv",
        index=False
    )

    print("\n========== PREDICTION ANALYSIS ==========")

    print(
        "Average actual demand:",
        round(results["Actual"].mean(), 2)
    )

    print(
        "Average predicted demand:",
        round(results["Predicted"].mean(), 2)
    )

    print(
        "Maximum absolute error:",
        round(results["Absolute_Error"].max(), 2)
    )

    print(
        "Underpredictions:",
        (results["Residual"] > 0).sum()
    )

    print(
        "Overpredictions:",
        (results["Residual"] < 0).sum()
    )

    # Actual vs predicted
    plt.figure(figsize=(8, 6))

    plt.scatter(
        results["Actual"],
        results["Predicted"],
        alpha=0.35
    )

    max_value = max(
        results["Actual"].max(),
        results["Predicted"].max()
    )

    plt.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.title("Actual vs Predicted Bike Demand")

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/actual_vs_predicted.png",
        dpi=150
    )

    plt.close()

    # Residual distribution
    plt.figure(figsize=(8, 5))

    plt.hist(
        results["Residual"],
        bins=40
    )

    plt.axvline(0, linestyle="--")

    plt.title("Prediction Error Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/residual_distribution.png",
        dpi=150
    )

    plt.close()

    return results


def hourly_error_analysis(results, raw_test):
    """Identify hours where model errors are high."""

    analysis = raw_test[
        ["hr", "weathersit", "workingday"]
    ].reset_index(drop=True).copy()

    analysis["Absolute_Error"] = (
        results["Absolute_Error"].values
    )

    analysis["Actual"] = results["Actual"].values

    hourly = (
        analysis.groupby("hr")
        .agg(
            MAE=("Absolute_Error", "mean"),
            Average_Demand=("Actual", "mean")
        )
        .reset_index()
    )

    print("\n========== HOURLY ERROR ANALYSIS ==========")

    print(
        hourly.sort_values(
            "MAE", ascending=False
        ).head(10).to_string(index=False)
    )

    hourly.to_csv(
        f"{REPORT_DIR}/hourly_error_analysis.csv",
        index=False
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        hourly["hr"],
        hourly["MAE"]
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Absolute Error")
    plt.title("Prediction Error by Hour")

    plt.xticks(range(0, 24))

    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/hourly_error.png",
        dpi=150
    )

    plt.close()

    # Weather-level error
    weather_analysis = (
        analysis.groupby("weathersit")
        .agg(
            MAE=("Absolute_Error", "mean"),
            Average_Demand=("Actual", "mean"),
            Records=("Actual", "count")
        )
        .reset_index()
    )

    print("\n========== WEATHER-WISE ERROR ==========")

    print(weather_analysis.to_string(index=False))

    weather_analysis.to_csv(
        f"{REPORT_DIR}/weather_error_analysis.csv",
        index=False
    )


def main():

    print("Loading trained model...")

    model = load_model()

    X_test, y_test, raw_test = get_test_data()

    print("Test records:", len(X_test))

    print("\nGenerating predictions...")

    predictions = np.maximum(
        model.predict(X_test),
        0
    )

    # Evaluation
    evaluate_predictions(y_test, predictions)

    # Interpretability
    feature_importance_analysis(
        model,
        X_test.columns.tolist()
    )

    # Error analysis
    results = prediction_analysis(
        y_test,
        predictions
    )

    # Business-level analysis
    hourly_error_analysis(
        results,
        raw_test
    )

    print("\nAll evaluation reports generated.")
    print("Check the reports/ directory.")


if __name__ == "__main__":
    main()
