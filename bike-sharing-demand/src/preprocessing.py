
import pandas as pd
import numpy as np


TARGET = "cnt"

LEAKAGE_COLUMNS = [
    "instant",
    "casual",
    "registered"
]


def load_data(file_path="data/hour.csv"):
    """Load and validate the bike-sharing dataset."""

    df = pd.read_csv(file_path)

    df["dteday"] = pd.to_datetime(df["dteday"])

    df = df.sort_values(
        ["dteday", "hr"]
    ).reset_index(drop=True)

    return df


def data_quality_report(df):
    """Generate basic data quality information."""

    print("\n========== DATA QUALITY REPORT ==========")

    print("\nDataset Shape:", df.shape)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nTarget Statistics:")
    print(df[TARGET].describe())

    print("\nUnique Values:")
    print(df.nunique())

    print("\nDate Range:")
    print(df["dteday"].min(), "to", df["dteday"].max())

    print("\n=========================================")


def feature_engineering(df):
    """Create useful time-based features."""

    df = df.copy()

    # Cyclical encoding of hour
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hr"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hr"] / 24
    )

    # Cyclical encoding of weekday
    df["weekday_sin"] = np.sin(
        2 * np.pi * df["weekday"] / 7
    )

    df["weekday_cos"] = np.cos(
        2 * np.pi * df["weekday"] / 7
    )

    # Cyclical encoding of month
    df["month_sin"] = np.sin(
        2 * np.pi * (df["mnth"] - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (df["mnth"] - 1) / 12
    )

    # Peak commuting hours
    df["is_peak_hour"] = df["hr"].isin(
        [7, 8, 9, 16, 17, 18, 19]
    ).astype(int)

    # Weekend indicator
    df["is_weekend"] = (
        df["weekday"].isin([0, 6])
    ).astype(int)

    # Remove original variables represented by cyclical features
    df.drop(
        columns=["hr", "weekday", "mnth", "dteday"],
        inplace=True
    )

    return df


def prepare_features(df):
    """Separate features and target without data leakage."""

    df = feature_engineering(df)

    X = df.drop(
        columns=LEAKAGE_COLUMNS + [TARGET]
    )

    y = df[TARGET]

    return X, y


def chronological_split(X, y, test_size=0.2):
    """Split data while preserving chronological order."""

    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    df = load_data()

    data_quality_report(df)

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = chronological_split(
        X, y
    )

    print("\n========== PREPROCESSING RESULTS ==========")

    print("Feature Shape:", X.shape)

    print("Training Shape:", X_train.shape)

    print("Testing Shape:", X_test.shape)

    print("\nFeatures Used:")
    print(X.columns.tolist())

    print("\nTarget:", TARGET)

    print("\nPreprocessing completed successfully.")
