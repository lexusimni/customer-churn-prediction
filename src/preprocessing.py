# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop ID if present
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Coerce TotalCharges to numeric and impute
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df

def split_xy(
    df: pd.DataFrame, target: str = "Churn", test_size: float = 0.2, seed: int = 42
):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
