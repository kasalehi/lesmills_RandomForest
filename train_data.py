import pandas as pd

class TrainingModel:
    def __init__(self, path: str):
        self.path = path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def read_data(self) -> pd.DataFrame:
        """Read CSV from the path given at init."""
        df = pd.read_csv(self.path)
        return df

    import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


class TrainingModel:
    def __init__(self, path: str):
        self.path = path
        self.model = None
        self.X_test = None
        self.y_test = None

    # -------- 1) Read data --------
    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        return df

    # -------- 2) Train model (with NaN cleaning) --------
    def training_data(self, df: pd.DataFrame):
        cat_features = ["Origin", "SubCategory", "Term", "PaymentFrequency", "Gender"]
        num_features = ["RegularPayment", "Age", "TotalAttendance"]
        target_col = "Churned"

        # Keep only the columns we need
        cols = cat_features + num_features + [target_col]
        df = df[cols].copy()

        # Replace inf/-inf with NaN just in case
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop any rows with NaN in features or target
        before = len(df)
        df = df.dropna(subset=cols)
        after = len(df)
        print(f"Dropped {before - after} rows due to NaNs in features/target. Remaining: {after}")

        X = df[cat_features + num_features]
        y = df[target_col].astype(int)

        # Final sanity check: no NaNs should remain
        if X.isna().any().any():
            print("Columns with remaining NaNs:")
            print(X.isna().sum())
            raise ValueError("Still have NaNs in X after dropping rows – check data.")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Preprocessing: encode cats + scale nums
        preprocess = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
                ("num", StandardScaler(), num_features),
            ]
        )

        log_reg = LogisticRegression(max_iter=500, class_weight="balanced")

        clf = Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", log_reg),
        ])

        # <<< This is where NaN used to blow up >>>
        clf.fit(X_train, y_train)

        # Store stuff on self for later use
        self.model = clf
        self.X_test = X_test
        self.y_test = y_test

        return clf

    # -------- 3) Metrics on test set --------
    def metrics(self):
        if self.model is None or self.X_test is None or self.y_test is None:
            raise RuntimeError("Model has not been trained yet. Call training_data() first.")

        y_pred = self.model.predict(self.X_test)

        # For ROC-AUC
        if hasattr(self.model, "predict_proba"):
            y_score = self.model.predict_proba(self.X_test)[:, 1]
        else:
            y_score = self.model.decision_function(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_score)
        cm = confusion_matrix(self.y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC:  {auc:.4f}")
        print("Confusion matrix:")
        print(cm)

        return {
            "accuracy": acc,
            "roc_auc": auc,
            "confusion_matrix": cm,
        }

if __name__=="__main__":
    tm = TrainingModel(
    path=r"C:\Users\ksalehi\OneDrive - Les Mills New Zealand Limited\Desktop\lesmills\lesmills_RandomForest\datasets\transform_data.csv"
)
    df = tm.read_data()
    tm.training_data(df)
    tm.metrics()
