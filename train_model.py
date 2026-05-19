from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("student_performance_grade.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_PATH = MODEL_DIR / "student_grade_model.joblib"

TARGET = "Grade"
DROP_COLUMNS = ["Student_ID"]
RANDOM_STATE = 42


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    labels = sorted(y.unique().tolist())
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    matrix = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels),
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    metadata = {
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET,
        "labels": labels,
    }
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, MODEL_PATH)

    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    matrix.to_csv(REPORT_DIR / "confusion_matrix.csv")

    print(f"Model tersimpan di: {MODEL_PATH}")
    print(f"Classification report tersimpan di: {REPORT_DIR / 'classification_report.txt'}")
    print(f"Confusion matrix tersimpan di: {REPORT_DIR / 'confusion_matrix.csv'}")
    print()
    print(report)


if __name__ == "__main__":
    main()
