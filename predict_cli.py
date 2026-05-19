from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/student_grade_model.joblib")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model belum ada. Jalankan dulu: python train_model.py"
        )

    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["metadata"]["feature_columns"]

    sample = {
        "Age": 20,
        "Gender": "Female",
        "Hours_Studied": 6.5,
        "Attendance": 85.0,
        "Sleep_Hours": 7.0,
        "Stress_Level": 3.0,
        "Screen_Time": 2.5,
        "Previous_GPA": 3.2,
        "Part_Time_Job": "No",
        "Study_Method": "Hybrid",
        "Diet_Quality": "Good",
        "Internet_Quality": "Good",
        "Extracurricular": "Yes",
        "Tutoring_Sessions_Per_Week": 2,
        "Family_Income_Level": "Middle",
        "Exam_Anxiety_Score": 2.5,
    }

    X_new = pd.DataFrame([sample], columns=feature_columns)
    prediction = pipeline.predict(X_new)[0]
    probabilities = pipeline.predict_proba(X_new)[0]

    print(f"Prediksi Grade: {prediction}")
    print("\nProbabilitas:")
    for label, probability in zip(pipeline.classes_, probabilities):
        print(f"- {label}: {probability:.2%}")


if __name__ == "__main__":
    main()
