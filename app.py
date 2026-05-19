from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/student_grade_model.joblib")
DATA_PATH = Path("student_performance_grade.csv")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)


st.set_page_config(page_title="Prediksi Grade Siswa", layout="wide")
st.title("Prediksi Grade Siswa")

if not MODEL_PATH.exists():
    st.error("Model belum ditemukan. Jalankan `python train_model.py` terlebih dahulu.")
    st.stop()

artifact = load_artifact()
pipeline = artifact["pipeline"]
metadata = artifact["metadata"]
df = load_reference_data()

feature_columns = metadata["feature_columns"]
numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]

st.sidebar.header("Input Siswa")

inputs = {}
for column in feature_columns:
    if column in numeric_features:
        series = df[column]
        min_value = float(series.min())
        max_value = float(series.max())
        mean_value = float(series.mean())

        if pd.api.types.is_integer_dtype(series):
            inputs[column] = st.sidebar.number_input(
                column,
                min_value=int(min_value),
                max_value=int(max_value),
                value=int(round(mean_value)),
                step=1,
            )
        else:
            inputs[column] = st.sidebar.number_input(
                column,
                min_value=min_value,
                max_value=max_value,
                value=round(mean_value, 2),
                step=0.1,
            )
    elif column in categorical_features:
        options = sorted(df[column].dropna().unique().tolist())
        inputs[column] = st.sidebar.selectbox(column, options)

X_new = pd.DataFrame([inputs], columns=feature_columns)
prediction = pipeline.predict(X_new)[0]
probabilities = pd.DataFrame(
    {
        "Grade": pipeline.classes_,
        "Probabilitas": pipeline.predict_proba(X_new)[0],
    }
).sort_values("Probabilitas", ascending=False)

left, right = st.columns([1, 2])

with left:
    st.metric("Prediksi Grade", prediction)
    st.dataframe(X_new, use_container_width=True, hide_index=True)

with right:
    st.subheader("Probabilitas Tiap Grade")
    st.bar_chart(probabilities.set_index("Grade"))
    probabilities["Probabilitas"] = probabilities["Probabilitas"].map(lambda x: f"{x:.2%}")
    st.dataframe(probabilities, use_container_width=True, hide_index=True)
