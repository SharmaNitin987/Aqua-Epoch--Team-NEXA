from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "water_potability_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"

st.set_page_config(page_title="The Aqua-Epoch", page_icon="💧", layout="wide")

st.markdown("""
<style>
.main {background: linear-gradient(180deg, #f5fbff 0%, #ffffff 55%);}
.hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#063b5c,#087e8b); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin-bottom:.25rem;}
.hero p {color:#e5f8ff; margin-bottom:0;}
.small-note {font-size: .9rem; color: #52616b;}
</style>
<div class="hero">
  <h1>💧 The Aqua-Epoch</h1>
  <p>Water quality insights powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("The trained model is not available yet.")
    st.markdown("**First run this command in your VS Code terminal:**")
    st.code("python train_model.py", language="bash")
    st.write("This creates the model and metadata files in the `models` folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

features = meta["features"]
defaults = meta["defaults"]
metric = meta.get("metrics", {})

with st.sidebar:
    st.header("About this project")
    st.write("Enter measured water-quality values to receive a model estimate.")
    st.caption("Use the units and measurement methods expected by your dataset/laboratory.")
    st.divider()
    st.write("**Model selected:**", meta.get("best_model", "Trained classifier"))
    st.write("**Training rows:**", meta.get("rows_used", "—"))
    st.caption("This is an educational screening aid, not a certified drinking-water test.")

st.subheader("Enter water quality measurements")
st.write("Fill in the available laboratory measurements. Values are initialized to training-set medians; replace them with your sample's actual results.")

# Arrange inputs in three columns for a cleaner dashboard.
values = {}
columns = st.columns(3)
for i, feature in enumerate(features):
    with columns[i % 3]:
        default = float(defaults.get(feature, 0.0))
        # Some datasets have extreme values; number_input avoids falsely imposing regulatory limits.
        values[feature] = st.number_input(
            feature.replace("_", " "),
            value=default,
            format="%.5f",
            key=f"input_{feature}",
            help="Enter the measured value in the same units and scale used in the training dataset."
        )

if st.button("Analyze water sample", type="primary", use_container_width=True):
    row = pd.DataFrame([[values[f] for f in features]], columns=features)

    # Basic impossible-value guard for measurements that cannot be negative.
    impossible = [f for f in ["pH", "Hardness", "Fluoride"] if f in row.columns and row.iloc[0][f] < 0]
    if impossible:
        st.error("Please check these measurements; negative values are not physically valid for: " + ", ".join(impossible))
    else:
        prediction = int(model.predict(row)[0])
        st.divider()
        st.subheader("Prediction result")
        if prediction == 1:
            st.success("Model prediction: Potable class (label 1)")
            st.write("The model classifies this sample as potable based on patterns in the supplied dataset.")
        else:
            st.warning("Model prediction: Non-potable class (label 0)")
            st.write("The model classifies this sample as non-potable based on patterns in the supplied dataset.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            class_labels = list(model.classes_)
            if 1 in class_labels:
                p = float(probabilities[class_labels.index(1)])
                st.metric("Model-estimated probability of label 1", f"{p:.1%}")
                st.progress(max(0.0, min(1.0, p)))
                st.caption("This is a model score, not a calibrated guarantee that the water is safe.")

        st.info("Important: Do not use this prediction alone to decide whether water is safe to drink. Confirm safety with an accredited laboratory and applicable drinking-water standards, including microbial testing.")

with st.expander("How to interpret the result"):
    st.markdown("""
    - **Potable class (1):** the model predicts the dataset's positive class.
    - **Non-potable class (0):** the model predicts the dataset's negative class.
    - The probability is the model's estimate, not the percentage of contaminants or a safety certification.
    - Model reliability depends on the dataset quality and how closely a new sample matches the training data.
    """)

with st.expander("Model evaluation (held-out test set)"):
    if metric:
        cols = st.columns(4)
        for col, label, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1 score"], ["accuracy", "precision", "recall", "f1"]):
            val = metric.get(key)
            col.metric(label, f"{float(val):.3f}" if val is not None else "—")
        st.caption("Scores are from a held-out test split of this dataset. They do not establish real-world water safety.")
    else:
        st.write("Run `python train_model.py` to create evaluation metrics.")

st.caption("The Aqua-Epoch • Educational machine-learning prototype")
