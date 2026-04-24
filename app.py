import streamlit as st
import pandas as pd
import pickle
import os

# ==========================
# Page Config
# ==========================
st.set_page_config(page_title="Productivity Insight Engine", page_icon="📊")

st.title("📊 Productivity Insight Engine")
st.write("AI-based Productivity Prediction System")

# ==========================
# Load Model Safely
# ==========================
@st.cache_resource
def load_files():
    scaler, model = None, None
    try:
        scaler_path = "notebook/scaler.pkl"
        model_path = "notebook/best_model.pkl"

        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)

    except Exception as e:
        st.warning(f"⚠️ Loading Issue: {e}")

    return scaler, model


scaler, model = load_files()

# ==========================
# Inputs
# ==========================
work_hours = st.slider("💼 Work Hours", 0.0, 24.0, 6.0)
sleep_hours = st.slider("😴 Sleep Hours", 0.0, 24.0, 7.0)
distractions = st.slider("📱 Distractions", 0.0, 10.0, 2.0)

# ==========================
# Prediction Function
# ==========================
def predict(work, sleep, distract):

    # 🟡 अगर model नहीं मिला → fallback
    if model is None or scaler is None:
        score = (work * 0.6 + sleep * 0.3 - distract * 0.5)
        return max(0, min(score, 10))

    try:
        # 🔥 original feature names auto detect
        feature_names = scaler.feature_names_in_

        data = pd.DataFrame(columns=feature_names)

        for col in feature_names:
            if col.lower() in ["work_hours", "duration", "hour"]:
                data.loc[0, col] = work
            elif col.lower() in ["sleep_hours", "energy"]:
                data.loc[0, col] = sleep
            elif col.lower() in ["distractions", "noise"]:
                data.loc[0, col] = distract
            else:
                data.loc[0, col] = 0   # बाकी columns default

        # transform + predict
        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)

        return float(pred[0])

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None


# ==========================
# Button
# ==========================
if st.button("🚀 Predict"):

    result = predict(work_hours, sleep_hours, distractions)

    if result is not None:
        st.success(f"📈 Productivity Score: {result:.2f}")

        # progress bar
        st.progress(int(max(0, min(result * 10, 100))))

        # smart feedback
        if result > 7:
            st.success("🔥 High Productivity")
        elif result > 4:
            st.info("👍 Moderate Productivity")
        else:
            st.warning("⚠️ Low Productivity")
    else:
        st.error("Prediction Failed")