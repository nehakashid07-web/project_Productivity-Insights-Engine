import pickle
import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# Prediction Function
# ==========================
def predict_productivity(work_hours, sleep_hours, distractions, scaler_path, model_path):
    try:
        # load the scaler
        with open(scaler.pkl, 'rb') as file1:
            pickle.dump(scaler,file1)

        # load the model
        with open(model.pkl, 'wb') as file2:
            pickle.dump(best_model,file2)

        # prepare input data (IMPORTANT: must match training columns)
        dct = {
            'work_hours': [work_hours],
            'sleep_hours': [sleep_hours],
            'distractions': [distractions]
        }

        x_new = pd.DataFrame(dct)

        # Transform input data
        xnew_pre = scaler.transform(x_new)

        # make predictions
        pred = model.predict(xnew_pre)

        return pred

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


# ==========================
# Streamlit UI
# ==========================
st.title("📊 Productivity Insight Engine")

# input fields
work_hours = st.number_input("Work Hours", min_value=0.0, step=0.5, value=6.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, step=0.5, value=7.0)
distractions = st.number_input("Distractions", min_value=0.0, step=0.5, value=2.0)

# ==========================
# Prediction Button
# ==========================
if st.button("Predict"):
    # file paths (IMPORTANT FIX)
    scaler_path = "notebook/scaler.pkl"
    model_path = "notebook/model.pkl"

    # call function
    pred = predict_productivity(
        work_hours, sleep_hours, distractions,
        scaler_path, model_path
    )

    # Display results
    if pred is not None:
        value = float(pred[0])
        st.subheader(f'📈 Predicted Productivity: {value:.2f}')
        st.progress(min(int(value * 10), 100))
    else:
        st.error("Prediction Failed. Check model files.")