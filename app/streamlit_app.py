# ==============================================
# Clinical Diabetes Risk Scorer - Streamlit App
# Author: Divya Prakash
# Dataset: NHANES 2017-2018 (CDC)
# ==============================================

import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clinical Diabetes Risk Score", page_icon='🩺', layout="wide")

# ── Load Artifacts ──────────────────────────────

@st.cache_resource
def load_artifacts():
    with open('src/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('src/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('src/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    with open('src/explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    return model, scaler, feature_columns, explainer

model, scaler, feature_columns, explainer = load_artifacts()

# ── Title ───────────────────────────────────────

st.title("🩺 Clinical Diabetes Risk Scorer")
st.markdown("""
Enter a patient's blood test results and demographics to calculate 
their diabetes risk score with clinical explainability.

> **Dataset:** CDC NHANES 2017-2018 | **Model:** Logistic Regression | 
> **ROC-AUC:** 0.850 | **Recall:** 0.872
""")
st.divider()

# ── Sidebar Inputs ──────────────────────────────

st.sidebar.header("🧪 Patient Blood Test Values")

st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age (years)", 18, 80, 45)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
bmi = st.sidebar.slider("BMI", 15.0, 60.0, 27.0)

st.sidebar.subheader("Complete Blood Count (CBC)")
wbc = st.sidebar.slider("WBC — White Blood Cells (SI)", 1.0, 20.0, 7.0,
                         help="Normal range: 4.5-11.0 * 10⁹/L")
rbc = st.sidebar.slider("RBC — Red Blood Cells (SI)", 2.0, 7.0, 4.5,
                         help="Normal: Male 4.7-6.1, Female 4.2-5.4 * 10¹²/L")
hemoglobin = st.sidebar.slider("Hemoglobin (g/dL)", 6.0, 20.0, 14.0,
                                help="Normal: Male 13.5-17.5, Female 12.0-15.5 g/dL")
hematocrit = st.sidebar.slider("Hematocrit (%)", 20.0, 60.0, 42.0,
                                help="Normal: Male 41-53%, Female 36-46%")
mcv = st.sidebar.slider("MCV — Mean Corpuscular Volume (fL)", 60.0, 120.0, 90.0,
                         help="Normal range: 80-100 fL")
platelets = st.sidebar.slider("Platelets (SI)", 50.0, 500.0, 250.0,
                               help="Normal range: 150-400 * 10⁹/L")

# ── Predict Button ──────────────────────────────

predict_btn = st.sidebar.button("🔍 Calculate Risk Score", use_container_width=True)

# ── Preprocessing Function ──────────────────────

def preprocess_input():
    #Engineering feature
    age_bmi = age * bmi
    anemia_flag = 1 if hemoglobin < 12.0 else 0
    high_wbc_flag = 1 if wbc > 11.0 else 0
    gender_encoded = 1 if gender == 'Male' else 0
    
    input_dict = {
        'WBC': wbc,
        'RBC': rbc,
        'Hemoglobin': hemoglobin,
        'Hematocrit': hematocrit,
        'MCV': mcv,
        'Platelets': platelets,
        'Age': age,
        'Gender': gender_encoded,
        'BMI': bmi,
        'Age_BMI_interaction': age_bmi,
        'Anemia_flag': anemia_flag,
        'High_WBC_flag': high_wbc_flag
    }
    
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)
    
    return input_df, input_df_scaled

# ── Output ──────────────────────────────────────

if predict_btn:
    try:
        input_raw, input_scaled = preprocess_input()
        risk_prob = model.predict_proba(input_scaled)[0][1]
        risk_pred = model.predict(input_scaled)[0]
        
        #Risk Tiers
        if risk_prob < 0.3:
            risk_level = "🟢 Low Risk"
            risk_color = "green"
            risk_advice = "Routine monitoring recommended. Encourage healthy lifestyle maintenance."
        elif risk_prob < 0.6:
            risk_level = "🟡 Medium Risk"
            risk_color = "orange"
            risk_advice = "Consider HbA1c test. Review diet, physical activity, and weight management."
        else:
            risk_level = "🔴 High Risk"
            risk_color = "red"
            risk_advice = "Immediate clinical follow-up recommended. Fasting glucose and HbA1c testing advised."
            
        # ── Risk Score Panel ─────────────────────
        
        st.subheader("🎯Risk Assessment Header")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Diabetes Prediction",
                      "High Risk ❌" if risk_pred == 1 else "Low Risk ✅")
        with col2:
            st.metric("Risk Probability", f"{risk_prob:.1%}")
        with col3:
            st.metric("Risk Category", risk_level)

        # Clinical advice
        if risk_color == "red":
            st.error(f"**Clinical Recommendation:** {risk_advice}")
        elif risk_color == "orange":
            st.warning(f"**Clinical Recommendation:** {risk_advice}")
        else:
            st.success(f"**Clinical Recommendation:** {risk_advice}")

        st.divider()
        
        # ── Risk Gauge ───────────────────────────
        st.subheader("📈 Risk Probability Gauge")

        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.barh(["Diabetes Risk"], [risk_prob],
                color=risk_color, height=0.4)
        ax.barh(["Diabetes Risk"], [1 - risk_prob],
                left=[risk_prob], color='lightgrey', height=0.4)
        ax.axvline(0.3, color='orange', linestyle='--',
                   linewidth=1, label='Medium Risk (0.3)')
        ax.axvline(0.6, color='red', linestyle='--',
                   linewidth=1, label='High Risk (0.6)')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Risk Probability")
        ax.set_title(f"Diabetes Risk Score: {risk_prob:.1%}")
        ax.legend(loc='lower right')
        st.pyplot(fig)

        st.divider()
        
        # ── SHAP Explanation ─────────────────────
        
        st.subheader("🔬 Why This Risk Score? (SHAP Explanation)")
        st.markdown("Shows which blood test values are **driving this patient's risk score**.")

        shap_vals = explainer.shap_values(input_scaled)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=input_scaled.iloc[0],
                feature_names=feature_columns
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig2)

        st.divider()
        
        # ── Clinical Flags ───────────────────────
        
        st.subheader("⚠️ Clinical Risk Flags")

        flags = []
        if age > 45:
            flags.append("📌 Age > 45 — risk increases significantly after 45")
        if bmi > 30:
            flags.append("📌 BMI > 30 — obesity is a primary Type 2 diabetes risk factor")
        if hemoglobin < 12.0:
            flags.append("📌 Low Hemoglobin — possible anemia, associated with diabetic complications")
        if wbc > 11.0:
            flags.append("📌 Elevated WBC — chronic inflammation marker associated with diabetes")
        if bmi > 25 and age > 40:
            flags.append("📌 Overweight + Age > 40 — combined metabolic risk factor")

        if flags:
            for flag in flags:
                st.warning(flag)
        else:
            st.success("✅ No major individual risk flags detected.")

        st.divider()
        
        # ── Patient Summary Table ────────────────
        
        st.subheader("📋 Patient Input Summary")

        summary_df = pd.DataFrame({
            'Parameter': ['Age', 'Gender', 'BMI', 'WBC', 'RBC',
                         'Hemoglobin', 'Hematocrit', 'MCV', 'Platelets'],
            'Value': [age, gender, bmi, wbc, rbc,
                     hemoglobin, hematocrit, mcv, platelets],
            'Normal Range': [
                '18-80', 'M/F', '18.5-24.9',
                '4.5-11.0', '4.2-6.1',
                '12.0-17.5', '36-53%',
                '80-100 fL', '150-400'
            ]
        })
        st.dataframe(summary_df, use_container_width=True)

        st.divider()
        st.caption("""
        Built by Divya Prakash | CDC NHANES 2017-2018 |
        Logistic Regression | ROC-AUC: 0.850 | Recall: 0.872 |
        ⚠️ For educational purposes only — not a substitute for clinical diagnosis
        """)
        
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.info("Check that all .pkl files exist in the src/ folder")
        
else:
    st.info("👈 Enter patient blood test values in the sidebar and click **Calculate Risk Score**.")
