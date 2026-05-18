import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── LOAD MODELS & PREPROCESSORS ────────────────────────────
@st.cache_resource
def load_artifacts():
    xgb_model     = joblib.load("xgb_model.pkl")
    rf_model      = joblib.load("rf_model.pkl")
    cb_model      = joblib.load("catboost_model.pkl")
    stack_model   = joblib.load("stacking_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    scaler        = joblib.load("scaler.pkl")
    return xgb_model, rf_model, cb_model, stack_model, feature_names, scaler

xgb_model, rf_model, cb_model, stack_model, feature_names, scaler = load_artifacts()

NUMERIC_COLS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'total_visits',
    'meds_per_day', 'age_numeric', 'num_meds_changed', 'high_utilizer'
]

# ── ENCODING DICTIONARIES (Fixed & Alphabetical) ───────────
gender_map  = {"Female": 0, "Male": 1}
race_map    = {"African American": 0, "Asian": 1, "Caucasian": 2, "Hispanic": 3, "Other": 4}
insulin_map = {"Down": 0, "No": 1, "Steady": 2, "Up": 3}
change_map  = {"Ch": 0, "No": 1}
diab_map    = {"No": 0, "Yes": 1}
diag_map    = {"Circulatory": 0, "Diabetes": 1, "Digestive": 2, "Genitourinary": 3,
               "Injury": 4, "Musculoskeletal": 5, "Neoplasms": 6, "Other": 7, "Respiratory": 8}

# ── SIDEBAR UI ──────────────────────────────────────────────
st.sidebar.title("🏥 Patient Data Entry")
st.sidebar.markdown("---")

st.sidebar.subheader("🧠 Select AI Architecture")
model_choice = st.sidebar.radio(
    "Choose the prediction engine:",
    ["Random Forest (Robust Bagging)", "XGBoost (Boosted Trees)", 
     "CatBoost (Optuna Tuned)", "Stacking (Meta-Ensemble)"]
)

if "Random Forest" in model_choice:
    active_model, model_name, model_auc = rf_model, "Random Forest", "0.878"
elif "XGBoost" in model_choice:
    active_model, model_name, model_auc = xgb_model, "XGBoost", "0.730"
elif "CatBoost" in model_choice:
    active_model, model_name, model_auc = cb_model, "CatBoost", "0.720"
else:
    active_model, model_name, model_auc = stack_model, "Stacking Ensemble", "0.876"

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Demographics")
age    = st.sidebar.slider("Age", 18, 100, 65)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
race   = st.sidebar.selectbox("Race", ["Caucasian","African American","Hispanic","Asian","Other"])

st.sidebar.subheader("🏨 Admission Details")
num_prev_admissions      = st.sidebar.slider("Prior Inpatient Admissions", 0, 10, 1)
length_of_stay           = st.sidebar.slider("Length of Stay (days)", 1, 30, 5)
admission_type_id        = st.sidebar.selectbox("Admission Type ID", [1,2,3,4,5,6,7,8])
discharge_disposition_id = st.sidebar.selectbox("Discharge Disposition ID", list(range(1,30)))
admission_source_id      = st.sidebar.selectbox("Admission Source ID", list(range(1,26)))

st.sidebar.subheader("💊 Medications & Labs")
num_medications    = st.sidebar.slider("Number of Medications", 0, 30, 8)
num_lab_procedures = st.sidebar.slider("Lab Procedures", 0, 100, 40)
num_procedures     = st.sidebar.slider("Medical Procedures", 0, 10, 1)
number_outpatient  = st.sidebar.slider("Outpatient Visits", 0, 20, 0)
number_emergency   = st.sidebar.slider("Emergency Visits", 0, 20, 0)
insulin            = st.sidebar.selectbox("Insulin", ["No","Steady","Up","Down"])
change_in_meds     = st.sidebar.selectbox("Medication Change", ["No","Ch"])
diabetes_meds      = st.sidebar.selectbox("On Diabetes Medication", ["No","Yes"])

st.sidebar.subheader("🩺 Diagnoses")
num_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 7)
diag_1 = st.sidebar.selectbox("Primary Diagnosis", ["Circulatory","Respiratory","Digestive","Diabetes","Injury","Musculoskeletal","Genitourinary","Neoplasms","Other"])
diag_2 = st.sidebar.selectbox("Secondary Diagnosis", ["Other","Circulatory","Respiratory","Digestive","Diabetes","Injury","Musculoskeletal","Genitourinary","Neoplasms"])
diag_3 = st.sidebar.selectbox("Tertiary Diagnosis", ["Other","Circulatory","Diabetes","Respiratory","Digestive","Injury","Musculoskeletal","Genitourinary","Neoplasms"])

predict_btn = st.sidebar.button("🔍 Predict Readmission Risk", use_container_width=True)

# ── DATA PIPELINE ─────────────────────────────────────────
def encode_and_scale():
    total_visits     = number_outpatient + number_emergency + num_prev_admissions
    meds_per_day     = num_medications / max(length_of_stay, 1)
    high_utilizer    = 1 if total_visits > 3 else 0
    num_meds_changed = 1 if insulin in ["Up","Down"] else 0

    input_dict = {
        "race": race_map.get(race, 2), "gender": gender_map.get(gender, 0),
        "age": min(age // 10, 9), "admission_type_id": float(admission_type_id),
        "discharge_disposition_id": float(discharge_disposition_id), "admission_source_id": float(admission_source_id),
        "time_in_hospital": float(length_of_stay), "num_lab_procedures": float(num_lab_procedures),
        "num_procedures": float(num_procedures), "num_medications": float(num_medications),
        "number_outpatient": float(number_outpatient), "number_emergency": float(number_emergency),
        "number_inpatient": float(num_prev_admissions), "diag_1": diag_map.get(diag_1, 7),
        "diag_2": diag_map.get(diag_2, 7), "diag_3": diag_map.get(diag_3, 7),
        "number_diagnoses": float(num_diagnoses), "max_glu_serum": 2, "A1Cresult": 2,
        "metformin": 1, "repaglinide": 1, "nateglinide": 1, "glimepiride": 1, "glipizide": 1,
        "glyburide": 1, "pioglitazone": 1, "rosiglitazone": 1, "insulin": insulin_map.get(insulin, 1),
        "change": change_map.get(change_in_meds, 1), "diabetesMed": diab_map.get(diabetes_meds, 0),
        "total_visits": float(total_visits), "meds_per_day": float(meds_per_day),
        "high_utilizer": float(high_utilizer), "age_numeric": float(age), "num_meds_changed": float(num_meds_changed),
    }

    df = pd.DataFrame([{f: input_dict.get(f, 0) for f in feature_names}])
    correct_order = scaler.feature_names_in_
    df[correct_order] = scaler.transform(df[correct_order]) 
    return df

# ── MAIN PAGE UI ──────────────────────────────────────────
st.title("🏥 Hospital 30-Day Readmission Risk Predictor")

if not predict_btn:
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Architecture", model_name)
    col2.metric("Engine ROC-AUC", model_auc)
    col3.metric("Features Used", str(len(feature_names)))
    st.info("👈 Fill in patient details in the sidebar and click **Predict Readmission Risk**")
else:
    input_df = encode_and_scale()
    
    prob_rf    = rf_model.predict_proba(input_df)[0][1]
    prob_xgb   = xgb_model.predict_proba(input_df)[0][1]
    prob_cb    = cb_model.predict_proba(input_df)[0][1]
    prob_stack = stack_model.predict_proba(input_df)[0][1]

    prob = active_model.predict_proba(input_df)[0][1]
    
    if prob >= 0.60:
        risk_level, risk_action, risk_fn = "🔴 HIGH RISK", "Immediate care coordination required.", st.error
    elif prob >= 0.35:
        risk_level, risk_action, risk_fn = "🟡 MEDIUM RISK", "Follow-up appointment within 7 days.", st.warning
    else:
        risk_level, risk_action, risk_fn = "🟢 LOW RISK", "Standard discharge process.", st.success

    col_main, col_side = st.columns([2, 1])
    with col_main:
        risk_fn(f"## {risk_level}")
        st.metric(f"Probability ({model_name})", f"{prob*100:.1f}%")
        st.progress(float(prob))
        st.info(f"📋 {risk_action}")
    with col_side:
        st.subheader("Patient Summary")
        st.write(f"**Age:** {age} | **Prior Admits:** {num_prev_admissions}")
        st.write(f"**Meds:** {num_medications} | **Diagnoses:** {num_diagnoses}")

    st.markdown("---")
    st.subheader("⚖️ Model Consensus: The 'Board of Directors'")
    c1, c2, c3 = st.columns(3)
    c1.metric("🌳 Random Forest", f"{prob_rf*100:.1f}%")
    c2.metric("⚡ XGBoost", f"{prob_xgb*100:.1f}%")
    c3.metric("🐱 CatBoost", f"{prob_cb*100:.1f}%")

    st.markdown("#### The Meta-Model (Final Judgement)")
    st.info(f"**Stacking Result: {prob_stack*100:.1f}%**")
    vote_data = pd.DataFrame({"AI Architecture": ["Random Forest", "XGBoost", "CatBoost", "Meta-Model (Final)"], "Predicted Risk (%)": [prob_rf*100, prob_xgb*100, prob_cb*100, prob_stack*100]})
    st.bar_chart(vote_data.set_index("AI Architecture"), color="#e74c3c")

    st.markdown("---")
    st.subheader(f"🔍 Why Did {model_name} Predict This?")
    if "Stacking" in model_name:
        st.warning("⚠️ SHAP values are not natively supported for Stacking Meta-Models. Please select a base model from the sidebar.")
    else:
        try:
            # Calculate the SHAP math in the background
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer(input_df)
        
            # --- 1. SIMPLE ENGLISH EXPLANATION (Comes First) ---
            st.markdown("### 💡 Why Did The Model Predict This?")
            st.write("Here are the primary reasons justifying this specific prediction:")

            # Extract impacts and find the top 5 most important factors
            impacts = shap_values[0, :, 1].values 
            impact_df = pd.DataFrame({
                'Medical Factor': input_df.columns,
                'Impact': impacts
            })
            impact_df['Abs_Impact'] = impact_df['Impact'].abs()
            top_factors = impact_df.sort_values(by='Abs_Impact', ascending=False).head(5)
        
            # Display clean, color-coded reasoning
            for _, row in top_factors.iterrows():
                # Clean up the variable names so they look nice (e.g., "time_in_hospital" -> "Time In Hospital")
                clean_name = row['Medical Factor'].replace('_', ' ').title()
            
                # MUST BE INDENTED INSIDE THE FOR LOOP
                if row['Impact'] > 0.01:
                    st.error(f"⬆️ **{clean_name}** is pushing the readmission risk HIGHER.")
                elif row['Impact'] < -0.01:
                    st.success(f"⬇️ **{clean_name}** is pushing the readmission risk LOWER.")

            # --- 2. THE SMALLER SHAP PLOT (Comes Second) ---
            st.markdown("---")
            st.markdown("#### 📊 Detailed Breakdown Plot")
        
            # fig, ax = plt.subplots(figsize=(6, 4)) creates a much smaller chart
            fig, ax = plt.subplots(figsize=(6, 4))
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Could not generate SHAP chart: {e}")