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

# ── LOAD MODEL & PREPROCESSORS ────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("best_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    scaler        = joblib.load("scaler.pkl")
    num_imputer   = joblib.load("num_imputer.pkl")
    return model, feature_names, scaler, num_imputer

model, feature_names, scaler, num_imputer = load_artifacts()

# These are the numeric columns that were scaled in Notebook 2
# Must match EXACTLY what was scaled during training
NUMERIC_COLS = [
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'total_visits',
    'meds_per_day', 'age_numeric', 'num_meds_changed', 'high_utilizer'
]

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("🏥 Patient Data Entry")
st.sidebar.markdown("---")

st.sidebar.subheader("👤 Demographics")
age    = st.sidebar.slider("Age", 18, 100, 65)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
race   = st.sidebar.selectbox("Race",
            ["Caucasian","African American","Hispanic","Asian","Other"])

st.sidebar.markdown("---")
st.sidebar.subheader("🏨 Admission Details")
num_prev_admissions      = st.sidebar.slider("Prior Inpatient Admissions", 0, 10, 1)
length_of_stay           = st.sidebar.slider("Length of Stay (days)", 1, 30, 5)
admission_type_id        = st.sidebar.selectbox("Admission Type ID", [1,2,3,4,5,6,7,8])
discharge_disposition_id = st.sidebar.selectbox("Discharge Disposition ID", list(range(1,30)))
admission_source_id      = st.sidebar.selectbox("Admission Source ID", list(range(1,26)))

st.sidebar.markdown("---")
st.sidebar.subheader("💊 Medications & Labs")
num_medications    = st.sidebar.slider("Number of Medications", 0, 30, 8)
num_lab_procedures = st.sidebar.slider("Lab Procedures", 0, 100, 40)
num_procedures     = st.sidebar.slider("Medical Procedures", 0, 10, 1)
number_outpatient  = st.sidebar.slider("Outpatient Visits", 0, 20, 0)
number_emergency   = st.sidebar.slider("Emergency Visits", 0, 20, 0)
insulin            = st.sidebar.selectbox("Insulin", ["No","Steady","Up","Down"])
change_in_meds     = st.sidebar.selectbox("Medication Change", ["No","Ch"])
diabetes_meds      = st.sidebar.selectbox("On Diabetes Medication", ["No","Yes"])

st.sidebar.markdown("---")
st.sidebar.subheader("🩺 Diagnoses")
num_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 7)
diag_1 = st.sidebar.selectbox("Primary Diagnosis",
            ["Circulatory","Respiratory","Digestive","Diabetes",
             "Injury","Musculoskeletal","Genitourinary","Neoplasms","Other"])
diag_2 = st.sidebar.selectbox("Secondary Diagnosis",
            ["Other","Circulatory","Respiratory","Digestive","Diabetes",
             "Injury","Musculoskeletal","Genitourinary","Neoplasms"])
diag_3 = st.sidebar.selectbox("Tertiary Diagnosis",
            ["Other","Circulatory","Diabetes","Respiratory","Digestive",
             "Injury","Musculoskeletal","Genitourinary","Neoplasms"])

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Predict Readmission Risk", use_container_width=True)

# ── ENCODING ──────────────────────────────────────────────
gender_map  = {"Male": 1, "Female": 0}
race_map    = {"Caucasian": 4, "African American": 0,
               "Hispanic": 2, "Asian": 1, "Other": 3}
insulin_map = {"No": 1, "Steady": 2, "Up": 3, "Down": 0}
change_map  = {"No": 0, "Ch": 1}
diab_map    = {"No": 0, "Yes": 1}
diag_map    = {"Circulatory": 0, "Respiratory": 6, "Digestive": 1,
               "Diabetes": 2, "Injury": 3, "Musculoskeletal": 5,
               "Genitourinary": 4, "Neoplasms": 7, "Other": 8}

def get_age_bucket(age):
    buckets = [10,20,30,40,50,60,70,80,90,100]
    for i, b in enumerate(buckets):
        if age <= b:
            return i
    return 9

def encode_and_scale():
    total_visits     = number_outpatient + number_emergency + num_prev_admissions
    meds_per_day     = num_medications / max(length_of_stay, 1)
    high_utilizer    = 1 if total_visits > 3 else 0
    num_meds_changed = 1 if insulin in ["Up","Down"] else 0

    # Build raw (unscaled) input matching training feature order
    input_dict = {
        "race":                      race_map.get(race, 3),
        "gender":                    gender_map.get(gender, 1),
        "age":                       get_age_bucket(age),
        "admission_type_id":         float(admission_type_id),
        "discharge_disposition_id":  float(discharge_disposition_id),
        "admission_source_id":       float(admission_source_id),
        "time_in_hospital":          float(length_of_stay),
        "num_lab_procedures":        float(num_lab_procedures),
        "num_procedures":            float(num_procedures),
        "num_medications":           float(num_medications),
        "number_outpatient":         float(number_outpatient),
        "number_emergency":          float(number_emergency),
        "number_inpatient":          float(num_prev_admissions),
        "diag_1":                    diag_map.get(diag_1, 8),
        "diag_2":                    diag_map.get(diag_2, 8),
        "diag_3":                    diag_map.get(diag_3, 8),
        "number_diagnoses":          float(num_diagnoses),
        "metformin":                 1,
        "repaglinide":               1,
        "nateglinide":               1,
        "glimepiride":               1,
        "glipizide":                 1,
        "glyburide":                 1,
        "pioglitazone":              1,
        "rosiglitazone":             1,
        "insulin":                   insulin_map.get(insulin, 1),
        "change":                    change_map.get(change_in_meds, 0),
        "diabetesMed":               diab_map.get(diabetes_meds, 1),
        "total_visits":              float(total_visits),
        "meds_per_day":              float(meds_per_day),
        "high_utilizer":             float(high_utilizer),
        "age_numeric":               float(age),
        "num_meds_changed":          float(num_meds_changed),
    }

    # Build dataframe in exact feature order
    df = pd.DataFrame([{f: input_dict.get(f, 0) for f in feature_names}])

    # Scale only the numeric columns (same as training)
    cols_to_scale = [c for c in NUMERIC_COLS if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df

# ── MAIN PAGE ─────────────────────────────────────────────
st.title("🏥 Hospital 30-Day Readmission Risk Predictor")
st.write("Enter patient data in the sidebar, then click **Predict** to assess risk.")
st.markdown("---")

if not predict_btn:
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "XGBoost")
    col2.metric("ROC-AUC Score", "0.664")
    col3.metric("Features Used", str(len(feature_names)))

    st.info("👈 Fill in patient details in the sidebar and click **Predict Readmission Risk**")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1️⃣ Enter Patient Data")
        st.write("Fill in demographics, admission details, medications and diagnoses.")
    with c2:
        st.subheader("2️⃣ Get Risk Score")
        st.write("XGBoost predicts 30-day readmission probability with a risk tier.")
    with c3:
        st.subheader("3️⃣ Understand Why")
        st.write("SHAP values explain which factors drove the prediction.")

else:
    # Encode and scale input
    input_df = encode_and_scale()
    prob     = model.predict_proba(input_df)[0][1]
    prob_pct = prob * 100

    # Debug line — remove after confirming it works
    st.caption(f"🔧 Debug — Raw probability from model: {prob:.4f}")

    if prob >= 0.60:
        risk_level  = "🔴 HIGH RISK"
        risk_action = "Immediate care coordination. Follow-up within 48–72 hours."
        risk_fn     = st.error
    elif prob >= 0.35:
        risk_level  = "🟡 MEDIUM RISK"
        risk_action = "Follow-up appointment within 7 days recommended."
        risk_fn     = st.warning
    else:
        risk_level  = "🟢 LOW RISK"
        risk_action = "Standard discharge. Routine follow-up within 2–4 weeks."
        risk_fn     = st.success

    col_main, col_side = st.columns([2, 1])

    with col_main:
        risk_fn(f"## {risk_level}")
        st.metric("30-Day Readmission Probability", f"{prob_pct:.1f}%")
        st.progress(float(prob))
        st.info(f"📋 {risk_action}")

    with col_side:
        st.subheader("Patient Summary")
        st.write(f"**Age:** {age} | **Gender:** {gender}")
        st.write(f"**Race:** {race}")
        st.write(f"**Stay:** {length_of_stay} days")
        st.write(f"**Medications:** {num_medications}")
        st.write(f"**Prior admissions:** {num_prev_admissions}")
        st.write(f"**Diagnoses:** {num_diagnoses}")
        st.write(f"**Primary Dx:** {diag_1}")

    st.markdown("---")
    st.subheader("🔍 Why Did the Model Predict This?")
    st.caption("SHAP values show which factors pushed risk up (red) or down (blue).")

    tab1, tab2, tab3 = st.tabs(["📊 Feature Impact","💧 Waterfall Chart","📋 Factor Table"])

    with tab1:
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_df)
            sv = shap_vals[0]
            fv = input_df.iloc[0].values

            impact      = list(zip(feature_names, sv, fv))
            impact_nz   = [(f,s,v) for f,s,v in impact if abs(s) > 0.001]
            impact_sort = sorted(impact_nz, key=lambda x: abs(x[1]), reverse=True)[:15]

            feats  = [x[0] for x in impact_sort][::-1]
            values = [x[1] for x in impact_sort][::-1]
            colors = ["#e53e3e" if v > 0 else "#3182ce" for v in values]

            fig, ax = plt.subplots(figsize=(9,5))
            ax.barh(feats, values, color=colors, height=0.6, edgecolor="none")
            ax.axvline(0, color="#2d3748", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP Value")
            ax.set_title("Top Factors Driving This Prediction", fontweight="bold")
            ax.spines[["top","right","left"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("🔴 Red = pushes risk higher   |   🔵 Blue = pushes risk lower")
        except Exception as e:
            st.warning(f"Chart unavailable: {e}")

    with tab2:
        try:
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(input_df)
            explanation = shap.Explanation(
                values        = shap_vals[0],
                base_values   = explainer.expected_value,
                data          = input_df.iloc[0].values,
                feature_names = feature_names
            )
            fig, ax = plt.subplots(figsize=(10,6))
            shap.waterfall_plot(explanation, max_display=12, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Waterfall unavailable: {e}")

    with tab3:
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_df)
            sv = shap_vals[0]
            fv = input_df.iloc[0].values
            rows = []
            for feat, shap_val, feat_val in sorted(
                    zip(feature_names, sv, fv),
                    key=lambda x: abs(x[1]), reverse=True):
                if abs(shap_val) > 0.001:
                    rows.append({
                        "Feature":       feat,
                        "Patient Value": round(float(feat_val), 3),
                        "SHAP Impact":   round(float(shap_val), 4),
                        "Direction":     "⬆ Increases risk" if shap_val > 0 else "⬇ Decreases risk"
                    })
            st.dataframe(pd.DataFrame(rows).head(20),
                         use_container_width=True, height=400)
        except Exception as e:
            st.warning(f"Table unavailable: {e}")

    st.markdown("---")
    st.subheader("📌 Clinical Recommendations")
    rec1, rec2 = st.columns(2)

    with rec1:
        if prob >= 0.60:
            st.error("🔴 HIGH RISK — Suggested Actions")
            st.markdown("""
            - Follow-up within **48–72 hours**
            - Assign dedicated care coordinator
            - Review medication reconciliation
            - Arrange home health services if needed
            """)
        elif prob >= 0.35:
            st.warning("🟡 MEDIUM RISK — Suggested Actions")
            st.markdown("""
            - Follow-up within **7 days**
            - Provide written discharge instructions
            - Consider telehealth check-in at day 3
            """)
        else:
            st.success("🟢 LOW RISK — Suggested Actions")
            st.markdown("""
            - Standard discharge process
            - Routine follow-up within **2–4 weeks**
            - Patient education on warning signs
            """)

    with rec2:
        st.subheader("Top Risk Drivers")
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_df)
            sv = shap_vals[0]
            fv = input_df.iloc[0].values
            top = sorted(zip(feature_names, sv, fv),
                         key=lambda x: x[1], reverse=True)[:5]
            for feat, shap_val, feat_val in top:
                if shap_val > 0.01:
                    st.write(f"⚠️ **{feat}** = {feat_val:.2f} (+{shap_val:.3f})")
        except:
            st.info("SHAP values not available.")

    st.markdown("---")
    st.caption("⚠️ Disclaimer: For research and educational purposes only. "
               "Not a substitute for clinical judgment.")