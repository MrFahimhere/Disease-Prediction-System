import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🩺",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem; color: #58a6ff;
    margin-bottom: 0.2rem; letter-spacing: -0.5px;
}
.hero-sub { color: #8b949e; font-size: 1.05rem; margin-bottom: 2rem; font-weight: 300; }

.accuracy-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: white; padding: 0.45rem 1.1rem;
    border-radius: 20px; font-weight: 600; font-size: 0.95rem; margin-bottom: 1.5rem;
}
.section-label {
    font-size: 0.8rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase; color: #8b949e; margin-bottom: 0.5rem;
}
.result-card {
    background: linear-gradient(135deg, #1c2128 0%, #161b22 100%);
    border: 1px solid #30363d; border-left: 4px solid #58a6ff;
    border-radius: 14px; padding: 1.8rem 2rem; margin-top: 2rem;
}
.result-disease { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #58a6ff; margin-bottom: 0.3rem; }
.result-label { font-size: 0.78rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #8b949e; }

div.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white; border: none; border-radius: 10px;
    padding: 0.7rem 2.2rem; font-size: 1rem; font-weight: 600;
    font-family: 'DM Sans', sans-serif; cursor: pointer; width: 100%;
}
div.stButton > button:hover { opacity: 0.88; }
hr { border-color: #21262d; }
[data-testid="stCheckbox"] label { color: #c9d1d9 !important; font-size: 0.88rem; }
.stTextInput input {
    background: #161b22 !important; color: #e6edf3 !important;
    border: 1px solid #30363d !important; border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load saved model artifacts ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load("model.pkl")
    mlb          = joblib.load("mlb.pkl")
    lb           = joblib.load("lb.pkl")
    accuracy     = joblib.load("accuracy.pkl")
    all_symptoms = joblib.load("symptoms.pkl")
    return model, mlb, lb, accuracy, all_symptoms

try:
    model, mlb, lb, accuracy, all_symptoms = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run `python model.py` first to train and save the model.")
    st.stop()


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🩺 Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Select the symptoms you are experiencing and get an instant AI-based disease prediction.</div>', unsafe_allow_html=True)
st.markdown(f'<div class="accuracy-badge">✦ Model Accuracy &nbsp;{accuracy * 100:.2f}%</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Symptom selector ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Search & select symptoms</div>', unsafe_allow_html=True)
search_query = st.text_input("", placeholder="Type to filter symptoms…", label_visibility="collapsed")

filtered_symptoms = (
    [s for s in all_symptoms if search_query.lower() in s.lower()]
    if search_query else all_symptoms
)

st.markdown(f"<small style='color:#8b949e;'>Showing {len(filtered_symptoms)} of {len(all_symptoms)} symptoms</small>", unsafe_allow_html=True)
st.markdown("")

selected_symptoms = []
cols = st.columns(3)
for i, symptom in enumerate(filtered_symptoms):
    with cols[i % 3]:
        if st.checkbox(symptom.replace("_", " ").title(), key=f"sym_{symptom}"):
            selected_symptoms.append(symptom)

st.markdown("---")

# ── Selected symptoms summary ───────────────────────────────────────────────────
if selected_symptoms:
    tags = " ".join(
        [f'<span style="background:#1f3a5f;color:#79c0ff;border-radius:6px;padding:2px 10px;margin:3px;display:inline-block;font-size:0.83rem;">{s.replace("_"," ").title()}</span>'
         for s in selected_symptoms]
    )
    st.markdown(f"**Selected ({len(selected_symptoms)}):** {tags}", unsafe_allow_html=True)
    st.markdown("")

# ── Predict button ──────────────────────────────────────────────────────────────
col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button("Predict Disease")

# ── Prediction ──────────────────────────────────────────────────────────────────
if predict_clicked:
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom before predicting.")
    else:
        input_vector = np.zeros(len(all_symptoms))
        for sym in selected_symptoms:
            if sym in all_symptoms:
                input_vector[all_symptoms.index(sym)] = 1

        input_df = pd.DataFrame([input_vector], columns=all_symptoms)
        pred_encoded = model.predict(input_df)[0]
        predicted_disease = lb.inverse_transform([pred_encoded])[0]
        confidence = model.predict_proba(input_df)[0][pred_encoded] * 100

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Condition</div>
            <div class="result-disease">{predicted_disease}</div>
            <div style="color:#8b949e;font-size:0.9rem;margin-top:0.6rem;">
                Confidence &nbsp;<strong style="color:#3fb950;">{confidence:.1f}%</strong>
                &nbsp;·&nbsp; Based on {len(selected_symptoms)} symptom(s)
                &nbsp;·&nbsp; Model accuracy&nbsp;<strong style="color:#58a6ff;">{accuracy*100:.2f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("⚕️ This tool is for educational purposes only. Always consult a qualified medical professional.")