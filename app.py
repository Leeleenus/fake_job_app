import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    model = joblib.load("models/fake_job_detector.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("📘 Project Info")
    st.markdown(
        """
        **Mini Project (IT41033)**  
        Intake 11 - Term 1  

        **Topic:** Fake Job Posting Detection  

        **Student:** *Your Name Here*  
        **Course:** System Administration & Maintenance  

        ---
        This app predicts whether a job posting is **real or fake** using 
        **Natural Language Processing (NLP)** and **Machine Learning**.
        """
    )
    st.caption(f"Last updated: {datetime.today().strftime('%d %B %Y')}")

# ========== MAIN TITLE ==========
st.markdown("<h1 style='text-align: center;'>🕵️ Fake Job Posting Detector</h1>", unsafe_allow_html=True)
st.write("Paste a job posting below and the system will analyze it for potential fraud.")

# ========== INPUT FORM ==========
with st.form("job_form"):
    st.subheader("Enter Job Posting Details")
    job_title = st.text_input("Job Title", placeholder="e.g. Software Engineer")
    company_profile = st.text_area("Company Profile", placeholder="Describe the company...")
    description = st.text_area("Job Description", placeholder="Enter full job description...")
    requirements = st.text_area("Requirements", placeholder="Enter job requirements...")

    submitted = st.form_submit_button("🔍 Analyze Job Posting")

# ========== PREDICTION ==========
if submitted:
    # Combine text
    text = job_title + " " + company_profile + " " + description + " " + requirements
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    # Confidence score (if supported by model)
    try:
        prob = model.predict_proba(X)[0][1]  # probability of being fake
    except:
        prob = None

    st.subheader("📊 Prediction Result")
    if pred == 1:
        st.error("🚨 This looks like a **FAKE** job posting!")
        if prob is not None:
            st.write(f"**Confidence (Fake):** {prob*100:.2f}%")
    else:
        st.success("✅ This looks like a **REAL** job posting.")
        if prob is not None:
            st.write(f"**Confidence (Real):** {(1-prob)*100:.2f}%")

# ========== FOOTER ==========
st.markdown(
    """
    ---
    ⚠️ **Disclaimer:**  
    This tool is for educational purposes only.  
    Always verify job postings manually before applying.  
    """
)
