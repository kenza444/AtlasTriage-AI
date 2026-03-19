import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration (The Tab Title)
st.set_page_config(page_title="AtlasTriage AI", page_icon="🏥", layout="centered")

# 2. Load the AI Model and Scaler
@st.cache_resource
def load_assets():
    model = joblib.load('cancer_triage_model.pkl')
    scaler = joblib.load('triage_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("⚠️ Error loading model files. Please ensure .pkl files are in the repository.")

# 3. Sidebar for Language Selection
st.sidebar.title("🌍 Language / لغة")
lang = st.sidebar.radio("Select Language:", ("English", "Français", "العربية"))

# 4. Content Dictionary
content = {
    "English": {"title": "AtlasTriage AI", "subtitle": "Breast Cancer Screening Assistant", "button": "Run Assessment", "results": "Results"},
    "Français": {"title": "AtlasTriage AI", "subtitle": "Assistant de dépistage du cancer du sein", "button": "Lancer l'évaluation", "results": "Résultats"},
    "العربية": {"title": "أطلس ترياج - ذكاء اصطناعي", "subtitle": "مساعد فحص سرطان الثدي", "button": "بدء الفحص", "results": "النتائج"}
}

st.title(content[lang]["title"])
st.write(f"### {content[lang]['subtitle']}")

# 5. Input Fields (The 6 Golden Features)
st.write("---")
col1, col2 = st.columns(2)

with col1:
    cp_worst = st.number_input("Concave Points (Worst)", value=0.10, step=0.01)
    peri_worst = st.number_input("Perimeter (Worst)", value=100.0, step=1.0)
    cp_mean = st.number_input("Concave Points (Mean)", value=0.05, step=0.01)

with col2:
    rad_worst = st.number_input("Radius (Worst)", value=15.0, step=0.1)
    peri_mean = st.number_input("Perimeter (Mean)", value=90.0, step=1.0)
    area_worst = st.number_input("Area (Worst)", value=800.0, step=10.0)

# 6. Prediction Logic
if st.button(content[lang]["button"]):
    # Prepare data
    input_data = np.array([[cp_worst, peri_worst, cp_mean, rad_worst, peri_mean, area_worst]])
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] if prediction == 1 else model.predict_proba(input_scaled)[0][0]
    
    status = "HIGH" if prediction == 1 else "LOW"
    
    # 7. Trilingual Result Display
    color = "#d9534f" if status == "HIGH" else "#5cb85c"
    
    st.markdown(f"""
    <div style="padding:20px; border-radius:10px; border-top:10px solid {color}; background-color:#f8f9fa; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
        <h2 style="text-align:center;">{content[lang]['results']}</h2>
        <div style="text-align:center; color:{color}; font-size:24px; font-weight:bold;">
            <p>{"🚨 HIGH RISK - PRIORITY" if status == "HIGH" else "✅ LOW RISK - ROUTINE"}</p>
            <p>{"RISQUE ÉLEVÉ - PRIORITÉ" if status == "HIGH" else "RISQUE FAIBLE - ROUTINE"}</p>
            <p style="direction:rtl;">{"خطر مرتفع - أولوية قصوى" if status == "HIGH" else "خطر منخفض - فحص روتيني"}</p>
        </div>
        <p style="text-align:center; color:#555;">Confidence / اليقين: {prob:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ Disclaimer: This is an AI prototype for triage support only. Consult a specialist / ملاحظة: هذا نظام تجريبي فقط.")
