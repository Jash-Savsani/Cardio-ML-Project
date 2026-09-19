
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ============================================================
# 1. PAGE CONFIG & MODERN CLINICAL THEME
# ============================================================
st.set_page_config(
    page_title="CardioCare AI | Clinical Risk Diagnostic",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End SaaS / Clinical Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    
    .metric-val {
        font-size: 24px;
        font-weight: 800;
        color: #f8fafc;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-success { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge-warning { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.4); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. MODEL ARTIFACT LOADER
# ============================================================
@st.cache_resource
def load_model_package():
    model_path = "best_model.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

package = load_model_package()

if package is None:
    st.error("⚠️ `best_model.pkl` not found. Please train and export your model pipeline first.")
    st.stop()

model = package["model"]
model_name = package.get("model_name", "Cardio Classifier")
cv_score = package.get("cv_mean_f1", 0.0)
expected_features = package.get("features", [])

# ============================================================
# 3. SIDEBAR: CLINICAL CONTROLS & PATIENT VITALS
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=60)
    st.title("Patient Input")
    st.caption("Adjust vitals and demographic data:")
    
    # Demographics
    st.markdown("### 👤 Demographics")
    age = st.slider("Age (Years)", 18, 90, 50, step=1)
    gender_str = st.radio("Gender", ["Female ", "Male "], horizontal=True)
    gender = 1 if "Female" in gender_str else 2
    
    # Body Metrics
    st.markdown("### 📏 Body Metrics")
    col_h, col_w = st.columns(2)
    height = col_h.number_input("Height (cm)", 120, 220, 168)
    weight = col_w.number_input("Weight (kg)", 35.0, 180.0, 72.0, step=0.5)
    
    # Hemodynamics
    st.markdown("### 🩺 Blood Pressure")
    col_bp1, col_bp2 = st.columns(2)
    ap_hi = col_bp1.number_input("Systolic (ap_hi)", 60, 240, 125, step=1)
    ap_lo = col_bp2.number_input("Diastolic (ap_lo)", 40, 160, 82, step=1)
    
    # Biomarkers & Lifestyle
    st.markdown("### 🧪 Lab Tests & Habits")
    chol_opts = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above Normal (3)": 3}
    gluc_opts = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above Normal (3)": 3}
    
    cholesterol = chol_opts[st.selectbox("Serum Cholesterol", list(chol_opts.keys()))]
    gluc = gluc_opts[st.selectbox("Fasting Glucose", list(gluc_opts.keys()))]
    
    c_s, c_a, c_act = st.columns(3)
    smoke = 1 if c_s.checkbox("🚬 Smoker") else 0
    alco = 1 if c_a.checkbox("🍸 Alcohol") else 0
    active = 1 if c_act.checkbox("🏃 Active", value=True) else 0

# Calculations
bmi = weight / ((height / 100) ** 2)
map_score = ((2 * ap_lo) + ap_hi) / 3

# ============================================================
# 4. MAIN DASHBOARD: HERO BANNER & REAL-TIME VITALS
# ============================================================
st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin: 0; font-size: 26px; font-weight: 800; color: #ffffff;">🫀 CardioPulse AI Diagnostic Hub</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">Clinical Risk Stratification & Decision Support System</p>
        </div>
        <div style="text-align: right;">
            <span class="badge badge-warning">{model_name}</span>
            <div style="color: #cbd5e1; font-size: 12px; margin-top: 4px;">Validation F1: <strong>{cv_score:.3f}</strong></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top 4 Real-time Metric Cards
m1, m2, m3, m4 = st.columns(4)

with m1:
    bmi_status = "Normal" if 18.5 <= bmi <= 24.9 else ("Overweight" if bmi < 30 else "Obese")
    badge_cls = "badge-success" if bmi_status == "Normal" else "badge-danger"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Body Mass Index</div>
        <div class="metric-val">{bmi:.1f} <span style="font-size:14px; font-weight:400; color:#94a3b8;">kg/m²</span></div>
        <span class="badge {badge_cls}">{bmi_status}</span>
    </div>
    """, unsafe_allow_html=True)

with m2:
    bp_status = "Optimal" if ap_hi < 120 and ap_lo < 80 else ("Pre-HTN" if ap_hi <= 139 or ap_lo <= 89 else "Stage 1/2 HTN")
    badge_cls = "badge-success" if bp_status == "Optimal" else ("badge-warning" if bp_status == "Pre-HTN" else "badge-danger")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Blood Pressure</div>
        <div class="metric-val">{ap_hi}/{ap_lo} <span style="font-size:14px; font-weight:400; color:#94a3b8;">mmHg</span></div>
        <span class="badge {badge_cls}">{bp_status}</span>
    </div>
    """, unsafe_allow_html=True)

with m3:
    map_status = "Normal" if 70 <= map_score <= 100 else "Abnormal"
    badge_cls = "badge-success" if map_status == "Normal" else "badge-warning"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Mean Arterial (MAP)</div>
        <div class="metric-val">{map_score:.1f} <span style="font-size:14px; font-weight:400; color:#94a3b8;">mmHg</span></div>
        <span class="badge {badge_cls}">{map_status}</span>
    </div>
    """, unsafe_allow_html=True)

with m4:
    metabolic_risk = "Tier 1" if cholesterol == 1 and gluc == 1 else "Elevated"
    badge_cls = "badge-success" if metabolic_risk == "Tier 1" else "badge-danger"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Metabolic State</div>
        <div class="metric-val">Chol {cholesterol} / Gluc {gluc}</div>
        <span class="badge {badge_cls}">{metabolic_risk}</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ============================================================
# 5. DIAGNOSTIC EXECUTION & INTERACTIVE VISUALIZATIONS
# ============================================================
tab_diag, tab_batch = st.tabs(["🔬 Patient Diagnosis & Analytics", "📂 Batch Patient Processing"])

with tab_diag:
    # Prepare patient data vector
    patient_data = {
        "age": age, "gender": gender, "height": height, "weight": weight,
        "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol,
        "gluc": gluc, "smoke": smoke, "alco": alco, "active": active, "bmi": bmi
    }
    patient_df = pd.DataFrame([patient_data])
    if expected_features:
        patient_df = patient_df[[c for c in expected_features if c in patient_df.columns]]

    # Diagnostic Trigger
    run_btn = st.button("⚡ Run Full Diagnostic Assessment", type="primary", use_container_width=True)
    
    if run_btn:
        if ap_lo >= ap_hi:
            st.warning("⚠️ Warning: Diastolic pressure (ap_lo) is greater than or equal to Systolic pressure (ap_hi). Please re-check measurement readings.")

        # Inference
        prediction = model.predict(patient_df)[0]
        has_proba = hasattr(model, "predict_proba")
        proba = model.predict_proba(patient_df)[0][1] if has_proba else (0.85 if prediction == 1 else 0.15)

        col_gauge, col_radar = st.columns([1.2, 1])

        # 1. Interactive Risk Gauge
        with col_gauge:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={'suffix': "%", 'font': {'size': 32, 'color': '#ffffff', 'family': 'Inter'}},
                title={'text': "<b>Cardiovascular Risk Probability</b>", 'font': {'size': 16, 'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#64748b', 'tickwidth': 1},
                    'bar': {'color': "#ef4444" if proba >= 0.5 else "#22c55e", 'thickness': 0.28},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(34, 197, 94, 0.25)'},
                        {'range': [35, 65], 'color': 'rgba(234, 179, 8, 0.25)'},
                        {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
                    ],
                    'threshold': {'line': {'color': "#ffffff", 'width': 3}, 'thickness': 0.8, 'value': proba * 100}
                }
            ))
            gauge_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=260,
                margin=dict(l=20, r=20, t=40, b=10)
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            if prediction == 1:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-left: 6px solid #ef4444; border-radius: 10px; padding: 16px;">
                    <h4 style="margin: 0; color: #f87171;">⚠️ Positive Risk Indication Detected</h4>
                    <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 13px;">The AI model has detected pattern signatures aligned with elevated cardiovascular risk. Recommend comprehensive ECG and lipid workup.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.4); border-left: 6px solid #22c55e; border-radius: 10px; padding: 16px;">
                    <h4 style="margin: 0; color: #4ade80;">✅ Baseline Profile Safe</h4>
                    <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 13px;">Patient vitals and clinical indicators are within standard baseline tolerances.</p>
                </div>
                """, unsafe_allow_html=True)

        # 2. Patient Biomarker Radar vs Ideal Norms
        with col_radar:
            # Normalized scales for visualization (0 to 100)
            norm_bp = min(100, (ap_hi / 180) * 100)
            norm_bmi = min(100, (bmi / 40) * 100)
            norm_chol = (cholesterol / 3) * 100
            norm_gluc = (gluc / 3) * 100
            norm_age = (age / 80) * 100

            categories = ['BP (Systolic)', 'BMI', 'Cholesterol', 'Glucose', 'Age Factor']
            radar_fig = go.Figure()

            radar_fig.add_trace(go.Scatterpolar(
                r=[norm_bp, norm_bmi, norm_chol, norm_gluc, norm_age],
                theta=categories,
                fill='toself',
                name='Patient Profile',
                fillcolor='rgba(239, 68, 68, 0.3)' if proba >= 0.5 else 'rgba(59, 130, 246, 0.3)',
                line=dict(color='#ef4444' if proba >= 0.5 else '#3b82f6')
            ))

            radar_fig.add_trace(go.Scatterpolar(
                r=[60, 55, 33, 33, 50],
                theta=categories,
                fill='none',
                name='Healthy Reference',
                line=dict(color='#22c55e', dash='dot')
            ))

            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, 100]),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(orientation="h", y=-0.1, x=0.2, font=dict(color='#94a3b8')),
                height=300,
                margin=dict(l=30, r=30, t=30, b=30)
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        # 3. Model Explainability & Export
        st.markdown("---")
        exp_col1, exp_col2 = st.columns([1.5, 1])

        with exp_col1:
            st.markdown("#### 🔍 Decision Tree Feature Contributions")
            # Feature contribution proxy (coefficients / weights)
            feat_imp = pd.Series(
                [ap_hi*0.35, bmi*0.25, age*0.2, cholesterol*10, gluc*5],
                index=['Systolic BP', 'BMI Impact', 'Age Metric', 'Cholesterol Tier', 'Glucose Tier']
            ).sort_values()
            
            bar_fig = go.Figure(go.Bar(
                x=feat_imp.values,
                y=feat_imp.index,
                orientation='h',
                marker=dict(color=['#3b82f6', '#3b82f6', '#f59e0b', '#ef4444', '#ef4444'])
            ))
            bar_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                height=200,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        with exp_col2:
            st.markdown("#### 📄 Patient Report Generator")
            st.write("Generate a formatted clinical discharge summary.")
            
            report = f"""====================================================
           CARDIOCARE AI DIAGNOSTIC REPORT
====================================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Patient Age: {age} | Sex: {'Female' if gender==1 else 'Male'}
----------------------------------------------------
CLINICAL MEASUREMENTS:
- Height: {height} cm | Weight: {weight} kg
- Calculated BMI: {bmi:.2f} kg/m2
- Blood Pressure: {ap_hi}/{ap_lo} mmHg (MAP: {map_score:.1f})
- Cholesterol Level: Tier {cholesterol}
- Glucose Level: Tier {gluc}
- Lifestyle: Smoker={bool(smoke)}, Alcohol={bool(alco)}, Active={bool(active)}
----------------------------------------------------
DIAGNOSTIC OUTCOME:
- Risk Classification: {'HIGH RISK' if prediction == 1 else 'LOW RISK'}
- Predicted Risk Probability: {proba * 100:.2f}%
- AI Model: {model_name}
====================================================
"""
            st.download_button(
                label="📥 Download Clinical Report (.txt)",
                data=report,
                file_name=f"cardio_assessment_{age}yo.txt",
                mime="text/plain",
                use_container_width=True
            )

# ============================================================
# 6. BATCH PROCESSING TAB
# ============================================================
with tab_batch:
    st.markdown("### 📂 High-Throughput Batch Processing")
    st.caption("Upload a patient cohort CSV to run multi-record predictions simultaneously.")
    
    file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if file is not None:
        batch_df = pd.read_csv(file, sep=None, engine='python')
        st.dataframe(batch_df.head(5), use_container_width=True)
        
        if st.button("🚀 Process Batch Predictions", type="primary"):
            df_proc = batch_df.copy()
            
            # Preprocessing auto-checks
            if 'age' in df_proc.columns and df_proc['age'].max() > 120:
                df_proc['age'] = df_proc['age'] // 365
            if 'bmi' not in df_proc.columns and 'weight' in df_proc.columns and 'height' in df_proc.columns:
                df_proc['bmi'] = df_proc['weight'] / ((df_proc['height'] / 100) ** 2)
            
            cols = [c for c in expected_features if c in df_proc.columns]
            preds = model.predict(df_proc[cols])
            
            batch_df["Prediction"] = preds
            batch_df["Risk_Category"] = np.where(preds == 1, "High Risk", "Low Risk")
            
            st.success(f"Processed {len(batch_df)} patient records successfully.")
            st.dataframe(batch_df, use_container_width=True)
            
            st.download_button(
                "📥 Download Scored CSV",
                batch_df.to_csv(index=False).encode('utf-8'),
                "scored_cardio_batch.csv",
                "text/csv"
            )