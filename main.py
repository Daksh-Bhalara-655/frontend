import streamlit as st
import pickle
import numpy as np
from reportlab.lib.pagesizes import A4
import plotly.express as px
import pandas as pd

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------------------------
# CARDIOLOGIST DATABASE
# -------------------------------------------------
top_cardiologists = [
    {"name": "Dr. Naresh Trehan", "loc": "Gurgaon", "hosp": "Medanta - The Medicity", "exp": "50+ Yrs", "qual": "MBBS, Diplomate American Board"},
    {"name": "Dr. Devi Prasad Shetty", "loc": "Bangalore", "hosp": "Narayana Health", "exp": "36+ Yrs", "qual": "MBBS, MS, FRCS"},
    {"name": "Dr. Ashok Seth", "loc": "Delhi", "hosp": "Fortis Escorts Heart Institute", "exp": "40+ Yrs", "qual": "MBBS, MD, FRCP"},
    {"name": "Dr. Ramakanta Panda", "loc": "Mumbai", "hosp": "Asian Heart Institute", "exp": "35+ Yrs", "qual": "MBBS, MCh"},
    {"name": "Dr. Balbir Singh", "loc": "Delhi", "hosp": "Max Super Specialty", "exp": "32+ Yrs", "qual": "MBBS, MD"},
]

# -------------------------------------------------
# GLOBAL UI STYLES
# -------------------------------------------------
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top, #020617, #000000 75%); color: #e5e7eb; }
[data-testid="stSidebar"] { display: none; }
.glass { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
border-radius: 22px; padding: 28px; margin-bottom: 20px; }
.neon { text-align: center; font-size: 48px; font-weight: 800;
background: linear-gradient(to right, #38bdf8, #818cf8);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.doc-card { background: rgba(15,23,42,0.9); border-radius: 18px; padding: 20px;
margin-bottom: 12px; border: 1px solid #1e293b; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------
def extract_years(exp):
    return int(exp.split("+")[0].split()[0])

def suggest_doctors(city, risk, age, doctors):
    matches = [d for d in doctors if d["loc"] == city]
    if not matches:
        matches = doctors

    if risk == "High" or age >= 60:
        matches.sort(key=lambda d: extract_years(d["exp"]), reverse=True)
    else:
        matches.sort(key=lambda d: extract_years(d["exp"]))

    return matches[:5]

def get_medicine_plan(prob):
    if prob < 40:
        return "Low", [
            "Lifestyle modification (diet, exercise, stress management)",
            "Regular health monitoring"
        ]
    elif 40 <= prob <= 70:
        return "Moderate", [
            "Statins: Atorvastatin, Rosuvastatin",
            "ACE Inhibitors / ARBs: Lisinopril, Losartan",
            "Calcium Channel Blockers: Amlodipine",
            "Beta Blockers: Metoprolol",
            "Antiplatelets: Aspirin / Clopidogrel"
        ]
    else:
        return "High", [
            "High-intensity Statins",
            "Antiplatelets: Aspirin + Clopidogrel",
            "GLP-1 / GIP Agonists: Semaglutide, Tirzepatide",
            "SGLT2 Inhibitors",
            "DOACs: Apixaban, Rivaroxaban",
            "ARNI: Sacubitril / Valsartan",
            "Loop Diuretics (Acute): Furosemide",
            "Vasopressors (ICU): Norepinephrine"
        ]
    



# -------------------------------------------------
# UI TABS
# -------------------------------------------------
t1, t2 = st.tabs(["🏠 ML Workflow", "🧪 Diagnostic Engine"])

# with t1:
#     st.markdown("<h1 class='neon'>CardioPredict AI</h1>", unsafe_allow_html=True)
#     steps = ["Problem Definition", "Data Collection", "Preprocessing",
#              "Model Training", "Evaluation", "Deployment"]
#     for i, s in enumerate(steps, 1):
#         st.markdown(f"<div class='glass'><b>STEP {i}</b><br>{s}</div>", unsafe_allow_html=True)

import plotly.express as px
import pandas as pd

with t1:
    st.markdown("<h1 class='neon'>CardioPredict AI</h1>", unsafe_allow_html=True)
    
    # --- 1. Workflow Steps ---
    steps = ["Problem Definition", "Data Collection", "Preprocessing",
             "Model Training", "Evaluation", "Deployment"]
    
    col_steps = st.columns(len(steps))
    for i, s in enumerate(steps):
        with col_steps[i]:
            st.markdown(f"<div class='glass' style='padding:10px; font-size:12px; text-align:center;'><b>STEP {i+1}</b><br>{s}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Dataset Exploratory Analysis")

    try:
        # Load the data
        df = pd.read_csv("cardio_train.csv" , sep=';')
        
        # FIX: Clean column names (removes spaces and converts to lowercase)
        df.columns = df.columns.str.strip().str.lower()

        # Check if required columns exist after cleaning
        required_cols = ['age', 'cholesterol', 'weight', 'smoke', 'alco', 'active']
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            st.error(f"Missing columns in CSV: {missing_cols}")
            st.info(f"Available columns: {list(df.columns)}")
        else:
            g1, g2 = st.columns(2)

            with g1:
                st.markdown("**Age Distribution in Dataset**")
                # Handle age in days vs years
                if df['age'].max() > 1000:
                    df['display_age'] = (df['age'] / 365.25).round()
                else:
                    df['display_age'] = df['age']
                    
                fig_age = px.histogram(df, x="display_age", nbins=30, color_discrete_sequence=['#38bdf8'])
                fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e5e7eb", height=350)
                st.plotly_chart(fig_age, use_container_width=True)

            with g2:
                # Lifestyle Factors (Sum of 1s in binary columns)
                st.markdown("**Lifestyle Factors Presence**")
                life_data = pd.DataFrame({
                    'Factor': ['Smoking', 'Alcohol', 'Active'],
                    'Count': [df['smoke'].sum(), df['alco'].sum(), df['active'].sum()]
                })
                fig_life = px.bar(life_data, x='Factor', y='Count', color='Factor',
                                  color_discrete_map={'Smoking':'#ef4444', 'Alcohol':'#f59e0b', 'Active':'#10b981'})
                fig_life.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e5e7eb", height=350)
                st.plotly_chart(fig_life, use_container_width=True)

            # Weight Distribution
            st.markdown("**Weight Distribution by Cholesterol Levels**")
            fig_weight = px.box(df, x="cholesterol", y="weight", color="cholesterol",
                                color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_weight.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e5e7eb", height=350)
            st.plotly_chart(fig_weight, use_container_width=True)

    except FileNotFoundError:
        st.warning("File 'cardio_data.csv' not found in the project directory.")
    except Exception as e:
        st.error(f"Error loading charts: {e}")
        
        with g3:
            # Lifestyle Bar Chart (Smoke, Alco, Active)
            st.markdown("**Lifestyle Factors Presence**")
            life_data = pd.DataFrame({
                'Factor': ['Smoking', 'Alcohol', 'Active'],
                'Count': [df['smoke'].sum(), df['alco'].sum(), df['active'].sum()]
            })
            fig_life = px.bar(life_data, x='Factor', y='Count', color='Factor',
                              color_discrete_map={'Smoking':'#ef4444', 'Alcohol':'#f59e0b', 'Active':'#10b981'})
            fig_life.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e5e7eb", height=350)
            st.plotly_chart(fig_life, use_container_width=True)

        with g4:
            # Weight Distribution by Cholesterol
            st.markdown("**Weight Distribution by Cholesterol Levels**")
            fig_weight = px.box(df, x="cholesterol", y="weight", color="cholesterol",
                                color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_weight.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e5e7eb", height=350)
            st.plotly_chart(fig_weight, use_container_width=True)

    except FileNotFoundError:
        st.warning("Please upload 'cardio_data.csv' to the root directory to see actual charts.")
        st.info("Showing visual placeholders...")

with t2:
    st.markdown("<h1 class='neon'>Diagnostic Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    try:
        model = pickle.load(open("model_2.pkl", "rb"))

        with st.form("diagnosis_form"):
            name = st.text_input("Patient Full Name")
            city = st.selectbox("Patient City", sorted({d["loc"] for d in top_cardiologists}))

            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", 1, 110, 45)
                gender = st.selectbox("Gender", ["Male", "Female"])
                # --- NEW FIELDS ADDED HERE ---
                height = st.number_input("Height (cm)", 50, 250, 170)
                weight = st.number_input("Weight (kg)", 10, 300, 70)
                # -----------------------------
                bp = st.number_input("Blood Pressure", 80, 240, 120)
                chol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "High"])

            with c2:
                gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "High"])
                smoke = st.radio("Smoking", ["No", "Yes"], horizontal=True)
                alc = st.radio("Alcohol", ["No", "Yes"], horizontal=True)
                act = st.radio("Physical Activity", ["No", "Yes"], horizontal=True)

            submit = st.form_submit_button("GENERATE REPORT")

        if submit and name:
            m = {"Normal": 1, "Above Normal": 2, "High": 3}
            X = np.array([[age, 1 if gender=="Male" else 0, bp, m[chol], m[gluc],
                           1 if smoke=="Yes" else 0, 1 if alc=="Yes" else 0, 1 if act=="Yes" else 0]])

            if X.shape[1] < model.n_features_in_:
                X = np.pad(X, ((0,0),(0,model.n_features_in_-X.shape[1])))

            prob = model.predict_proba(X)[0][1] * 100
            risk, meds = get_medicine_plan(prob)
            docs = suggest_doctors(city, risk, age, top_cardiologists)

            st.markdown(f"### 🧠 Risk Probability: **{prob:.2f}%**")
            st.markdown(f"### 📊 Risk Category: **{risk} Risk**")

            st.markdown("### 💊 Medicine Recommendations")
            for m in meds:
                st.write("•", m)

            st.markdown("### 👨‍⚕️ Recommended Cardiologists")
            for d in docs:
                st.markdown(
                    f"<div class='doc-card'><b>{d['name']}</b><br>🏥 {d['hosp']} ({d['loc']})<br>🎓 {d['qual']}<br>⏳ {d['exp']}</div>",
                    unsafe_allow_html=True
                )

            pdata = {
                "Name": name, "Age": age, "Gender": gender, "City": city,
                "BP": bp, "Cholesterol": chol, "Glucose": gluc
            }

            

    except FileNotFoundError:
        st.error("model.pkl not found.")

    st.markdown("</div>", unsafe_allow_html=True)
