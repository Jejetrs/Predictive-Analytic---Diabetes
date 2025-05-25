import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Load dataset dan model machine learning
df = pd.read_csv("app/cleaned_dataset/Cleaned_Healthcare_Diabetes.csv")
knn_model = joblib.load('app/model/best_knn_model.pkl')
rf_model = joblib.load('app/model/best_rf_model.pkl')
logreg_model = joblib.load('app/model/best_lr_model.pkl') 
scaler = joblib.load('app/model/scaler.pkl')

# --- CUSTOM CSS ---
st.markdown("""
<style>
/* Header styling */
.header-container {
    background: linear-gradient(90deg, #0d47a1, #1976d2, #42a5f5);
    padding: 40px 0;
    border-radius: 12px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.35);
    margin-bottom: 40px;
    text-align: center;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header-title {
    font-size: 3.2rem;
    font-weight: 900;
    color: white;
    margin: 0;
    text-shadow: 1.5px 1.5px 5px rgba(0,0,0,0.6);
}
.header-subtitle {
    font-size: 1.4rem;
    color: #e0e0e0;
    margin: 8px 0 0 0;
    font-style: italic;
}

/* Sidebar image style */
.sidebar .sidebar-content img {
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}

/* Sidebar text style */
.sidebar .sidebar-content h3 {
    font-weight: 700;
    color: #0d47a1;
    margin-bottom: 8px;
}

.sidebar .sidebar-content p, .sidebar .sidebar-content li {
    font-size: 0.95rem;
    line-height: 1.4;
}

.sidebar .sidebar-content hr {
    margin: 30px 0;
    border-top: 1.5px solid #1976d2;
}

/* Form input cards */
.input-card {
    background-color: #f7f9fc;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(33, 150, 243, 0.1);
    margin-bottom: 30px;
}

/* Prediction result boxes */
.result-positive {
    background-color: #fde2e2;
    padding: 20px;
    border-radius: 12px;
    color: #b02a37;
    font-weight: 700;
    box-shadow: 0 3px 12px rgba(176, 42, 55, 0.4);
    margin-top: 20px;
}

.result-negative {
    background-color: #d1e7dd;
    padding: 20px;
    border-radius: 12px;
    color: #0f5132;
    font-weight: 700;
    box-shadow: 0 3px 12px rgba(15, 81, 50, 0.4);
    margin-top: 20px;
}

.stButton>button {
    background-color: #1976d2;
    color: white;
    font-weight: 600;
    padding: 12px 24px;
    border-radius: 8px;
    transition: background-color 0.3s ease;
}

.stButton>button:hover {
    background-color: #0d47a1;
    color: white;
}

.info {
    font-style: italic;
    color: #555;
    margin-top: -10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="header-container">
    <h1 class="header-title">💉 Diabetes Predictive Analytic</h1>
    <p class="header-subtitle">Explore diabetes data and predict risk based on your health measurements.</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.image("app/diabetes-ribbonblue.jpg", use_column_width=True, caption="Stay Healthy!")

st.sidebar.markdown("""
### About This App
This app helps you explore the diabetes dataset and predict diabetes risk based on your health data.

- Dataset statistics and visualizations  
- Predict diabetes using KNN, Random Forest, or Logistic Regression  
- Built with Streamlit and Scikit-learn
""")

st.sidebar.markdown("---")
st.sidebar.header("Health Tips")
st.sidebar.write("""
- Maintain a balanced diet 🥗  
- Exercise regularly 🏃‍♂️  
- Monitor glucose levels  
- Visit your doctor for routine checkups  
""")

# TABS
tab1, tab2 = st.tabs(["📊 Dataset", "🧪 Prediction"])

# --- TAB 1: Dataset Stats & Visualization ---
with tab1:
    st.markdown("<h2 style='color:#0d47a1;'>Explore Dataset & Visualizations</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe().style.format("{:.2f}").background_gradient(cmap="Blues"))
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Class Distribution")
    outcome_counts = df['Outcome'].value_counts().sort_index()
    outcome_df = outcome_counts.reset_index()
    outcome_df.columns = ['Outcome', 'Count']
    outcome_df['Outcome'] = outcome_df['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})

    fig = go.Figure(data=[
        go.Bar(
            x=outcome_df['Outcome'],
            y=outcome_df['Count'],
            marker_color=['#0d47a1', '#b02a37']
        )
    ])

    fig.update_layout(
        yaxis_title='Count',
        xaxis_title='Outcome',
        xaxis_tickangle=0,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Total samples: {len(df)}")

    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("3D Scatter Plot Visualization")

    
    st.markdown("""
    <p class="info">
    Select variables for X, Y, and Z axes to explore 3D relationships and their impact on diabetes diagnosis.
    </p>
    """, unsafe_allow_html=True)

    variable_options = ["-- Select --"] + list(df.columns[:-1])
    x_3d = st.selectbox("X Variable", variable_options, index=0, key="x3d")
    y_3d = st.selectbox("Y Variable", variable_options, index=0, key="y3d")
    z_3d = st.selectbox("Z Variable", variable_options, index=0, key="z3d")

    if (x_3d != "-- Select --" and y_3d != "-- Select --" and z_3d != "-- Select --"):
        fig_3d = px.scatter_3d(
            df,
            x=x_3d,
            y=y_3d,
            z=z_3d,
            color=df['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'}),
            title=f"3D Scatter Plot: {x_3d} vs {y_3d} vs {z_3d} by Outcome",
            opacity=0.7,
            labels={"color": "Outcome"},
            color_discrete_map={'No Diabetes': '#0d47a1', 'Diabetes': '#b02a37'}
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("Please select X, Y, and Z variables for the 3D plot above.")

# --- TAB 2: Diabetes Prediction ---
with tab2:
    st.markdown("<h2 style='color:#6a1b9a;'>Input Your Health Data</h2>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top:-15px; margin-bottom:20px;'>Fill the form below and select a model to predict diabetes risk.</p>", unsafe_allow_html=True)

    with st.container():
        with st.form("input_form"):
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
                glucose = st.number_input("Glucose", min_value=0, max_value=300, value=0, step=1)
                blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0, step=1)
                skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0, step=1)

            with col2:
                insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=0, step=1)
                bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=0.0, format="%.2f")
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0, format="%.3f")
                age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            model_option = st.selectbox("Choose Prediction Model", ("KNN - recommended", "Random Forest", "Logistic Regression"))

            submitted = st.form_submit_button("Predict")

    if submitted:
    # Validasi input minimal masuk akal
        errors = []
        if glucose <= 0:
            errors.append("Glucose must be greater than 0.")
        if blood_pressure <= 0:
            errors.append("Blood Pressure must be greater than 0.")
        if bmi <= 0:
            errors.append("BMI must be greater than 0.")
        if age <= 0:
            errors.append("Age must be greater than 0.")
        # Bisa tambah validasi lain jika perlu

        if errors:
            for err in errors:
                st.error(err)
        else:
            # Prepare input for model
            input_features = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
            input_scaled = scaler.transform(input_features)

            # Pilih model sesuai opsi
            if model_option == "KNN":
                model = knn_model
            elif model_option == "Random Forest":
                model = rf_model
            else:
                model = logreg_model

            # Prediksi dan probabilitas
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]  # Probabilitas diabetes (label=1)

            # Tampilkan hasil
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-positive" style="color:#7b1a1a;">  <!-- merah tua -->
                        <h4 style="color:#7b1a1a">🔴 High risk of diabetes.</h4>
                        <p>Probability: <b>{probability*100:.2f}%</b> chance of having diabetes.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-negative" style="color:#145214;">  <!-- hijau tua -->
                        <h4 style="color:#145214">🟢 Low risk of diabetes. Keep living healthy!</h4>
                        <p>Probability: <b>{(1 - probability)*100:.2f}%</b> chance of not having diabetes.</p>
                    </div>
                """, unsafe_allow_html=True)
