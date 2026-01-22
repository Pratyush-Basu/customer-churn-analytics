"""Main module."""
import streamlit as st
import pandas as pd
import joblib

from feature_engineering import add_engineered_features

# --------------------
# Load model
# --------------------
model_bundle = joblib.load("../../models/final_churn_model.pkl")
model = model_bundle["model"]
THRESHOLD = model_bundle["threshold"]

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("Customer Churn Prediction Dashboard")



if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = 0


st.sidebar.title(" Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Single Customer Prediction",
        "Batch Prediction",
        "CSV Analysis"
    ]
)



if page == "Home":
    st.header("Welcome to Customer Churn Analytics Dashboard")

    st.markdown("""
    ### 🎯 What this dashboard does
    - Predict customer churn risk for individual customers
    - Perform batch churn scoring for campaigns
    - Analyze churn risk to prioritize retention actions

    ### 🧠 Model Highlights
    - Trained on real-world Telco customer data
    - Business-optimized threshold for recall
    - Experiment tracking with MLflow
    - Explainable predictions using engineered features

    👉 Use the sidebar to get started.
    """)

    st.info("This dashboard is designed for business and analytics teams.")




#------- Tab 1: Single Customer Prediction --------
if page == "Single Customer Prediction":

    st.header("Single Customer Churn Prediction")

    col1, col2, col3 = st.columns(3)

    # -------- Customer Demographics --------
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    # -------- Services --------
    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )
        InternetService = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No"])

    # -------- Billing & Contract --------
    with col3:
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
        Contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        MonthlyCharges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )
        TotalCharges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=800.0
        )

    # -------- Validation --------
    if st.button("Predict Churn"):
        if tenure == 0 and TotalCharges > 0:
            st.error(
                "Total Charges cannot be greater than 0 when tenure is 0."
            )
            st.stop()

        # -------- Create Input DataFrame --------
        input_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])

        # -------- Feature Engineering --------
        input_df = add_engineered_features(input_df)

        # -------- Prediction --------
        churn_prob = model.predict_proba(input_df)[:, 1][0]
        churn_pred = int(churn_prob >= THRESHOLD)

        # -------- Results --------
        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{churn_prob:.2%}")

        if churn_pred == 1:
            st.error("High Risk of Churn")
        else:
            st.success("Low Risk of Churn")

        st.caption(
            "Prediction considers tenure, contract type, service usage, and billing behavior."
        )



#------- Tab 2: Batch Prediction --------
if page == "Batch Prediction":

    st.header("Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        df_fe = add_engineered_features(df)

        probs = model.predict_proba(df_fe)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)

        df["Churn_Probability"] = probs
        df["Churn_Prediction"] = preds
        df["Risk_Level"] = pd.cut(
            probs,
            bins=[0, 0.4, 0.6, 1.0],
            labels=["Low", "Medium", "High"]
        )

        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions",
            csv,
            "churn_predictions.csv",
            "text/csv"
        )
        
        st.session_state["analysis_df"] = df
        
        if st.button("Go to Analysis Dashboard"):
            st.session_state["active_tab"] = 2
            st.rerun()
        
        
 
#------- Tab 3: Analysis Dashboard --------       
if page == "CSV Analysis":

    st.header("Churn Risk Analysis Dashboard")

    df = None

    # --------- DATA SOURCE SELECTION ---------
    if "analysis_df" in st.session_state:
        df = st.session_state["analysis_df"]
        st.info("Using data from Batch Prediction")

    else:
        analysis_file = st.file_uploader(
            "Upload Batch Prediction CSV",
            type=["csv"],
            key="analysis_csv"
        )

        if analysis_file is not None:
            df = pd.read_csv(analysis_file)
        else:
            st.warning("Upload a CSV or run Batch Prediction first.")
            st.stop()

    # --------- ANALYSIS STARTS HERE ---------
    st.subheader("🔹 Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("High Risk Customers", (df["Risk_Level"] == "High").sum())
    col3.metric(
        "High Risk %",
        f"{(df['Risk_Level'] == 'High').mean() * 100:.1f}%"
    )
    col4.metric(
        "Avg Churn Probability",
        f"{df['Churn_Probability'].mean():.2f}"
    )

    st.divider()

    # ---------- RISK DISTRIBUTION ----------
    st.subheader("🔹 Risk Distribution")
    st.bar_chart(df["Risk_Level"].value_counts())

    st.divider()

    # ---------- RISK BY CONTRACT ----------
    st.subheader("🔹 Churn Risk by Contract Type")
    contract_risk = (
        df.groupby("Contract")["Churn_Probability"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(contract_risk)

    st.divider()

    # ---------- RISK BY TENURE ----------
    st.subheader("🔹 Churn Risk by Tenure Group")

    df["Tenure_Group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 100],
        labels=["0–12 months", "12–24 months", "24+ months"]
    )

    tenure_risk = (
        df.groupby("Tenure_Group")["Churn_Probability"]
        .mean()
    )
    st.bar_chart(tenure_risk)

    st.divider()

    # ---------- HIGH RISK EXPORT ----------
    st.subheader("🔹 High-Risk Customer List")

    high_risk_df = df[df["Risk_Level"] == "High"]
    st.dataframe(high_risk_df.head(20))

    csv = high_risk_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download High-Risk Customers",
        csv,
        "high_risk_customers.csv",
        "text/csv"
    )
    

