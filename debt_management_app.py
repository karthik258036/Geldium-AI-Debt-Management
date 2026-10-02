import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Geldium Debt Management Assistant",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("geldium_logistic_regression_model.pkl")


model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("💳 Geldium AI Debt Management Assistant")
st.write(
    "AI-assisted customer delinquency risk assessment and "
    "support recommendation."
)

st.divider()

# -----------------------------
# Customer Information
# -----------------------------
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

    income = st.number_input(
        "Income",
        min_value=0.0,
        value=100000.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=600.0
    )

    credit_utilization = st.number_input(
        "Credit Utilization",
        min_value=0.0,
        max_value=2.0,
        value=0.50
    )

with col2:
    missed_payments = st.number_input(
        "Missed Payments",
        min_value=0,
        max_value=20,
        value=2
    )

    loan_balance = st.number_input(
        "Loan Balance",
        min_value=0.0,
        value=50000.0
    )

    dti = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        max_value=2.0,
        value=0.30
    )

    account_tenure = st.number_input(
        "Account Tenure (Years)",
        min_value=0,
        max_value=50,
        value=10
    )

# -----------------------------
# Categorical Information
# -----------------------------
st.subheader("Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    employment_status = st.selectbox(
        "Employment Status",
        [
            "Employed",
            "Unemployed",
            "Retired",
            "Self-employed"
        ]
    )

with col2:
    credit_card_type = st.selectbox(
        "Credit Card Type",
        [
            "Standard",
            "Gold",
            "Platinum"
        ]
    )

with col3:
    location = st.selectbox(
        "Location",
        [
            "Houston",
            "Phoenix",
            "Los Angeles",
            "New York"
        ]
    )

# -----------------------------
# Payment History
# -----------------------------
st.subheader("Recent Payment History")

col1, col2, col3 = st.columns(3)

with col1:
    month_1 = st.selectbox(
        "Month 1",
        ["On-time", "Late", "Missed"]
    )

    month_2 = st.selectbox(
        "Month 2",
        ["On-time", "Late", "Missed"]
    )

with col2:
    month_3 = st.selectbox(
        "Month 3",
        ["On-time", "Late", "Missed"]
    )

    month_4 = st.selectbox(
        "Month 4",
        ["On-time", "Late", "Missed"]
    )

with col3:
    month_5 = st.selectbox(
        "Month 5",
        ["On-time", "Late", "Missed"]
    )

    month_6 = st.selectbox(
        "Month 6",
        ["On-time", "Late", "Missed"]
    )

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🔍 Assess Delinquency Risk", type="primary"):

    customer_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Credit_Score": credit_score,
        "Credit_Utilization": credit_utilization,
        "Missed_Payments": missed_payments,
        "Loan_Balance": loan_balance,
        "Debt_to_Income_Ratio": dti,
        "Employment_Status": employment_status,
        "Account_Tenure": account_tenure,
        "Credit_Card_Type": credit_card_type,
        "Location": location,
        "Month_1": month_1,
        "Month_2": month_2,
        "Month_3": month_3,
        "Month_4": month_4,
        "Month_5": month_5,
        "Month_6": month_6
    }])

    prediction = model.predict(customer_data)[0]
    probability = model.predict_proba(customer_data)[0][1]

    # -----------------------------
    # Results
    # -----------------------------
    st.header("Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:

        if probability < 0.30:

            st.error("🔴 Delinquent")

        elif probability < 0.35:

            st.warning("🟠 Moderate")

        elif probability < 0.40:

            st.markdown(
                """
                <div style="
                    background-color:#fff9c4;
                    padding:12px;
                    border-radius:8px;
                    color:#5f5f00;
                    font-weight:bold;
                    font-size:20px;
                ">
                    🟡 Normal
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.success("🟢 Non-Delinquent")

    with col2:

        st.metric(
            "Predicted Delinquency Probability",
            f"{probability:.2%}"
        )

    # -----------------------------
    # Recommendation
    # -----------------------------
    st.subheader("Recommended Action")

    if dti > 0.40:

        st.warning(
            "Prioritize proactive outreach. "
            "Provide payment reminders and information about "
            "available repayment support."
        )

    elif missed_payments >= 3:

        st.warning(
            "Consider early payment-support outreach "
            "and monitor repayment behavior."
        )

    else:

        st.info(
            "Continue normal payment monitoring and "
            "standard customer communication."
        )