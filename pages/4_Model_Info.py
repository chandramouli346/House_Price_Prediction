import streamlit as st

st.set_page_config(
    page_title="Model Information",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Model Information")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    st.subheader("📌 Model Details")

    st.info("""
**Algorithm Used:** Random Forest Regressor

**Dataset:** Housing.csv

**Training Data:** 80%

**Testing Data:** 20%

**Target Variable:** Price

**Features Used:** 12
""")

with col2:

    st.subheader("📊 Model Performance")

    st.metric("MAE", "₹ 10,25,290")
    st.metric("RMSE", "₹ 14,01,263")
    st.metric("R² Score", "0.612")

st.markdown("---")

st.subheader("🛠 Technologies Used")

st.success("""
✔ Python

✔ Pandas

✔ NumPy

✔ Scikit-learn

✔ Random Forest Regressor

✔ Joblib

✔ Streamlit
""")

st.markdown("---")

st.subheader("📖 About the Model")

st.write("""
The Random Forest Regressor is an ensemble machine learning algorithm that
combines multiple decision trees to improve prediction accuracy.

The model was trained using the Housing dataset and predicts house prices
based on property features such as area, bedrooms, bathrooms, stories,
parking, furnishing status, and other amenities.

Model evaluation was performed using MAE, RMSE, and R² Score.
""")