import streamlit as st

st.set_page_config(
    page_title="Model Info",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Model Information")

st.markdown("""
## Model Used

✅ **Random Forest Regressor**

The Random Forest Regressor is an ensemble machine learning algorithm that combines multiple decision trees to predict house prices accurately.

---

## Why Random Forest?

- Handles large datasets efficiently
- Reduces overfitting
- High prediction accuracy
- Works well with numerical and categorical features

---

## Features Used

- Area
- Bedrooms
- Bathrooms
- Stories
- Main Road
- Guest Room
- Basement
- Hot Water Heating
- Air Conditioning
- Parking
- Preferred Area
- Furnishing Status

---

## Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
""")