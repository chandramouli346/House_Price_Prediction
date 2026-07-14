import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load Logo
logo = Image.open("images/logo.png")

# Display Logo
st.image(logo, width=180)

# Title
st.title("🏠 House Price Prediction Dashboard")

# Welcome Text
st.markdown("""
# Welcome 👋

This is a House Price Prediction project built using **Machine Learning**.

### Technologies Used
- 🐍 Python
- 📊 Pandas
- 🤖 Scikit-learn
- 🌐 Streamlit
- 🌲 Random Forest

### Features
- 📊 Dashboard
- 🏠 House Price Prediction
- 📈 Exploratory Data Analysis
- 🤖 Model Information
- 👨‍💻 About Project

👈 **Use the left sidebar to navigate between pages.**
""")