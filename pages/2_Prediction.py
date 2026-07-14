import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# ----------------------------
# Indian Currency Format
# ----------------------------
def format_indian_currency(num):
    num = int(num)
    s = str(num)

    if len(s) <= 3:
        return s

    last3 = s[-3:]
    rest = s[:-3]

    parts = []

    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]

    if rest:
        parts.insert(0, rest)

    return ",".join(parts) + "," + last3


# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model/best_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")


# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Prediction",
    page_icon="🏠",
    layout="wide"
)

# Image Styling
st.markdown("""
<style>
[data-testid="stImage"] img{
    border-radius:20px;
    border:2px solid #dddddd;
    box-shadow:0px 6px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.title("🏠 House Price Prediction")
st.subheader("Enter House Details")

# ----------------------------
# Input Fields
# ----------------------------

left, right = st.columns(2)

with left:
    area = st.number_input("Area (sq ft)", 500, 20000, 2000)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 10, 2)
    stories = st.slider("Stories", 1, 5, 2)
    parking = st.slider("Parking", 0, 5, 1)

with right:
    mainroad = st.selectbox("Main Road", ["yes", "no"])
    guestroom = st.selectbox("Guest Room", ["yes", "no"])
    basement = st.selectbox("Basement", ["yes", "no"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])
    furnishingstatus = st.selectbox(
        "Furnishing Status",
        ["furnished", "semi-furnished", "unfurnished"]
    )

st.divider()

# ----------------------------
# Prediction Button
# ----------------------------

if st.button("🏠 Predict House Price", use_container_width=True):

    input_data = pd.DataFrame({
        "area":[area],
        "bedrooms":[bedrooms],
        "bathrooms":[bathrooms],
        "stories":[stories],
        "mainroad":[mainroad],
        "guestroom":[guestroom],
        "basement":[basement],
        "hotwaterheating":[hotwaterheating],
        "airconditioning":[airconditioning],
        "parking":[parking],
        "prefarea":[prefarea],
        "furnishingstatus":[furnishingstatus]
    })

    categorical_cols = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
        "furnishingstatus"
    ]

    for col in categorical_cols:
        input_data[col] = label_encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)
    formatted_price = format_indian_currency(prediction[0])

    # ----------------------------
    # Result Layout
    # ----------------------------

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.markdown(f"""
        <div style="
            height:370px;
            background:#d4edda;
            border:2px solid green;
            border-radius:20px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
        ">

        <h2 style="color:green;">🏡 Estimated House Price</h2>

        <h1 style="color:#0b7a39;font-size:55px;">
            ₹ {formatted_price}
        </h1>

        <h3>Prediction Successful ✅</h3>

        </div>
        """, unsafe_allow_html=True)

    with result_col2:

        image = Image.open("images/house.png")

        st.image(
            image,
            use_container_width=True,
        )