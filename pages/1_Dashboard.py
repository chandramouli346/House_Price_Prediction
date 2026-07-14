import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# Load dataset
df = pd.read_csv("data/Housing.csv")

st.title("📊 House Price Dashboard")

# ==========================
# KPI Cards
# ==========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🏠 Total Houses", len(df))

with col2:
    st.metric("💰 Average Price", f"₹ {df['price'].mean():,.0f}")

with col3:
    st.metric("📉 Minimum Price", f"₹ {df['price'].min():,.0f}")

with col4:
    st.metric("📈 Maximum Price", f"₹ {df['price'].max():,.0f}")

st.divider()

# ==========================
# Charts
# ==========================

left, right = st.columns(2)

with left:

    st.subheader("Price Distribution")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df["price"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

with right:

    st.subheader("Area vs Price")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df, x="area", y="price", ax=ax)
    st.pyplot(fig)