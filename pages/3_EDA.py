import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA", page_icon="📈", layout="wide")

df = pd.read_csv("data/Housing.csv")

st.title("📈 Exploratory Data Analysis")

# -----------------------
# Price Distribution
# -----------------------

st.subheader("Price Distribution")

fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df["price"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# -----------------------
# Box Plot
# -----------------------

st.subheader("House Price Box Plot")

fig, ax = plt.subplots(figsize=(8,4))
sns.boxplot(y=df["price"], ax=ax)
st.pyplot(fig)

# -----------------------
# Furnishing Status
# -----------------------

st.subheader("Furnishing Status")

fig, ax = plt.subplots(figsize=(8,4))
sns.countplot(data=df, x="furnishingstatus", ax=ax)
st.pyplot(fig)

# -----------------------
# Correlation Heatmap
# -----------------------

st.subheader("Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)