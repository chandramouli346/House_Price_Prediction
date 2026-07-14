import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ======================================
# Page Configuration
# ======================================

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv("data/Housing.csv")

# ======================================
# Sidebar Filters
# ======================================

st.sidebar.header("🔍 Dashboard Filters")

selected_bedrooms = st.sidebar.multiselect(
    "Select Bedrooms",
    options=sorted(df["bedrooms"].unique()),
    default=sorted(df["bedrooms"].unique()),
    key="bedroom_filter"
)

selected_furnishing = st.sidebar.multiselect(
    "Furnishing Status",
    options=sorted(df["furnishingstatus"].unique()),
    default=sorted(df["furnishingstatus"].unique()),
    key="furnishing_filter"
)

# Apply Filters
filtered_df = df[
    (df["bedrooms"].isin(selected_bedrooms)) &
    (df["furnishingstatus"].isin(selected_furnishing))
]

# ======================================
# Dashboard Title
# ======================================

st.title("📊 House Price Dashboard")

# ======================================
# KPI Cards
# ======================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🏠 Total Houses", len(filtered_df))

with col2:
    st.metric(
        "💰 Average Price",
        f"₹ {filtered_df['price'].mean():,.0f}"
    )

with col3:
    st.metric(
        "📉 Minimum Price",
        f"₹ {filtered_df['price'].min():,.0f}"
    )

with col4:
    st.metric(
        "📈 Maximum Price",
        f"₹ {filtered_df['price'].max():,.0f}"
    )

st.divider()

# ======================================
# Price Distribution & Area vs Price
# ======================================

left, right = st.columns(2)

with left:

    st.subheader("📈 Price Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.histplot(
        filtered_df["price"],
        bins=30,
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

with right:

    st.subheader("🏠 Area vs Price")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.scatterplot(
        data=filtered_df,
        x="area",
        y="price",
        ax=ax
    )

    st.pyplot(fig)

st.divider()

# ======================================
# Feature Importance
# ======================================

st.subheader("⭐ Feature Importance")

feature_importance = joblib.load("model/feature_importance.pkl")

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(10,5))

ax.barh(
    feature_importance["Feature"],
    feature_importance["Importance"],
    color="skyblue"
)

ax.set_title("Random Forest Feature Importance")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")

st.pyplot(fig)

st.divider()

# ======================================
# Correlation Heatmap
# ======================================

st.subheader("🔥 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])

corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax
)

st.pyplot(fig)