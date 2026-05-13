import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="IPO Subscription Predictor",
    page_icon="📈",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load(r"C:\Projects\ipo-subscription-prediction\models\xgboost_final.pkl")
    le = joblib.load(r"C:\Projects\ipo-subscription-prediction\models\label_encoder.pkl")
    return model, le

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Projects\ipo-subscription-prediction\data\processed\ipo_enriched.csv")

model, le = load_model()
df = load_data()

# Sidebar
st.sidebar.title("📈 IPO Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Predict", "Dashboard", "Data Explorer"])

# ─── PAGE 1: PREDICT ───
if page == "Predict":
    st.title("🔮 IPO Subscription Predictor")
    st.markdown("Enter IPO details below to predict subscription category.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("IPO Details")
        company_name = st.text_input("Company Name", placeholder="e.g. TechCorp Ltd")
        issue_price = st.number_input("Issue Price (₹)", min_value=1, max_value=10000, value=500)
        issue_size = st.number_input("Issue Size (₹ Crore)", min_value=1, max_value=50000, value=2000)
        sector = st.selectbox("Sector", sorted(df["sector"].unique()))

    with col2:
        st.subheader("Market Conditions")
        nifty_avg = st.number_input("Nifty 50 Average (current year)", min_value=1000, max_value=100000, value=22000)
        nifty_volatility = st.number_input("Nifty Volatility (%)", min_value=0.1, max_value=5.0, value=0.9, step=0.1)
        nifty_return = st.number_input("Nifty Yearly Return (%)", min_value=-50.0, max_value=100.0, value=15.0, step=0.5)

    st.markdown("---")

    if st.button("🚀 Predict Subscription", use_container_width=True):
        sector_encoded = le.transform([sector])[0] if sector in le.classes_ else 0
        issue_size_log = np.log1p(issue_size)

        features = pd.DataFrame([{
            "issue_price": issue_price,
            "issue_size_cr": issue_size,
            "issue_size_log": issue_size_log,
            "nifty_avg": nifty_avg,
            "nifty_volatility": nifty_volatility,
            "nifty_yearly_return": nifty_return,
            "sector_encoded": sector_encoded
        }])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        labels = ["Under-subscribed", "Low (1–10x)", "Medium (10–50x)", "High (50x+)"]
        colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
        emojis = ["⚠️", "📊", "📈", "🚀"]

        st.markdown("### Prediction Result")
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric("Predicted Category", f"{emojis[prediction]} {labels[prediction]}")
        with col_r2:
            st.metric("Confidence", f"{probabilities[prediction]*100:.1f}%")
        with col_r3:
            st.metric("Company", company_name if company_name else "N/A")

        st.markdown("#### Probability Distribution")
        prob_df = pd.DataFrame({
            "Category": labels,
            "Probability": probabilities
        })

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(labels, probabilities, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Probability")
        ax.set_title("Subscription Category Probabilities", fontweight="bold")
        ax.set_ylim(0, 1)
        for bar, prob in zip(bars, probabilities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{prob:.2f}", ha="center", fontweight="bold")
        st.pyplot(fig)

# ─── PAGE 2: DASHBOARD ───
elif page == "Dashboard":
    st.title("📊 IPO Market Dashboard")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total IPOs", len(df))
    col2.metric("Avg Subscription", f"{df['total_sub'].mean():.1f}x")
    col3.metric("Highest Ever", f"{df['total_sub'].max():.0f}x")
    col4.metric("Success Rate (>1x)", f"{(df['total_sub']>=1).mean()*100:.0f}%")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Subscription by Sector")
        sector_avg = df.groupby("sector")["total_sub"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(sector_avg.index, sector_avg.values, color="#3498db", edgecolor="white")
        ax.set_xlabel("Avg Subscription (x)")
        ax.set_title("Top 10 Sectors", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    with col_r:
        st.subheader("Year-wise IPO Trend")
        year_data = df.groupby("year").agg(
            count=("company", "count"),
            avg_sub=("total_sub", "mean")
        ).reset_index()
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.bar(year_data["year"], year_data["count"], color="#3498db", alpha=0.6, label="Count")
        ax2.plot(year_data["year"], year_data["avg_sub"], color="#e74c3c", marker="o", linewidth=2, label="Avg Sub")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Number of IPOs", color="#3498db")
        ax2.set_ylabel("Avg Subscription (x)", color="#e74c3c")
        ax1.set_title("IPO Count & Avg Subscription by Year", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Top 10 Most Subscribed IPOs")
    top10 = df.nlargest(10, "total_sub")[["company", "year", "sector", "total_sub", "listing_gain_pct"]]
    top10.columns = ["Company", "Year", "Sector", "Subscription (x)", "Listing Gain (%)"]
    st.dataframe(top10, use_container_width=True)

# ─── PAGE 3: DATA EXPLORER ───
elif page == "Data Explorer":
    st.title("🔍 Data Explorer")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        selected_sector = st.multiselect("Filter by Sector", df["sector"].unique(), default=df["sector"].unique()[:3])
    with col2:
        selected_year = st.multiselect("Filter by Year", sorted(df["year"].unique()), default=sorted(df["year"].unique()))

    filtered = df[df["sector"].isin(selected_sector) & df["year"].isin(selected_year)]
    st.markdown(f"Showing **{len(filtered)}** IPOs")

    display_cols = ["company", "year", "sector", "issue_price", "issue_size_cr", "total_sub", "listing_gain_pct", "subscription_label"]
    st.dataframe(filtered[display_cols].sort_values("total_sub", ascending=False), use_container_width=True)