# 🇮🇳 Indian IPO Subscription Prediction

An end-to-end machine learning project that predicts the subscription category of Indian IPOs using historical data, market conditions, and company fundamentals.

## 📌 Problem Statement
IPO subscription prediction is a critical problem in Indian capital markets. This project builds a classification model to predict whether an IPO will be under-subscribed, low, medium, or highly subscribed — before the subscription window opens.

## 🎯 Target Variable
| Category | Subscription Range |
|---|---|
| Under-subscribed | < 1x |
| Low | 1x – 10x |
| Medium | 10x – 50x |
| High | 50x+ |

## 📊 Dataset
- 89 Indian IPOs (2019–2024)
- Built from scratch — no Kaggle dataset used
- Enriched with Nifty 50 market data via yfinance

## 🔑 Features Used
| Feature | Description |
|---|---|
| issue_price | IPO issue price in ₹ |
| issue_size_cr | Total issue size in ₹ Crore |
| issue_size_log | Log-transformed issue size |
| nifty_avg | Nifty 50 average for IPO year |
| nifty_volatility | Market volatility during IPO year |
| nifty_yearly_return | Nifty return % during IPO year |
| sector_encoded | Encoded sector of the company |

## 🧠 Models Compared
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost ✅ (Best)
- LightGBM

## ⚙️ Methodology
1. Data collection & pipeline building
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model training with 5-Fold Stratified Cross Validation
5. Hyperparameter tuning with Optuna (100 trials)
6. Feature importance analysis
7. Streamlit dashboard deployment

## 🚀 How to Run

### Setup
```bash
conda create -n ipo_project python=3.11 -y
conda activate ipo_project
pip install -r requirements.txt
```

### Run Dashboard
```bash
streamlit run app/dashboard.py
```

## 📁 Project Structure
ipo-subscription-prediction/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modelling.ipynb
├── app/
│   └── dashboard.py
├── models/
└── reports/
## 📈 Key Findings
- Issue size is the strongest predictor — larger IPOs tend to be less subscribed
- Nifty yearly return significantly impacts subscription sentiment
- Sector plays a moderate role in subscription behaviour
- Market volatility has lower predictive power than expected

## 🛠️ Tech Stack
- **Data:** pandas, numpy, yfinance
- **Modelling:** scikit-learn, XGBoost, LightGBM, Optuna
- **Explainability:** XGBoost Feature Importance
- **Deployment:** Streamlit
- **Experiment Tracking:** MLflow
- **Version Control:** Git & GitHub

## 👤 Author
**Faizan** — GATE DA 2026 | AIR 2600