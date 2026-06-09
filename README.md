# Credit-Card-Spend-Prediction
Predict customers' monthly credit card spending using machine learning based on demographic, credit, and behavioral features.

# 💳 Credit Card Monthly Spend Prediction

## 📌 Project Overview

This project builds machine learning models to predict customers' **monthly credit card spending** based on demographic profiles, credit history, and transaction behavior. The goal is to accurately forecast individual spending levels using multi-dimensional customer data.

## 📊 Dataset

- **Training set**: 40,000 records, 23 features
- **Test set**: 10,000 records, 22 features
- **Target variable**: `monthly_spend`

### Key Features

| Category | Features |
|----------|----------|
| Demographics | age, gender, marital_status, num_children, education_level, region, employment_status |
| Financial | annual_income, credit_score, credit_limit, has_auto_loan, owns_home |
| Behavioral | num_transactions, avg_transaction_value, online_shopping_freq, reward_points_balance, travel_frequency, utility_payment_count |

## 🔍 Exploratory Data Analysis (EDA)

### Key Findings

1. **Target Distribution**: `monthly_spend` is approximately normally distributed with a spending floor around **$100** ("floor customers")

2. **Correlation Analysis** (with monthly_spend):
   - `credit_limit`: **0.72** (strongest)
   - `credit_score`: 0.50
   - `annual_income`: 0.43
   - `reward_points_balance`: 0.43

3. **Multicollinearity Warning**:
   - `credit_limit` ↔ `annual_income`: 0.89
   - `credit_limit` ↔ `reward_points`: 0.88
   - `annual_income` ↔ `reward_points`: 0.98
   
   > ⚠️ These three features represent essentially the same signal and will break linear models.

4. **Categorical Feature Insights**:
   - `card_type`: Standard card users have the lowest spending (~$100 floor); Platinum card users have the highest median (~$780)

## 🏗️ Modeling Strategy

### Two-Stage Framework

1. **Stage 1 - Classification**
   - Identify "floor customers" (spending ≤ $100)
   - LightGBM classifier achieves **AUC of 0.961** out-of-fold

2. **Stage 2 - Regression**
   - Use classifier probabilities as a feature
   - Predict precise spending amounts for non-floor customers

> 💡 This is the **single most impactful design decision** of the project.

## 🛠️ Models Used

| Model | Purpose |
|-------|---------|
| LightGBM | Classification (floor customer detection) + Regression |
| XGBoost | Regression |
| CatBoost | Regression |
| MLPRegressor | Neural network regression |

## 📈 Visualizations

The notebook includes:
- Target variable distribution plot
- Full feature correlation heatmap
- Box/Violin/Strip plots for categorical features vs. target

## 🚀 Quick Start

### Requirements
```bash
pip install pandas numpy lightgbm xgboost catboost scikit-learn matplotlib seaborn
