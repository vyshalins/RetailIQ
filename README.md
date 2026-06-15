# RetailIQ – Causal Sales Forecasting Platform

## Overview

RetailIQ is a retail demand forecasting platform built on Walmart's M5 Forecasting dataset. The project combines machine learning and causal inference techniques to forecast product demand while evaluating the true impact of promotional activities on sales.

The system integrates LightGBM forecasting, feature engineering, DoWhy-based causal inference, Propensity Score Matching (PSM), Counterfactual Analysis, SHAP explainability, and MLflow experiment tracking.

---

## Architecture

```text
M5 Dataset
     ↓
Data Preprocessing
     ↓
Feature Engineering
     ↓
LightGBM Forecasting
     ↓
Causal Inference
 ├─ DoWhy DAG
 ├─ Backdoor Adjustment
 ├─ Propensity Score Matching
 └─ Counterfactual Analysis
     ↓
Model Evaluation
 ├─ RMSE
 ├─ MAE
 ├─ WAPE
 └─ SHAP
     ↓
MLflow Tracking
     ↓
Business Insights
```

---

## Dataset

This project uses Walmart's M5 Forecasting Dataset from Kaggle.

### Files Used

* sales_train_validation.csv
* calendar.csv
* sell_prices.csv

### Dataset Characteristics

* 30,490 item-store combinations
* 1,913 days of sales history
* 10 stores across 3 states
* 3 product categories
* 7 departments
* Weekly price information
* Calendar events and SNAP indicators

---

## Key Features

### Time-Series Forecasting

* LightGBM demand forecasting model
* Time-based train-validation split
* Retail demand prediction across stores and products

### Feature Engineering

Demand Features:

* lag_7
* lag_14
* lag_28
* rolling_mean_7
* rolling_mean_28

Pricing Features:

* sell_price
* price_change_pct
* discount_flag

Calendar Features:

* month
* week_of_year
* is_weekend
* has_event
* snap_CA
* snap_TX
* snap_WI

### Causal Inference

* Causal DAG construction using DoWhy
* Backdoor adjustment for treatment effect estimation
* Average Treatment Effect (ATE) estimation
* Refutation testing
* Propensity Score Matching (PSM)
* Counterfactual sales estimation

### Model Interpretability

* SHAP feature importance analysis
* Actual vs Predicted visualization

### Experiment Tracking

* MLflow experiment tracking
* Metric comparison across forecasting models
* Artifact logging and reproducibility

---

## Results

### Forecasting Performance

| Metric | Baseline LightGBM | Causal LightGBM |
| ------ | ----------------: | --------------: |
| RMSE   |            1.8563 |          1.8653 |
| MAE    |            0.8407 |          0.8415 |
| WAPE   |            98.84% |          98.92% |

### Causal Inference Results

| Method                | Estimated Effect |
| --------------------- | ---------------: |
| DoWhy ATE             |          -0.0557 |
| PSM Treatment Effect  |          +0.2165 |
| Counterfactual Uplift |          +0.2165 |

### SHAP Insights

Top forecasting drivers:

1. rolling_mean_28
2. is_weekend
3. sell_price
4. rolling_mean_7
5. lag_7

Historical demand patterns were significantly more influential than promotional variables.

---

## Key Findings

* Historical demand features were the strongest predictors of future sales.
* Promotional interventions were extremely sparse in the sampled dataset (0.26% of observations).
* DoWhy and Propensity Score Matching produced different treatment effect estimates, highlighting the importance of using multiple causal estimation approaches.
* Counterfactual analysis estimated an average promotion uplift of approximately 0.22 sales per treated observation.
* Causal correction provided valuable business insights even though forecasting accuracy improvements were minimal.

---

## Tech Stack

### Data Processing

* Python
* Pandas
* NumPy

### Forecasting

* LightGBM
* Scikit-Learn

### Causal Inference

* DoWhy
* Propensity Score Matching

### Explainability

* SHAP

### Experiment Tracking

* MLflow

### Development Environment

* Jupyter Notebook
* VS Code

---

## Future Improvements

* Deploy forecasting model using FastAPI
* Build an interactive dashboard for demand forecasting
* Incorporate additional promotion and marketing variables
* Evaluate advanced causal methods such as Double Machine Learning (DML)
* Scale the pipeline to the complete M5 dataset

---

## Author

Vyshalini S

Built as an end-to-end Machine Learning and Causal Inference project for retail demand forecasting and promotional impact analysis.
