# Customer Churn Prediction & Analytics Dashboard

An end-to-end customer churn analytics project that combines machine learning, business-driven evaluation, experiment tracking, and an interactive dashboard to support data-driven retention decisions.

# Project Overview

Customer churn is a critical business problem for subscription-based companies.
This project predicts which customers are likely to churn, explains why, and provides actionable insights to help businesses design effective retention strategies.

The solution goes beyond model training by including:

Business-oriented evaluation

Threshold optimization

Experiment tracking with MLflow

A production-style Streamlit dashboard

Batch prediction and analytical views

# Key Objectives

Predict churn probability for individual customers

Optimize models for business impact (recall over accuracy)

Enable batch churn scoring for retention campaigns

Provide post-prediction analytics for decision-makers

Translate model outputs into business-friendly insights

# Dataset

Source: IBM Telco Customer Churn Dataset

Type: Realistic, industry-standard telecom dataset

Rows: ~7,000 customers

Target Variable: Churn (Yes / No)

Features include:

Customer demographics

Contract details

Service subscriptions

Billing and payment behavior

Synthetic data is additionally generated for batch-prediction demos while preserving real-world distributions.

# Feature Engineering

Key engineered features:

AvgChargesPerMonth – normalizes total charges by tenure

TotalServices – measures service dependency

LongTermContract – commitment indicator

These features improve business interpretability and stability.

# Modeling Approach
Models Trained

Logistic Regression (baseline)

Random Forest

XGBoost

Evaluation Strategy

Train/Test split (≈ 80/20)

Focus on Recall (Churn = 1) over accuracy

ROC-AUC and F1 for secondary comparison

Threshold tuning to balance recall vs business cost

Final Selection

Model chosen based on business objective: minimize missed churners

Final threshold calibrated for high recall

# Experiment Tracking (MLflow)

MLflow is used to:

Track experiments across multiple models

Log metrics (Recall, ROC-AUC, F1, Accuracy)

Store confusion matrices as artifacts

Register the final production model

This ensures reproducibility, transparency, and auditability.

# Interactive Dashboard (Streamlit)

The Streamlit app provides three core workflows:

# Home

Overview of the solution

Business context and usage guidance

# Single Customer Prediction

Input customer details

Get churn probability & risk level

Business-friendly interpretation

Optional AI explanation layer

# Batch Prediction

Upload CSV of customers

Score churn risk in bulk

Download enriched dataset with:

Churn probability

Risk level (Low / Medium / High)

# CSV Analysis Dashboard

High-level KPIs

Risk distribution

Churn risk by contract type

Churn risk by tenure group

Export high-risk customers for campaigns

# AI Integration (Explainability Layer) - in progress

AI is used only as a decision-support layer, not for prediction.

Explains why a customer is at risk

Suggests possible retention actions

Converts ML output into plain business language

AI does not replace the ML model or SHAP; it enhances communication.

# Project Structure
customer_churn_analytics/
├── app/                    # Streamlit dashboard
├── data/                   # Raw & synthetic datasets
├── models/                 # Final trained model
├── notebooks/              # EDA, preprocessing, training
├── src/                    # Feature engineering utilities
├── mlruns/                 # MLflow experiment tracking
├── requirements.txt
└── README.md

# How to Run the Project
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
cd app
python -m streamlit run app.py

# Business Value

Enables proactive customer retention

Supports campaign prioritization

Improves interpretability and trust

Bridges the gap between ML and business decisions

# Future Enhancements (Optional)

SHAP visual explanations in the dashboard

Cost-based threshold optimization

CRM integration

Cloud deployment

# Author

Pratyush Basu
(Project built for applied ML and consulting-oriented roles)

![PyPI version](https://img.shields.io/pypi/v/customer_churn_analytics.svg)
[![Documentation Status](https://readthedocs.org/projects/customer_churn_analytics/badge/?version=latest)](https://customer_churn_analytics.readthedocs.io/en/latest/?version=latest)

End-to-end customer churn analysis and prediction using data analytics and machine learning.

* PyPI package: https://pypi.org/project/customer_churn_analytics/
* Free software: MIT License
* Documentation: https://customer_churn_analytics.readthedocs.io.

## Features

* TODO

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
