# Assignments-celebal-internship
Celebal Internship — ML Assignments

Aaradhya Jain

This repository contains weekly assignments completed as part of the Celebal Technologies Machine Learning Internship.

## Week 1 — ML Foundations
File: week1_Aaradhya_Jain_clean.ipynb
A comprehensive notebook covering the mathematical and programmatic foundations of Machine Learning.
 Topics Covered
PartTopicKey ConceptsPart 1Python FundamentalsData types, control flow, exceptions, functions & lambdasPart 2NumPyArray creation, indexing, slicing, matrix operations, dot productsPart 3PandasDataFrames, Series, iloc/loc, filtering, groupby, missing dataPart 4Linear AlgebraEigenvalues/eigenvectors, SVD, PCAPart 5StatisticsDescriptive stats, distributions, hypothesis testing, PSIPart 6Probability TheoryBayes' Theorem, distributions, Central Limit Theorem
Part Breakdown


Part 1 — Python Fundamentals

classify_number() — number classification using control flow
Word frequency counter using dicts, sets, and list comprehensions
safe_divide() — exception handling with ZeroDivisionError & TypeError
Higher-order functions and lambda expressions

Part 2 — NumPy

Array reshaping: 1D → 2D (3,4) → 3D (2,2,3)
Boolean indexing and array slicing
Element-wise vs matrix multiplication (* vs @)
Dot products and scalar operations

Part 3 — Pandas

Employee dataset with 10 records across Engineering, Marketing, and HR
iloc/loc selection, department-wise groupby aggregations
Missing value imputation using median salary and mean age

Part 4 — Linear Algebra

Eigenvalue/eigenvector decomposition
Singular Value Decomposition (SVD)
Principal Component Analysis (PCA)

Part 5 — Statistics

Descriptive statistics: mean, median, variance, standard deviation
Probability distributions and hypothesis testing
Population Stability Index (PSI) for model monitoring

Part 6 — Probability Theory

Marble bag problem: joint and conditional probabilities
Normal, Binomial, and Poisson distribution plots with ML use-cases
Bayes' Theorem applied to spam filtering
Central Limit Theorem demonstrated on exponential population (n=5000 samples, KS test validation)




## Week - 2 
## Tesla EV Deliveries — End-to-End ML Pipeline (2015–2025)

 Overview
An end-to-end machine learning pipeline built on Tesla's global delivery and production dataset spanning 10 years (2015–2025). The pipeline covers data preprocessing, exploratory data analysis, feature engineering, regression modeling, hyperparameter tuning, and time series forecasting.

**Dataset:** [Tesla EA Deliveries and Production Data (2015–2025) — Kaggle](https://www.kaggle.com/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025)  
**Records:** 2,640 rows × 12 features | 4 Regions | 5 Models

---

## Pipeline Structure

| Block | Stage | Description |
|-------|-------|-------------|
| 1 | Data Loading | Load dataset, inspect shape, dtypes, and sample rows |
| 2 | EDA | Null checks, statistics, distributions, correlations, heatmap |
| B | Enhanced EDA | Boxplots, outlier detection (IQR), yearly trends, seasonality |
| 3 | Preprocessing | Label encoding, feature engineering, lag features, train/test split, scaling |
| 4 | Regression Models | Linear Regression, Ridge, Lasso — baseline comparison |
| 5 | Cross Validation | TimeSeriesSplit (5-fold) — chronological, leakage-free validation |
| 6 | ADF Test | Augmented Dickey-Fuller stationarity test + rolling mean/std plot |
| 7 | Ensemble Models | Random Forest, XGBoost — advanced non-linear modeling |
| 8 | Hyperparameter Tuning | GridSearchCV on XGBoost — optimal params selection |
| 9 | Time Series Visualization | Actual vs predicted delivery plot across full dataset |
| 10 | Final Evaluation | MAE, RMSE, R² on held-out test set |
| — | Forecast | 2026 quarterly delivery predictions using tuned XGBoost |

---

## Key Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 108.71 | 153.49 | 0.9984 |
| Ridge | 108.84 | 154.05 | 0.9984 |
| Lasso | 108.23 | 152.66 | 0.9984 |
| Random Forest | 67.64 | 108.14 | 0.9992 |
| **XGBoost (Tuned)** | **53.84** | **75.49** | **0.9996** |

**Best Model:** Tuned XGBoost with `learning_rate=0.1`, `max_depth=5`, `n_estimators=200`

---

## Feature Engineering
- `Lag_1`, `Lag_2` — previous month delivery values
- `Rolling_Mean_3`, `Rolling_Std_3` — 3-month rolling statistics
- `Production_Efficiency` — deliveries / production units
- `Est_Revenue_USD` — deliveries × avg price
- `CO2_per_Delivery` — environmental impact per unit
- `Range_per_kWh` — battery efficiency metric
- `Quarter` — seasonal grouping

---

## 2026 Forecast

| Quarter | Predicted Deliveries |
|---------|---------------------|
| Q1 2026 | ~9,900 |
| Q2 2026 | ~9,900 |
| Q3 2026 | ~9,900 |
| Q4 2026 | ~9,900 |

*Forecast uses tuned XGBoost with lag-based autoregression on mean feature values.*

---

## Tech Stack
- **Language:** Python 3.12
- **Libraries:** pandas, numpy, scikit-learn, xgboost, statsmodels, matplotlib, seaborn
- **Environment:** Kaggle Notebooks (GPU/CPU)

---

## Key Insights
- **XGBoost** outperforms all linear models with a 75% reduction in RMSE
- **Production_Units** and **lag features** are the strongest delivery predictors
- **Q4 seasonality** — Tesla consistently peaks in October–December
- **ADF test** confirms stationarity — no differencing needed before forecasting
- **North America** leads in delivery volume across all years
- **Model 3 & Model Y** are the primary volume drivers
