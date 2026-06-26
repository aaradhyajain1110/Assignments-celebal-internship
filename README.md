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




## Week - 3
# Unsupervised Learning on Country Data


---

## 📌 Objective
To categorise 167 countries using socio-economic and health factors
that determine their overall development level, and help HELP International
NGO decide how to strategically allocate their $10 million aid fund.

---

## 📂 Dataset
- **Source:** Kaggle — Unsupervised Learning on Country Data
- **Link:** https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data
- **Size:** 167 rows × 10 columns
- **Features:** child_mort, exports, health, imports, income,
  inflation, life_expec, total_fer, gdpp

---

## 🛠️ Libraries Used
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (KMeans, DBSCAN, PCA, StandardScaler, silhouette_score)

---

## 📋 Notebook Workflow

| Step | Description |
|------|-------------|
| 1 | Install & import required libraries |
| 2 | Load dataset via Kaggle Hub API |
| 3 | Quick inspection (shape, dtypes, describe) |
| 4 | Data cleaning (strip, deduplicate, numeric, impute) |
| 5 | EDA — Correlation heatmap & boxplots |
| 6 | Feature scaling using StandardScaler |
| 7 | Elbow method to find optimal k |
| 8 | KMeans training with best_k = 3 |
| 9 | Silhouette Score evaluation |
| 10 | DBSCAN comparative clustering |
| 11 | PCA 2D visualization |
| 12 | Cluster profiling |
| 13 | Final insights & NGO recommendations |

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Optimal Clusters (k) | 3 |
| Silhouette Score | 0.2833 |
| PCA Variance Explained | 63.13% |
| DBSCAN Outliers Found | 30 countries |

---

## 🌍 Cluster Summary

| Cluster | Type | Countries | Avg GDPP | Avg Child Mortality |
|---------|------|-----------|----------|-------------------|
| Cluster 0 | Developed | 36 | $42,494 | 5.00 |
| Cluster 1 | Underdeveloped | 47 | $1,922 | 92.96 |
| Cluster 2 | Developing | 84 | $6,486 | 21.93 |

---

## 🔍 Key Observations

1. **High Mortality Cluster** — Cluster 1 has child mortality of 92.96
   per 1000 births. Countries like Afghanistan, Angola, and Nigeria
   need immediate aid.

2. **Top Economic Zone** — Cluster 0 includes Australia, Norway, USA
   with average GDPP of $42,494 and life expectancy of 80.13 years.

3. **Developing Nations** — Cluster 2 (84 countries) including India
   and China are transitioning but need targeted investment.

4. **DBSCAN Outliers** — 30 anomalous countries like Qatar, Singapore,
   and Haiti need individual assessment strategies.

5. **PCA Insights** — PC1 (45.95%) captures economic development axis.
   PC2 (17.18%) captures trade pattern variation.

---

## 🎯 NGO Recommendation

| Priority | Cluster | Action |
|----------|---------|--------|
| 🔴 Immediate Aid | Cluster 1 (47 countries) | Deploy funds to Afghanistan, Angola, Nigeria |
| 🟡 Support | Cluster 2 (84 countries) | Targeted healthcare & education investment |
| 🟢 Donor Nations | Cluster 0 (36 countries) | Partner for co-funding initiatives |
| ⚪ Individual Review | 30 DBSCAN Outliers | Case-by-case evaluation |

---
# Week 4 — CIFAR-10: ANN vs CNN

Image classification on CIFAR-10 comparing an Artificial Neural Network (ANN) against a Convolutional Neural Network (CNN), with student tasks exploring architecture depth, filter scaling, training duration, EarlyStopping, and data augmentation.

## Contents
- Baseline ANN (Dense + Dropout) and baseline CNN (Conv2D + BatchNorm + MaxPooling)
- Accuracy & loss curve comparisons
- 5 student tasks: deeper ANN, scaled CNN filters (32→64→128), 20-epoch training, EarlyStopping, data augmentation (RandomFlip/RandomRotation/RandomZoom)
- Final train/val/test accuracy comparison across all model variants

## Results Summary

| Model | Test Accuracy |
|---|---|
| Baseline ANN | ~42% |
| Deeper ANN | ~39-41% |
| Baseline CNN | ~67-68% |
| Scaled CNN (32→64→128) | ~70-73% |
| CNN, 20 epochs | ~72-73% |
| CNN + EarlyStopping | ~72-73% |
| Augmented CNN | ~70% (best generalization, smallest train/val gap) |

## Key Takeaway
CNNs significantly outperform ANNs on image data by preserving spatial structure. Among training strategies, data augmentation produced the most generalizable model — not the highest raw accuracy, but the smallest gap between train and validation accuracy.

## Tech Stack
TensorFlow / Keras, Matplotlib, NumPy, Pandas

## Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10.html) — 60,000 32×32 color images across 10 classes (50,000 train / 10,000 test)

# Week 5 — Text Generation using RNN, LSTM & GRU

A deep learning project comparing Vanilla RNN, LSTM, and GRU architectures for next-word prediction and text generation using TensorFlow/Keras.

---

## 📌 Objective
Build and compare three sequence models on a text corpus to understand how gated architectures handle language patterns better than Vanilla RNN.

---

## 🧠 Models Compared
| Model | Gates | Strength |
|---|---|---|
| Vanilla RNN | None | Simple, fast, short patterns only |
| LSTM | Input, Forget, Output | Best for long-range dependencies |
| GRU | Reset, Update | Faster than LSTM, similar performance |

---

## 📂 What's Inside
- Data preprocessing — tokenization and n-gram sequence building
- Three model architectures built with identical optimizer config
- Training loss comparison plot across 100 epochs
- Text generation using `np.argmax` over probability arrays
- 5 beginner customization tasks

---

## ✅ Student Tasks Completed
1. **Custom Corpus** — replaced boilerplate with 10-sentence AI/ML paragraph
2. **Embedding Dim** — increased from 32 → 64
3. **Epochs** — expanded training from 100 → 200
4. **Hidden Units** — widened layers from 64 → 128
5. **Text Generation** — extended output from 5 → 10 words

---

## 📊 Key Findings
- LSTM and GRU converge faster and lower than Vanilla RNN
- GRU matches LSTM performance with fewer parameters
- Larger corpus and more epochs improve generation quality
- Vanilla RNN loses coherence on longer generated sequences

---

