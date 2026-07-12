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

# Week 6 — Image Denoising with Autoencoders on MNIST

**Author:** Aaradhya Jain

This notebook builds and compares four autoencoder architectures for removing Gaussian noise from MNIST handwritten digit images. It covers a feedforward (dense) baseline and three convolutional variants, trained and evaluated side by side using reconstruction loss, PSNR, and SSIM.

## Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Models](#models)
5. [Training Setup](#training-setup)
6. [Results](#results)
7. [Key Takeaways](#key-takeaways)
8. [Requirements](#requirements)
9. [Usage](#usage)
10. [Outputs](#outputs)

## Overview

Given a clean MNIST digit corrupted with Gaussian noise, each model learns to reconstruct the original, denoised image. The notebook walks through:

- Environment/GPU check (T4 recommended)
- Downloading and loading the dataset
- Adding synthetic noise to create paired (noisy, clean) training data
- Building, training, and evaluating four autoencoder variants
- Visual and quantitative comparison across all models

## Dataset

- **Source:** [`awsaf49/mnist-dataset`](https://www.kaggle.com/datasets/awsaf49/mnist-dataset) on Kaggle, downloaded via `kagglehub`
- **Format:** MNIST digits as PNG images, organized into per-digit folders (`0`–`9`) under `mnist_png/training` and `mnist_png/testing`
- **Size:** 60,000 training images, 10,000 test images, 28×28 grayscale

## Preprocessing

1. Load PNGs into NumPy arrays via PIL (`'L'` grayscale mode)
2. Normalize pixel values from `[0, 255]` to `[0, 1]`
3. Reshape to `(28, 28, 1)` to add a channel dimension
4. Generate noisy versions by adding Gaussian noise:
   ```python
   noisy = images + noise_factor * np.random.normal(0.0, 1.0, size=images.shape)
   noisy = np.clip(noisy, 0.0, 1.0)
   ```
   with `noise_factor = 0.4`
5. For the FFNN model, images are additionally flattened to 784-length vectors

## Models

| # | Model | Encoder | Decoder | Loss |
|---|---|---|---|---|
| 1 | **FFNN AE** | `Dense(64, relu)` on flattened 784-dim input | `Dense(784, sigmoid)` | MSE |
| 2 | **Transpose CNN AE** | `Conv2D` → `BatchNorm` → `LeakyReLU` → `MaxPooling2D` (×2, 28→14→7) | `Conv2DTranspose` (strided, learned upsampling) → `BatchNorm` → `LeakyReLU` (×2, 7→14→28), final `Conv2D(1, sigmoid)` | MSE |
| 3 | **Upsampled CNN AE (MSE)** | `Conv2D(32, relu)` + `MaxPooling2D` (×2, 28→14→7) | `Conv2D(32, relu)` + `UpSampling2D` (×2, 7→14→28), final `Conv2D(1, sigmoid)` | MSE |
| 4 | **Upsampled CNN AE (BCE)** | Same architecture as #3 | Same architecture as #3 | Binary Cross-Entropy |

**Why compare these four?** They isolate two independent design choices:
- **Dense vs. convolutional** (#1 vs. #2/#3/#4) — does preserving 2D spatial structure matter?
- **Learned vs. non-learned upsampling** (#2 vs. #3) — does letting the decoder learn its own upsampling filters (`Conv2DTranspose`) beat fixed nearest-neighbor `UpSampling2D`?
- **MSE vs. BCE loss** (#3 vs. #4) — does treating pixels as near-binary probabilities (BCE) beat treating them as continuous values (MSE), given pixel intensities are normalized to `[0, 1]`?

## Training Setup

- **Optimizer:** Adam (default settings for models 1, 3, 4; explicit `learning_rate=0.001` for model 2)
- **Epochs:** 20 (all models)
- **Batch size:** 128
- **Shuffle:** enabled
- **Validation:** noisy test set → clean test set, evaluated every epoch
- **Input/target pairs:** `(noisy image, clean image)` for all models

## Results

Evaluated on the full 10,000-image test set.

| Model | PSNR (dB) | SSIM | PSNR Gain over Noisy |
|---|---|---|---|
| Noisy input (no model) | 10.99 | 0.446 | — |
| FFNN AE | 19.51 | 0.790 | +8.51 dB |
| Transpose CNN AE | 21.35 | 0.898 | +10.35 dB |
| Upsampled CNN AE (MSE) | *computed at runtime* | *computed at runtime* | *computed at runtime* |
| **Upsampled CNN AE (BCE)** | **21.46** | **0.907** | **+10.46 dB** |

> The Upsampled CNN AE (MSE) metrics are computed dynamically in the notebook (`psnr_model5`, `ssim_model5`) rather than hardcoded — run the notebook to populate these values, or check the printed test-set evaluation output.

**Winner: Upsampled CNN AE (BCE)** — highest PSNR and SSIM of all four models.

## Key Takeaways

1. **Convolutions beat dense layers.** All three CNN-based models outperform the FFNN AE. Flattening an image into a 784-length vector throws away spatial relationships between neighboring pixels; convolutional filters exploit local structure (edges, strokes) that a dense bottleneck cannot.
2. **Loss function matters as much as architecture.** Swapping MSE for BCE on the *identical* Upsampled CNN architecture (#3 → #4) produces a bigger SSIM improvement than switching to learned transposed-convolution upsampling (#3 → #2). Since MNIST pixels are mostly pure black or pure white, BCE — which penalizes uncertain, "gray" predictions far more steeply near 0 and 1 — pushes the model toward sharper, higher-contrast, more confident reconstructions than MSE's tendency to average toward gray.
3. **Learned upsampling gives a smaller, secondary boost.** The Transpose CNN AE's strided `Conv2DTranspose` layers modestly outperform plain MSE-trained `UpSampling2D`, but this effect is smaller than the effect of choosing the right loss function.
4. **Net effect:** convolutional structure + a loss function matched to the data distribution (BCE for near-binary, [0,1]-normalized pixels) together produce the best denoising results — the Upsampled CNN AE (BCE) model.

## Requirements

```
tensorflow
numpy
pillow
matplotlib
scikit-image
kagglehub
```

GPU (e.g., NVIDIA T4) recommended but not required — training will fall back to CPU (slower) otherwise.

## Usage

Run top to bottom in Google Colab (recommended, GPU runtime) or a local Jupyter environment:

```bash
jupyter notebook week6_Aaradhya_Jain.ipynb
```

Kaggle API credentials must be configured for `kagglehub.dataset_download` to fetch the dataset.

## Outputs

Trained models are saved to disk in Keras format:

- `ffnn_autoencoder_mnist.keras`
- `transpose_cnn_autoencoder_mnist.keras`
- `upsampled_cnn_autoencoder_mnist.keras` *(BCE-trained model)*

The notebook also produces:
- Clean vs. noisy sample visualizations
- Combined train/validation loss curves for all four models
- Side-by-side grid of original / noisy / denoised outputs across all models
- Bar charts comparing PSNR, SSIM, and PSNR improvement over the noisy baseline

# 📄 Week 7 — Document Question Answering System (RAG)

**Author:** Aaradhya Jain
**Notebook:** `week7_Aaradhya_Jain_.ipynb`

A Retrieval-Augmented Generation (RAG) system that answers questions grounded in
custom documents — PDFs, text files, or Hugging Face dataset archives — instead
of relying on a language model's internal, possibly outdated or hallucinated
knowledge. Includes an interactive Gradio UI for uploading documents and asking
questions live.

🔗 **Live UI link:** generated fresh each time the notebook's last cell
(`demo.launch(share=True, ...)`) runs — copy the ` https://9cb3ed8455a1b79f49.gradio.live`
URL Gradio prints in the cell output and paste it here after running:
`<PASTE_GRADIO_LINK_HERE>`
*(Gradio share links are temporary and expire after the Colab session ends —
re-run the last cell to generate a new one.)*

**Dataset used for evaluation:** [`vectara/open_ragbench`](https://huggingface.co/datasets/vectara/open_ragbench)
(1,000 arXiv papers, BEIR format, with 3,045 human-written ground-truth questions)

---

## Table of Contents
- [Live UI Link](#-live-ui-link)
- [Overview](#overview)
- [Why RAG](#why-rag)
- [System Architecture](#system-architecture)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Using the Interactive UI](#using-the-interactive-ui)
- [Evaluation Results](#evaluation-results)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Key Learnings](#key-learnings)

---

## Overview

This project implements a full Retrieval-Augmented Generation pipeline end to
end: document ingestion → chunking → embedding → vector storage → retrieval →
grounded answer generation, plus two retrieval upgrades (hybrid search and
cross-encoder re-ranking) and a hallucination guardrail, all validated against
thousands of real questions with known correct answers.

**Objectives:**
- Understand and implement Retrieval-Augmented Generation from scratch
- Build a pipeline that combines retrieval and generation into one system
- Enable question answering over custom, private, or domain-specific documents
- Compare retrieval strategies (dense vs. hybrid vs. reranked) using real metrics, not guesses
- Provide an interactive interface non-technical users can use directly

---

## Why RAG

A plain language model can only answer from what it learned during training —
it has no access to your documents, and it will often *hallucinate* a confident
but wrong answer rather than admit it doesn't know. RAG fixes this by:

1. **Retrieval** — finding the most relevant passages from your own documents
   using semantic (and keyword) search
2. **Augmentation** — inserting those passages into the model's prompt as context
3. **Generation** — having the model answer using *only* that context, and
   explicitly say so when the answer isn't present

This makes answers traceable back to a real source document, and lets the
system work over private or specialized data the base model was never trained on.

---

## System Architecture

```
┌─────────────┐    ┌───────────┐    ┌────────────┐    ┌──────────────┐
│  Documents   │───▶│  Chunking  │───▶│ Embedding  │───▶│ FAISS Vector │
│ (PDF/TXT/HF) │    │  (800/150) │    │(MiniLM-L6) │    │    Index     │
└─────────────┘    └───────────┘    └────────────┘    └──────┬───────┘
                                                                │
┌─────────────┐    ┌───────────┐    ┌────────────┐            │
│  Question   │───▶│   Query    │───▶│  Hybrid +  │◀───────────┘
│   (User)    │    │ Embedding  │    │  Rerank    │
└─────────────┘    └───────────┘    └──────┬─────┘
                                             │
                                             ▼
                                  ┌────────────────────┐
                                  │  Relevance Guardrail │
                                  │  (blocks off-topic)  │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌────────────────────┐
                                  │  Qwen2.5-1.5B-Instruct│
                                  │  (grounded answer)   │
                                  └────────────────────┘
```

### Stage-by-stage breakdown

| # | Stage | What it does | Tool / Model |
|---|---|---|---|
| 1 | Document Ingestion | Loads raw text from PDF, `.txt`, or HF JSON into a unified `(text, title)` format | Custom `ingest_document()` |
| 2 | Chunking | Splits text into overlapping 800-char chunks so retrieval is focused, not document-wide | LangChain `RecursiveCharacterTextSplitter` |
| 3 | Embedding | Converts each chunk into a 384-dim vector capturing its meaning | `sentence-transformers/all-MiniLM-L6-v2` |
| 4 | Vector Storage | Indexes all chunk vectors for fast similarity search | FAISS `IndexFlatL2` |
| 5 | Query Embedding | Embeds the user's question with the same model as the chunks | `embed_query()` |
| 6 | Retrieval | Returns the top-k most similar chunks to the question | FAISS similarity search |
| 7 | Hybrid Search | Fuses vector similarity with BM25 keyword matching to catch exact-term matches | `rank_bm25` (BM25Okapi) |
| 8 | Re-ranking | Re-scores the hybrid candidate pool with a cross-encoder for higher precision | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 9 | Guardrail | Blocks generation if the best chunk's relevance score is below a calibrated threshold | Rerank-score gate |
| 10 | Generation | Produces the final answer using only the retrieved context | `Qwen/Qwen2.5-1.5B-Instruct` |
| 11 | UI | Upload documents and ask questions through a web interface | Gradio |

---

## Setup & Installation

The notebook installs its own dependencies inline via `!pip install`, so no
separate `requirements.txt` setup is required. Libraries used:

| Library | Purpose |
|---|---|
| `pypdf` | PDF text extraction |
| `huggingface_hub` | Downloading the evaluation dataset |
| `langchain-text-splitters` | Text chunking |
| `sentence-transformers` | Embedding model + cross-encoder reranker |
| `faiss-cpu` | Vector similarity search |
| `rank_bm25` | Keyword-based (BM25) retrieval |
| `transformers`, `accelerate`, `torch` | Loading and running the language model |
| `gradio` | Interactive web UI |

**Recommended environment:** Google Colab with a GPU runtime (T4 or better) —
the language model and cross-encoder run substantially faster on GPU, and
embedding ~150K chunks on CPU alone would take considerably longer.

---

## How to Run

1. Open `week7_Aaradhya_Jain_.ipynb` in Google Colab (or Jupyter with a GPU).
2. Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order, top to bottom (**Runtime → Run all**). Each `!pip
   install` cell only needs to run once per session.
4. The notebook will:
   - Download and ingest ~1,000 papers from `vectara/open_ragbench`
   - Chunk and embed the full corpus (~150K chunks)
   - Build the FAISS and BM25 indexes
   - Load the language model
   - Run through validation questions and print live examples
   - Run the full retrieval-strategy comparison and save `rag_metrics_report.md`
   - Launch the Gradio UI at the end

> ⏱️ **Note:** embedding the full corpus and running the 300-question strategy
> comparison are the most time-consuming steps (several minutes each on GPU).
> To test the pipeline faster, you can load a smaller subset of documents in
> Step 2b before running the rest.

---

## Using the Interactive UI

The final notebook cell launches a Gradio app with two tabs.

🔗 **Access it here:** ` https://9cb3ed8455a1b79f49.gradio.live` — active only while the
Colab notebook session is running.

**📤 Upload Documents**
Upload your own PDF or `.txt` files. They are ingested, chunked, embedded, and
added directly into the existing FAISS index and BM25 index — so they become
searchable immediately, alongside the ~1,000 papers already loaded, without
restarting the notebook.

**💬 Ask Questions**
Type a question and get a grounded answer, with:
- A toggle to enable/disable the hallucination guardrail
- A toggle to show which source documents the answer came from
- A few example questions to try immediately

`demo.launch(share=True, ...)` generates a public link so the app can be used
from a browser or phone without exposing the Colab notebook itself.

---

## Evaluation Results

Evaluated on 3,045 real, human-written questions from `open_ragbench` (each
with a ground-truth source paper), using a 300-question sample for the
strategy comparison:

| Strategy | Top-1 Accuracy | Top-5 Accuracy | Time (300 questions) |
|---|---|---|---|
| Dense only (FAISS) | 90.33% | 96.33% | 11.2s |
| Hybrid (BM25 + Vector) | 94.00% | 98.00% | 262.4s |
| Hybrid + Rerank | 92.33% | **98.67%** | 280.8s |

**Key finding:** hybrid search improved Top-1 accuracy by ~3.7 points over
dense-only retrieval, at roughly 23x the latency. Re-ranking pushed Top-5
accuracy to its best overall value but slightly reduced Top-1 versus hybrid
alone — the general-purpose cross-encoder occasionally favors a topically
similar chunk from the wrong paper over the exact ground-truth source, since
it wasn't fine-tuned on this arXiv domain.

**Concrete example:** for the question *"Why is it important for models to
generalize effectively across different editing styles?"*, dense-only search
failed completely (0/5 correct chunks — it only matched the abstract concept
of "generalization"). Hybrid search correctly found the target paper by
matching literal terms like "editing styles" that the embedding model missed.

**Guardrail calibration:** an in-domain question scored 7.32 on the reranker's
relevance scale (answer generated normally); a clearly unrelated question
("What's the capital of France?") scored -6.38 and was correctly blocked,
returning *"I cannot find this information in the provided documents"* instead
of a hallucinated answer.

The full report, including dataset statistics and embedding/vector-store
details, is regenerated on each run and saved to `rag_metrics_report.md`.

---

## Key Design Decisions

- **Chunk size (800 / 150 overlap):** balances retrieval precision (small
  enough that chunks stay topically focused) against context loss at chunk
  boundaries (overlap prevents key sentences from being split apart).
- **`IndexFlatL2` (exact search) over approximate indexes:** at ~150K vectors,
  exact search is still fast enough and guarantees the true nearest neighbors.
  An approximate index (IVF/HNSW) would be the right tradeoff at millions of
  vectors.
- **Hybrid retrieval before re-ranking:** BM25 and vector search are cheap and
  can run over the full corpus; the more expensive cross-encoder only needs to
  re-score a small candidate pool (20 chunks), not the entire index.
- **Guardrail based on rerank score, not a fixed keyword filter:** calibrating
  the relevance threshold against real in-domain vs. unrelated question scores
  makes the cutoff evidence-based rather than an arbitrary guess.

---

## Known Limitations

- **BM25 rebuild cost:** the keyword index isn't incremental — adding new
  documents through the UI requires re-tokenizing the entire corpus, which
  gets slower as the corpus grows.
- **Off-the-shelf re-ranker:** `ms-marco-MiniLM-L-6-v2` is trained on general
  web passages, not scientific papers, so it isn't a strict precision upgrade
  over hybrid search alone on this dataset.
- **Single global guardrail threshold:** reliably blocks clearly irrelevant
  questions, but doesn't catch subtler cases where a moderately-high score
  still points to the wrong document.
- **Hybrid search latency:** ~23x slower than dense-only in the current
  unoptimized implementation, since BM25 scoring is recomputed per query.

---

## Future Improvements

- Compare multiple chunk size/overlap configurations against retrieval accuracy
- Try alternative or larger embedding models (e.g. `bge-small-en-v1.5`)
- Fine-tune or swap the cross-encoder for a domain-specific reranker
- Cache or precompute BM25 scoring to reduce hybrid search latency
- Move from `IndexFlatL2` to an approximate FAISS index for larger corpora
- Add per-chunk or per-document deletion/reset controls to the UI

---

## Key Learnings

- Dense retrieval alone is fast and strong (90%+ Top-1) but can fail on
  questions that hinge on specific terminology general-purpose embeddings
  don't capture well.
- Hybrid search meaningfully closes that gap, especially for keyword-heavy
  questions, at a real, measurable latency cost.
- Re-ranking is not automatically an upgrade — its value depends on whether
  the reranker matches the target domain.
- A single global relevance threshold is a reasonable, evidence-calibrated
  guardrail against hallucination, but has known blind spots worth documenting
  rather than hiding.
- Validating against real, human-labeled questions (rather than a handful of
  hand-picked examples) is what turns "this seems to work" into an actual,
  defensible measurement.
