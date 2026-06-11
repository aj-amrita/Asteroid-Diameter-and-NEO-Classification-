# 🚀 Asteroid Diameter Prediction & NEO Hazard Classification

Final Thesis Project Showcase — Data Scientist: Amrita Jattan

## 📖 Project Overview
This repository delivers an end-to-end machine learning workflow utilizing multi-source NASA and Kaggle space data to solve two distinct operational challenges:
1. **Asteroid Diameter Prediction (Regression):** Engineering a robust pipeline to bypass massive missing data bottlenecks, utilizing a novel **Hybrid Two-Step Architecture** alongside Base and Simple comparative baselines.
2. **Near-Earth Object (NEO) Hazard Classification (Classification):** Training supervised learning models to accurately isolate rare, potentially hazardous objects from highly skewed distributions.

---

## 📊 Key Results & Performance Summary

### 1. Asteroid Diameter Prediction Baselines
* **The Missing Data Bottleneck:** Over **85% of real-world asteroid records** are missing critical physical characteristics like `albedo` (reflectivity) and baseline `diameter` values.
* **The Solution:** While the **Base Model** provides an ideal benchmark (trained on records where albedo is present), the **Hybrid Model** presents a deployment-ready architectural breakthrough. It uses a tuned two-step model pipeline where **Model A predicts the missing albedo value**, and **Model B utilizes that prediction to forecast the final diameter**.

#### Tuned Regression Model Performance Comparison (CatBoost Regression):
| Model Pipeline Strategy | Mean Squared Error (MSE) | Coefficient of Determination ($R^2$) | Operational Context |
| :--- | :---: | :---: | :--- |
| **Base Model (Tuned)** | **0.5115** | **0.9815** | *Ideal Benchmark (Albedo completely known)* |
| **Simple Model (Tuned)** | **2.9420** | **0.8968** | *Constrained Baseline (Albedo completely excluded)* |
| **Hybrid Model (Tuned)** | **1.8389** | **0.9355** | 🏆 **Production-Ready Strategy (Predicts missing albedo first)** |

*Quantile Regression analysis verified prediction interval stability ($R^2 = 0.9325$ at $\alpha = 0.5$ median).*

### 2. Model Explainability & Feature Importance (SHAP Analysis)
* **Diameter Prediction Drivers:** SHAP and gradient-boosting internal feature importances revealed that Absolute Magnitude (`H`) is the single strongest predictor of diameter across the models (contributing **63.59%–64.07%** to model decisions), followed heavily by `albedo` (predicted or actual) at **22.42%–30.78%**. 
* **Albedo Predictors:** When isolating features to handle missing values, orbital inclination (`i`), semi-major axis (`a`), and mean motion (`n`) were identified as the leading drivers.

### 3. NEO Hazard Classification
* **Handling Imbalanced Classes:** The hazardous class constitutes an extreme minority (**only 7.6% of total observations**; 2,071 hazardous vs. 25,027 non-hazardous objects).
* **Strategy:** Implemented **SMOTE (Synthetic Minority Over-sampling Technique)** coupled with a tuned **CatBoost Classifier** to maximize minority-class recall without eroding overall prediction precision.

---

## 🛠️ Tools & Libraries Used
* **Data Engineering & Optimization:** `pandas`, `NumPy`, `autoML`, `RandomizedSearchCV`
* **Machine Learning Frameworks:** `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`
* **Imbalance & Interpretability:** `imblearn` (SMOTE), `SHAP`
* **Visualization Engine:** `matplotlib`, `seaborn`

---

## ⚙️ Repository Pipeline & Notebook Structure

To execute the modeling pipeline, run the notebooks sequentially:

1. **`asteroid_and_neo_preprocessing_pipeline.ipynb`**
   Handles initial multi-source data merging, drop-criteria for features missing extreme volume, categorical transformation, and feature engineering.
2. **`1_base_model.ipynb`**
   Establishes optimal upper-bound performance metrics utilizing true albedo features.
3. **`2_simple_model.ipynb`**
   Measures structural predictive degradation when dropping missing physical constants entirely.
4. **`3_hybrid_model.ipynb`**
   Constructs the full multi-stage regression framework (Step 1: Predict Albedo via CatBoost ➡️ Step 2: Feed predicted features into Diameter Regressor).
5. **`4_neo_classification.ipynb`**
   Deploys ensemble classifiers optimized via SMOTE to map hazard vulnerabilities using NASA datasets.

---

## 📚 Data Dimensions & References
* **Asteroid Dataset 1:** 839,000+ entries & 27 features (basu369victor/Kaggle)
* **Asteroid Dataset 2:** 958,000+ entries & 45 features (sakhawat18/Kaggle)
* **NASA Near-Earth Objects Dataset:** 90,800+ entries & 10 features (sameepvani/Kaggle)
