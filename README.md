# 🚀 Asteroid Diameter Prediction & NEO Hazard Classification

Final Thesis Project Showcase — Data Scientist: Amrita Jattan  
*Leveraging Machine Learning to Estimate Asteroid Dimensions and Detect Planetary Threats.*

---

## 📖 Project Overview
This repository delivers an end-to-end machine learning workflow utilizing multi-source NASA and Kaggle space data to solve two distinct operational aerospace challenges:
1. **Asteroid Diameter Prediction (Regression):** Engineering a robust pipeline to bypass massive missing data bottlenecks. This features a comparative study across a **Base Model**, a **Simple Model**, and a **Hybrid Two-Stage Machine Learning Pipeline**.
2. **Near-Earth Object (NEO) Hazard Classification (Classification):** Training supervised ensemble learning models to accurately isolate rare, potentially hazardous objects from highly skewed celestial distributions.

---

## 🌟 Key Highlights

✅ Built an end-to-end machine learning pipeline using over 1.8 million asteroid records

✅ Developed a hybrid two-stage modeling approach to address 85% missing albedo values

✅ Achieved R² = 0.9815 with a tuned CatBoost Regressor

✅ Applied SMOTE and ensemble learning techniques for hazardous NEO classification

✅ Used SHAP to explain model decisions and identify key predictive features

---

## 📂 Repository Structure

├── data/
│   ├── asteroid_1.csv
│   ├── asteroid_2.csv
│   └── neo.csv
│
├── notebooks/
│   ├── asteroid_and_neo_preprocessing_pipeline.ipynb
│   ├── 1_base_model.ipynb
│   ├── 2_simple_model.ipynb
│   ├── 3_hybrid_model.ipynb
│   └── 4_neo_classification.ipynb
│
├── requirements.txt
└── README.md

---

## 🛠️ Tools & Libraries Used
* **Data Engineering & Optimization:** `pandas`, `NumPy`, `autoML`, `RandomizedSearchCV`
* **Machine Learning Frameworks:** `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`
* **Imbalance & Interpretability:** `imblearn` (SMOTE), `SHAP` (Shapley Additive exPlanations)
* **Visualization Engine:** `matplotlib`, `seaborn`

---

## ⚙️ Repository Pipeline & Notebook Structure

To execute the modeling pipeline, run the Jupyter notebooks sequentially:

1. **`asteroid_and_neo_preprocessing_pipeline.ipynb`** Handles multi-source data merging, drop-criteria for features missing extreme volume, categorical transformation, and initial feature engineering.
2. **`1_base_model.ipynb`** Establishes optimal upper-bound performance metrics utilizing true `albedo` features.
3. **`2_simple_model.ipynb`** Measures structural predictive degradation when dropping missing physical constants entirely.
4. **`3_hybrid_model.ipynb`** Constructs the full multi-stage regression framework (**Step 1:** Predict Albedo via CatBoost ➡️ **Step 2:** Feed predicted features into a secondary Diameter Regressor).
5. **`4_neo_classification.ipynb`** Deploys ensemble classifiers optimized via SMOTE to map hazard vulnerabilities using NASA datasets.

---

## 📊 Comprehensive Results & Performance Analysis

### 1. Asteroid Diameter Prediction (Regression Frameworks)

#### A. Base Model Comparison (Albedo Completely Known)
This benchmark outlines the upper limits of predictability when physical constants are fully available.

| Algorithm | Mean Squared Error (MSE) | Coefficient of Determination ($R^2$) |
| :--- | :---: | :---: |
| Linear Regression | 2.6441 | 0.9045 |
| Random Forest | 0.5663 | 0.9795 |
| XGBoost | 0.8928 | 0.9678 |
| LightGBM | 0.5795 | 0.9791 |
| CatBoost (Untuned) | 0.5228 | 0.9811 |
| **🏆 CatBoost (Tuned via RandomizedSearchCV)** | **0.5115** | **0.9815** |

* **Quantile Regression Stability (Tuned CatBoost):** * $\alpha = 0.1 \rightarrow R^2: 0.9576$  
  * $\alpha = 0.5 \rightarrow R^2: 0.9806$  
  * $\alpha = 0.9 \rightarrow R^2: 0.9567$

#### B. Simple Model Comparison (Albedo Excluded)
Dropping albedo creates an extreme information gap, leading to a visible drop in evaluation metrics across all models.

| Algorithm | Mean Squared Error (MSE) | Coefficient of Determination ($R^2$) |
| :--- | :---: | :---: |
| Linear Regression | 8.5767 | 0.6991 |
| Random Forest | 3.1878 | 0.8882 |
| XGBoost | 3.3062 | 0.8840 |
| LightGBM | 3.1132 | 0.8908 |
| CatBoost (Untuned) | 3.0271 | 0.8938 |
| **🏆 CatBoost (Tuned via RandomizedSearchCV)** | **2.9420** | **0.8968** |

* **Quantile Regression Stability (Simple CatBoost):** * $\alpha = 0.1 \rightarrow R^2: 0.7793$  
  * $\alpha = 0.5 \rightarrow R^2: 0.8904$  
  * $\alpha = 0.9 \rightarrow R^2: 0.7737$

#### C. Production Hybrid Architecture (Two-Step Pipeline)
Over **85% of real-world asteroid records** lack reflectivity data. While the Simple model suffers heavily from this data loss, this **Hybrid Pipeline** successfully bridges the gap by predicting the missing value first.

* **Step 1: Albedo Prediction Performance (Model A)** * **🏆 Tuned CatBoost** $\rightarrow$ **MSE: 0.0032** | **$R^2$: 0.5846** * *(Baseline Benchmarks: Random Forest $R^2$: 0.5757 | XGBoost $R^2$: 0.5598 | Linear Regression $R^2$: 0.3592)*

* **Step 2: Ultimate Diameter Prediction Performance (Model B)** Using the generated albedo feature from Step 1 alongside standard orbital metrics to calculate ultimate diameter size.

| Algorithm | Mean Squared Error (MSE) | Coefficient of Determination ($R^2$) |
| :--- | :---: | :---: |
| Linear Regression | 2.9188 | 0.8976 |
| Random Forest | 1.9787 | 0.9306 |
| XGBoost | 2.3291 | 0.9183 |
| LightGBM | 1.9338 | 0.9322 |
| CatBoost (Untuned) | 1.8798 | 0.9340 |
| **🏆 CatBoost (Tuned via RandomizedSearchCV)** | **1.8389** | **0.9355** |

* **Quantile Regression Stability (Hybrid Architecture):** * $\alpha = 0.1 \rightarrow R^2: 0.8515$  
  * $\alpha = 0.5 \rightarrow R^2: 0.9325$  
  * $\alpha = 0.9 \rightarrow R^2: 0.8416$

---

## 🎯 Key Takeaways

- Missing albedo values were the primary limitation affecting diameter prediction.
- Removing albedo reduced CatBoost performance from R² = 0.9815 to R² = 0.8968.
- The Hybrid Two-Stage Pipeline recovered much of the lost predictive power, achieving R² = 0.9355.
- SHAP analysis confirmed that Absolute Magnitude and Albedo were the most influential predictors.
- Ensemble boosting methods consistently outperformed traditional linear models.

---

### 2. Model Explainability & Feature Importances (SHAP Insights)

Global feature analysis using SHAP and gradient-boosting internal metrics revealed how the feature reliance shifts depending on the modeling pipeline context:

* **Base Model (Albedo Known):** Absolute Magnitude (`H`) heavily dominates the model's decision-making process at **64.07%**, followed closely by true `albedo` at **30.78%**.
* **Simple Model (Albedo Excluded):** Deprived of reflectivity variables, the predictive workload is distributed across orbital features. Absolute Magnitude (`H`) drops to **41.27%**, forcing Semi-major axis (`a`: **11.51%**), Mean Motion (`n`: **10.85%**), and Inclination (`i`: **7.07%**) to take on significantly more statistical weight.
* **Hybrid Model Step 1 (Reconstructing Albedo):** When rebuilding the missing reflectivity matrix from scratch, orbital properties become highly predictive. The top drivers include Inclination (`i`: **11.73%**), Semi-major axis (`a`: **11.44%**), Mean Motion (`n`: **11.35%**), and Absolute Magnitude (`H`: **8.91%**).
* **Hybrid Model Step 2 (Final Diameter Calculation):** Relying on the predicted albedo attribute, the final decision vector successfully returns to its optimal geometric focus: Absolute Magnitude (`H`) commands **63.59%** of the weight, and the generated `albedo_predicted` feature accounts for **22.42%**.

---

### 3. Near-Earth Object (NEO) Hazard Classification

* **The Class Imbalance Problem:** Hazardous near-earth items represent an extreme minority within the active aerospace data grid, comprising **only 7.6% of the overall dataset** (2,071 hazardous objects vs. 25,027 non-hazardous objects).
* **Mitigation Strategy:** Implemented **SMOTE (Synthetic Minority Over-sampling Technique)** alongside an automated hyperparameter-tuned **CatBoost Classifier** to expand the minority class boundary, heavily optimizing minority-class Recall without causing an unacceptable drop in overall Precision metrics.

---

## 📚 Dataset Dimensions & Data Sources
* **Asteroid Dataset 1:** 839,000+ entries & 27 features | [Prediction of Asteroid Diameter (basu369victor)](https://www.kaggle.com/datasets/basu369victor/prediction-of-asteroid-diameter)
* **Asteroid Dataset 2:** 958,000+ entries & 45 features | [Asteroid Dataset (sakhawat18)](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset/data)
* **NASA Near-Earth Objects Dataset:** 90,800+ entries & 10 features | [NASA Nearest Earth Objects (sameepvani)](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects?select=neo.csv)
