# 🚀 Asteroid Diameter Prediction & NEO Hazard Classification - aj

## 📖 Project Overview

This contains a comprehensive workflow for predicting asteroid diameter and classifying hazardous Near-Earth Objects (NEOs) using machine learning models.

This project provides a complete machine learning workflow for:

- Predicting asteroid diameters from physical characteristics.
- Classifying Near-Earth Objects (NEOs) as potentially hazardous or not.

Several modeling strategies are explored to boost regression accuracy and classification reliability using real-world space data.

---

## 📂 Project Notebooks

1. **Preprocessing and Cleaning** (`asteroid_and_neo_preprocessing_pipeline.ipynb`)  
   Data cleaning, feature engineering, and preparation for modeling.

2. **Base Model** (`1_base_model.ipynb`)  
   Regression model using both **albedo** and **diameter** features.

3. **Simple Model** (`2_simple_model.ipynb`)  
   Regression model using only **diameter** (without albedo).

4. **Hybrid Model** (`3_hybrid_model.ipynb`)  
   Two-step model:  
   - First, predict albedo  
   - Then, use predicted albedo to predict diameter.

5. **NEO Classification Model** (`4_neo_classification.ipynb`)  
   Classification model to identify hazardous Near-Earth Objects.

---

## 📊 Datasets Used

This project uses three main datasets:

- **Asteroid Dataset 1**  
  Source: [Prediction of Asteroid Diameter by basu369victor](https://www.kaggle.com/datasets/basu369victor/prediction-of-asteroid-diameter)  
  Loaded as: `asteroid_1 = pd.read_csv('data/asteroid_1.csv')`

- **Asteroid Dataset 2**  
  Source: [Asteroid Dataset by sakhawat18](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset/data)  
  Loaded as: `asteroid_2 = pd.read_csv('data/asteroid_2.csv')`

- **NASA Near-Earth Objects (NEO) Dataset**  
  Source: [NASA Nearest Earth Objects by sameepvani](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects?select=neo.csv)  
  Loaded as: `df_neo = pd.read_csv('data/neo.csv')`

---

## 🧰 Requirements

To run this project, install the required Python packages listed in `requirements.txt`:

---

## 🚀 How to Use
- Begin with data preprocessing in asteroid_and_neo_preprocessing_pipeline.ipynb to clean and prepare your data.
- Explore the base regression model in 1_base_model.ipynb that uses albedo and diameter.
- Compare results with the simple model in 2_simple_model.ipynb which excludes albedo.
- Run the hybrid model in 3_hybrid_model.ipynb, which predicts albedo first then diameter.
- Use 4_neo_classification.ipynb to train and evaluate a classification model that flags hazardous NEOs.

--

## ⚙️ Tools & Libraries
- pandas, numpy — Data manipulation
- scikit-learn — Machine learning models and evaluation
- xgboost, lightgbm, catboost — Gradient boosting models for regression and classification
- imblearn — Handling imbalanced datasets (e.g., SMOTE)
- matplotlib, seaborn — Visualization
- shap — Model explainability and interpretation
- jupyter — Notebook environment

--

## 📚 References & Resources
- NASA Near-Earth Object Web Service (NEOWS)
- Kaggle Asteroid Diameter Dataset (basu369victor)
- Kaggle Asteroid Dataset (sakhawat18)
- Kaggle NASA NEO Dataset (sameepvani)
- Scikit-learn Documentation
- XGBoost Documentation
- SHAP Documentation


## Steps
### Data loading
These datasets are downloaded from Kaggle.

### Data Cleaning
- Checked for duplicate rows
- Dropped features with excessive missing data
- Dropped irrelevent features
- Converted categorical features
- Removed prefix from features
- One-Hot Encode

### Filled Missing values
- with "0"
- with Median
- with Mode
- by predicting using a regression model

### Feature Engineering
New features were created for easy interpretation.
Few existing features were categorized.

### Data Visualization
Various visuals are created:
- Bar Plot
- Count Plot
- Boxplot
- Histogram Plot
- Correlation Matrix
- ROC Curve
- Precision-Recall Curve
- Calibration Plot

### Machine Learning
Two machine learning model were created, fitted and evaluated:
- Linear regression
- Random Forest Regression
- XGBoost Regression
- LightGBM
- Catboost Regression
- Quantile refression
- Random Forest Classifier
- XGBClassifier
- LGBMClassifier
- CatBoostClassifier