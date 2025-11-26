# Predicting Survival After Donor Transplants (HCT)

## Project Overview
This project focuses on predicting survival outcomes for patients undergoing donor hematopoietic cell transplantation (HCT). Accurate survival predictions are crucial for guiding treatment decisions and improving patient outcomes. The project emphasizes both predictive accuracy and fairness, especially across patients of diverse racial and socioeconomic backgrounds.

We use synthetic data mimicking real-world patient characteristics, allowing model development without compromising privacy.

---

## Project Roadmap & Techniques

### 1. Data Exploration (EDA)
**Purpose:** Understand the dataset, distributions, missing values, and relationships between features.  

**Key Techniques:**
- Summary statistics (`describe()`) for numerical features  
- Value counts and unique analysis for categorical features  
- Visualizations like histograms, boxplots, and correlation heatmaps  

**Why:** Helps identify data issues, trends, and potential predictive variables.

---

### 2. Feature Engineering
**Purpose:** Enhance raw data to improve model learning.  

**Techniques Used:**
- Handle missing values (`fillna`, custom replacements)  
- Correct rare or outlier values  
- Create new features combining clinical data (e.g., `donor_age - age_at_hct`, `comorbidity_score * karnofsky_score`)  

**Why:** Creates more informative features, reduces noise, and captures complex relationships between variables.

---

### 3. Categorical Feature Embedding
**Purpose:** Convert categorical variables into numerical vectors that preserve semantic relationships.  

**Technique Used:** Word2Vec embeddings on categorical columns.  

**Why:** Traditional one-hot encoding can result in sparse data. Embeddings provide dense vector representations that improve model learning for high-cardinality categories.

---

### 4. Label Transformation
**Purpose:** Transform survival outcomes into continuous risk scores suitable for regression models.  

**Technique Used:** Nelson-Aalen estimator to compute cumulative hazard.  

**Why:** Converts censored survival data into a continuous target, allowing machine learning models to predict risk scores directly.

---

### 5. Data Preprocessing
**Steps:**
- Split into train/validation sets (80/20)  
- Remove duplicate columns  
- Align train and test sets to have identical columns  
- Standardize feature names (remove special characters)  

**Why:** Ensures data consistency, avoids training errors, and allows seamless model evaluation.

---

### 6. Model Selection & Training
**Purpose:** Predict continuous risk scores from processed features.  

**Models Used:**
- LightGBM – Gradient boosting with efficient tree-based learning  
- XGBoost – Gradient boosting optimized for structured/tabular data  
- CatBoost – Gradient boosting that handles categorical features natively  

**Why Multiple Models:** Combining models often improves prediction stability and reduces overfitting.

---

### 7. Model Evaluation
**Metrics Used:**
- RMSE (Root Mean Squared Error) – Measures average prediction error  
- MSE (Mean Squared Error) – Penalizes larger errors more heavily  

**Why:** Provides a clear measure of prediction accuracy on training and validation sets.

**Model Performance Table:**

| Model                 | Train RMSE | Val RMSE | Train MSE | Val MSE |
|-----------------------|------------|----------|-----------|---------|
| LGBM                  | 0.1572     | 0.2790   | 0.0247    | 0.0778  |
| XGB                   | 0.2453     | 0.2765   | 0.0602    | 0.0764  |
| CatBoost              | 0.2495     | 0.2765   | 0.0623    | 0.0764  |
| Ensemble (Averaging)  | 0.2152     | 0.2757   | 0.0463    | 0.0760  |

---

### 8. Ensemble Learning
**Purpose:** Combine predictions from multiple models to improve performance.  

**Technique:** Simple averaging of LGBM, XGBoost, and CatBoost predictions  

**Why:** Reduces model variance, stabilizes predictions, and leverages strengths of individual models.

---

## Evaluation Metric

**Stratified Concordance Index (C-index):**
- Measures how well predicted risk scores rank actual survival times  
- Stratified across racial groups to ensure fairness  
- Score range: 0–1, where higher is better  

**Why:** Encourages models that are accurate and equitable, reducing bias across patient groups.

---

## Key Features
- **Patient demographics:** Age, sex, race  
- **Clinical variables:** DRI score, cytogenetics, comorbidities  
- **Transplant variables:** HLA matching (high/low resolution), graft type, TBI, melphalan dose  
- **Health conditions:** Cardiac, pulmonary, diabetes, prior tumors

