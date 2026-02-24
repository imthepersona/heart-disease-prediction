# Heart Disease Prediction – AGE Analytics

A machine learning proof-of-concept for early heart disease detection using routine clinical measurements.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Installation & Usage](#installation--usage)
- [Limitations & Future Work](#limitations--future-work)
- [Contact](#contact)
- [License](#license)

---

## Overview

This project develops a classification model to predict heart disease presence from routine patient vitals and test results. It is an educational proof-of-concept covering the full ML workflow from exploratory data analysis through model training and evaluation.

**Business goal:** enable earlier detection of at-risk patients so they can be prioritized for follow-up clinical evaluation.

### Project Components

The analysis is structured across three Jupyter notebooks:

- PT1 – Exploratory Data Analysis & Cleaning
- PT2 – Feature Engineering & Preprocessing
- PT3 – Model Training & Evaluation

---

## Key Results

The selected Logistic Regression model achieved the following metrics:

- Validation: recall 0.889, precision 0.889, F1-score 0.889, ROC-AUC 0.934, accuracy 0.900.
- Test: recall 1.000, precision 0.857, F1-score 0.923, ROC-AUC 0.993, accuracy 0.927.

Key points:

- Perfect recall on the test set – the model identified all heart disease cases in the held-out data.
- Exceeds initial targets – recall at least 0.80 and ROC-AUC between 0.75 and 0.85.
- Simpler and more interpretable than the Random Forest alternative.

---

## Dataset

- Source: Kaggle Heart Disease Prediction Dataset.
- Size: 270 patient records and 14 features.
- License: CC0 (Public Domain).

### Target variable

- `heart_disease_flag` (binary): 1 = presence of heart disease, 0 = absence.

### Features

Numeric features:

- age – patient age in years
- bp – resting blood pressure (mm Hg)
- cholesterol – serum cholesterol (mg/dl)
- max_hr – maximum heart rate achieved
- st_depression – ST depression induced by exercise

Categorical features:

- sex – 0 = female, 1 = male
- chest_pain_type – 1-4 (typical angina to asymptomatic)
- fbs_over_120 – fasting blood sugar > 120 mg/dl (0 = no, 1 = yes)
- ekg_results – 0-2 (normal to left ventricular hypertrophy)
- exercise_angina – 0 = no, 1 = yes
- slope_of_st – 1-3 (upsloping to downsloping)
- number_of_vessels_fluro – 0-3 major vessels narrowed
- thallium – 3 = normal, 6 = fixed defect, 7 = reversible defect

---

## Project Structure

Project folders:

- data/raw – original Kaggle CSV.
- data/interim – cleaned data from PT1 (df_heart_cleaned_v1.csv).
- data/processed – final processed data for modeling (df_heart_v2ohe.csv).
- notebooks – PT1, PT2, PT3 notebooks.
- models – saved preprocessing pipeline and trained models.
- reports – PACE document and executive summary.

---

## Methodology

### 1️⃣ PT1 – Exploratory Data Analysis & Cleaning

Objectives:

- Inspect distributions, missingness, and outliers.
- Understand relationships between features and `heart_disease_flag`.
- Produce a clean dataset ready for modeling.

Key activities:

- Data quality audit using info, describe, and missing value checks.
- Univariate plots (histograms, boxplots) for numeric and categorical features.
- Bivariate plots (chest pain, thallium, ST slope versus the target).
- Correlation analysis (age vs max_hr shows a moderate negative correlation; st_depression vs max_hr also shows a modest negative correlation).

Output:

- Cleaned dataset saved to data/interim/df_heart_cleaned_v1.csv.

### 2️⃣ PT2 – Feature Engineering & Preprocessing

Objectives:

- Encode the target as a numeric flag.
- Define feature groups (categorical vs numeric).
- Build a reusable preprocessing pipeline.

Main steps:

- Target engineering: map heart_disease to heart_disease_flag (Absence to 0, Presence to 1).
- Create train, validation, and test splits stratified by the target: 70% train, 15% validation, 15% test, with a fixed random state for reproducibility.
- Preprocessing pipeline: numeric features imputed with median and scaled; categorical features imputed with most-frequent value and one-hot encoded; combined with a column transformer and fit on the training data.

Outputs:

- Processed dataset saved to data/processed/df_heart_v2ohe.csv.
- Fitted preprocessing pipeline saved to models/heart_preprocessor.joblib.

### 3️⃣ PT3 – Model Training & Evaluation

Models compared:

- Logistic Regression (linear model with regularization).
- Random Forest (ensemble of decision trees with 200 estimators).

Both models were wrapped in the same pipeline with the shared preprocessing step.

Threshold tuning:

- Default decision threshold was 0.50 on the predicted probability of class 1.
- Thresholds from 0.10 to 0.55 were evaluated on the validation set.
- A threshold of 0.45 was chosen as a good operating point, slightly trading precision for higher recall and F1-score.
- This reflects the screening use case, where missing a true disease case is more harmful than producing an extra false positive.

Model selection:

- Logistic Regression generalized better to the test set across accuracy, recall, F1-score, and ROC-AUC.
- It is also easier to interpret for clinical stakeholders, so it was chosen as the primary model.

---

## Model Performance

Final test set metrics for the selected Logistic Regression model (threshold 0.50):

- Accuracy: 0.927
- Precision: 0.857
- Recall: 1.000
- F1-score: 0.923
- ROC-AUC: 0.993

Confusion matrix (described):

- Most healthy patients were correctly classified as healthy.
- A small number of healthy patients were incorrectly flagged as having heart disease (false positives).
- No true disease cases were missed (zero false negatives).
- All patients with heart disease in the test set were correctly identified.

Model comparison on the test set:

- Logistic Regression: Accuracy 0.927, Precision 0.857, Recall 1.000, F1-score 0.923, ROC-AUC 0.993.
- Random Forest: Accuracy 0.854, Precision 0.773, Recall 0.944, F1-score 0.850, ROC-AUC 0.960.

Logistic Regression outperformed Random Forest across all key metrics and was therefore selected as the primary model.

---

## Installation & Usage

Prerequisites:

- Python 3.8 or higher
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

Setup steps:

1. Clone the repository and move into the project directory.
2. Install required packages using pip install -r requirements.txt.
3. Run the notebooks in order: PT1_EDA_Cleaning, PT2_Feature_Engineering, PT3_Modeling.

Making predictions (conceptual):

- Load the trained pipeline from models/heart_logreg_pipeline.joblib.
- Create a one-row pandas DataFrame with all the required features (age, sex, bp, cholesterol, max_hr, st_depression, and each categorical feature).
- Call the pipeline predict method to obtain the predicted class (heart disease present or absent).
- Call predict_proba to obtain the probability of heart disease for that patient.

---

## Limitations & Future Work

Current limitations:

- Small dataset (270 patients); performance may not generalize to broader, more diverse populations.
- Uses a public Kaggle dataset rather than real clinical electronic health record data.
- Built as an educational proof-of-concept, not as a production or clinically approved tool.
- Fairness across demographic groups (such as age and sex) has not yet been evaluated.
- Limited feature set; no longitudinal information, lab trends, or richer clinical history.

Planned and recommended next steps:

- Validate the model on larger, real-world clinical datasets.
- Assess performance and fairness across age, sex, and other demographic subgroups.
- Work with clinicians to refine feature engineering and interpretation of model outputs.
- Incorporate temporal and additional clinical data to improve predictive power.
- Build a deployment-ready API and add monitoring plus explainability tooling for production use.

---

## Contact

Willber Escalante
AGE Analytics
Email: reasonwithwill@gmail.com

---

## License

This project uses the Kaggle Heart Disease Prediction Dataset under the CC0 (Public Domain) license.

# Heart Disease Prediction – AGE Analytics

A machine learning proof-of-concept for early heart disease detection using routine clinical measurements.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Installation & Usage](#installation--usage)
- [Limitations & Future Work](#limitations--future-work)
- [Contact](#contact)
- [License](#license)

---

## Overview

This project develops a classification model to predict heart disease presence from routine patient vitals and test results. It is an educational proof-of-concept covering the full ML workflow from exploratory data analysis through model training and evaluation.

**Business goal:** enable earlier detection of at-risk patients so they can be prioritized for follow-up clinical evaluation.

### Project Components

The analysis is structured across three Jupyter notebooks:

- PT1 – Exploratory Data Analysis & Cleaning  
- PT2 – Feature Engineering & Preprocessing  
- PT3 – Model Training & Evaluation  

---

## Key Results

The selected Logistic Regression model achieved the following metrics:

- Validation: recall 0.889, precision 0.889, F1-score 0.889, ROC-AUC 0.934, accuracy 0.900.  
- Test: recall 1.000, precision 0.857, F1-score 0.923, ROC-AUC 0.993, accuracy 0.927.  

Key points:

- Perfect recall on the test set – the model identified all heart disease cases in the held-out data.  
- Exceeds initial targets – recall at least 0.80 and ROC-AUC between 0.75 and 0.85.  
- Simpler and more interpretable than the Random Forest alternative.

---

## Dataset

- Source: Kaggle Heart Disease Prediction Dataset.  
- Size: 270 patient records and 14 features.  
- License: CC0 (Public Domain).  

### Target variable

- `heart_disease_flag` (binary): 1 = presence of heart disease, 0 = absence.

### Features

Numeric features:

- age – patient age in years  
- bp – resting blood pressure (mm Hg)  
- cholesterol – serum cholesterol (mg/dl)  
- max_hr – maximum heart rate achieved  
- st_depression – ST depression induced by exercise  

Categorical features:

- sex – 0 = female, 1 = male  
- chest_pain_type – 1–4 (typical angina to asymptomatic)  
- fbs_over_120 – fasting blood sugar > 120 mg/dl (0 = no, 1 = yes)  
- ekg_results – 0–2 (normal to left ventricular hypertrophy)  
- exercise_angina – 0 = no, 1 = yes  
- slope_of_st – 1–3 (upsloping to downsloping)  
- number_of_vessels_fluro – 0–3 major vessels narrowed  
- thallium – 3 = normal, 6 = fixed defect, 7 = reversible defect  

---

## Project Structure

Project folders (conceptual):

- `data/raw` – original Kaggle CSV.  
- `data/interim` – cleaned data from PT1 (for example, `df_heart_cleaned_v1.csv`).  
- `data/processed` – final processed data for modeling (for example, `df_heart_v2ohe.csv`).  
- `notebooks` – PT1, PT2, PT3 notebooks.  
- `models` – saved preprocessing pipeline and trained models.  
- `reports` – PACE document and executive summary.  

---

## Methodology

### 1️⃣ PT1 – Exploratory Data Analysis & Cleaning

Objectives:

- Inspect distributions, missingness, and outliers.  
- Understand relationships between features and `heart_disease_flag`.  
- Produce a clean dataset ready for modeling.  

Key activities:

- Data quality audit using info, describe, and missing value checks.  
- Univariate plots (histograms, boxplots) for numeric and categorical features.  
- Bivariate plots (for example, chest pain, thallium, ST slope versus the target).  
- Correlation analysis (age versus `max_hr` shows a moderate negative correlation; `st_depression` versus `max_hr` also shows a modest negative correlation).  

Output:

- Cleaned dataset saved to `data/interim/df_heart_cleaned_v1.csv`.

### 2️⃣ PT2 – Feature Engineering & Preprocessing

Objectives:

- Encode the target as a numeric flag.  
- Define feature groups (categorical versus numeric).  
- Build a reusable preprocessing pipeline.  

Main steps:

- Target engineering: map the text label `heart_disease` to `heart_disease_flag` (Absence to 0, Presence to 1).  
- Create train, validation, and test splits stratified by the target: 70% train, 15% validation, 15% test, with a fixed random state for reproducibility.  
- Preprocessing pipeline:
  - Numeric features: impute missing values using the median and scale them.  
  - Categorical features: impute missing values using the most frequent value and one‑hot encode categories.  
  - Combine numeric and categorical pipelines using a column transformer and fit it on the training data.  

Outputs:

- Processed dataset saved to `data/processed/df_heart_v2ohe.csv`.  
- Fitted preprocessing pipeline saved to `models/heart_preprocessor.joblib`.

### 3️⃣ PT3 – Model Training & Evaluation

Models compared:

- Logistic Regression (linear model with regularization).  
- Random Forest (ensemble of decision trees with 200 estimators).  

Both models were wrapped in the same pipeline with the shared preprocessing step.

Threshold tuning:

- Default decision threshold was 0.50 on the predicted probability of class 1.  
- Thresholds from 0.10 to 0.55 were evaluated on the validation set.  
- A threshold of 0.45 was chosen as a good operating point, slightly trading precision for higher recall and F1-score.  
- This reflects the screening use case, where missing a true disease case (false negative) is more harmful than producing an extra false positive.

Model selection:

- Logistic Regression generalized better to the test set across accuracy, recall, F1-score, and ROC-AUC.  
- It is also easier to interpret for clinical stakeholders, so it was chosen as the primary model.

---

## Model Performance

Final test set metrics for the selected Logistic Regression model (threshold 0.50):

- Accuracy: 0.927  
- Precision: 0.857  
- Recall: 1.000  
- F1-score: 0.923  
- ROC-AUC: 0.993  

Confusion matrix (described):

- Most healthy patients were correctly classified as healthy.  
- A small number of healthy patients were incorrectly flagged as having heart disease (false positives).  
- No true disease cases were missed (zero false negatives).  
- All patients with heart disease in the test set were correctly identified.

Model comparison on the test set:

- Logistic Regression:
  - Accuracy: 0.927  
  - Precision: 0.857  
  - Recall: 1.000  
  - F1-score: 0.923  
  - ROC-AUC: 0.993  
- Random Forest:
  - Accuracy: 0.854  
  - Precision: 0.773  
  - Recall: 0.944  
  - F1-score: 0.850  
  - ROC-AUC: 0.960  

Logistic Regression outperformed Random Forest across all key metrics and was therefore selected as the primary model.

---

## Installation & Usage

Prerequisites:

- Python 3.8 or higher  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- joblib  

Setup steps:

1. Clone the repository and move into the project directory.  
2. Install required packages using `pip install -r requirements.txt`.  
3. Run the notebooks in order:
   - PT1_EDA_Cleaning  
   - PT2_Feature_Engineering  
   - PT3_Modeling  

Making predictions (conceptual):

- Load the trained pipeline from `models/heart_logreg_pipeline.joblib`.  
- Create a one‑row pandas DataFrame with all the required features (age, sex, bp, cholesterol, max_hr, st_depression, and each categorical feature).  
- Call the pipeline’s `predict` method to obtain the predicted class (heart disease present or absent).  
- Call `predict_proba` to obtain the probability of heart disease for that patient.

---

## Limitations & Future Work

Current limitations:

- Small dataset (270 patients); performance may not generalize to broader, more diverse populations.  
- Uses a public Kaggle dataset rather than real clinical electronic health record data.  
- Built as an educational proof-of-concept, not as a production or clinically approved tool.  
- Fairness across demographic groups (such as age and sex) has not yet been evaluated.  
- Limited feature set; no longitudinal information, lab trends, or richer clinical history.

Planned and recommended next steps:

- Validate the model on larger, real-world clinical datasets.  
- Assess performance and fairness across age, sex, and other demographic subgroups.  
- Work with clinicians to refine feature engineering and interpretation of model outputs.  
- Incorporate temporal and additional clinical data to improve predictive power.  
- Build a deployment-ready API and add monitoring plus explainability tooling for production use.

---

## Contact

Willber Escalante  
AGE Analytics  
Email: reasonwithwill@gmail.com  

---

## License

This project uses the Kaggle Heart Disease Prediction Dataset under the CC0 (Public Domain) license.
