# Telecom Customer Churn Prediction

## Project Overview

Telecom Customer Churn Prediction is a Machine Learning classification project that predicts whether a customer is likely to leave a telecom service or continue using it.

The project uses customer demographic information, account details, billing information, and subscribed services to identify patterns associated with customer churn.

The main objective is to build and evaluate classification models that can help telecom companies identify customers at risk of leaving and support customer retention strategies.

## Problem Statement

Customer churn is a major challenge for telecom companies. Losing existing customers can affect revenue and business growth.

This project aims to predict customer churn using historical customer data and compare different Machine Learning algorithms to understand their predictive performance.

**Target Variable:** `Churn`

- `0` – Customer will not churn (No)
- `1` – Customer will churn (Yes)

## Dataset Description

The project uses the **Telco Customer Churn Dataset**.

- Total Records: 7,043
- Total Features: 21
- Target Variable: Churn

The dataset contains the following types of information:

| Category | Features |
|---|---|
| Customer Information | Gender, SeniorCitizen, Partner, Dependents |
| Account Information | Tenure, Contract, PaperlessBilling, PaymentMethod |
| Service Information | PhoneService, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Billing Information | MonthlyCharges, TotalCharges |
| Target | Churn |

## Technologies and Libraries Used

- Python
- Pandas – Data manipulation and analysis
- NumPy – Numerical computations
- Matplotlib – Data visualization
- Seaborn – Statistical visualization
- Scikit-learn – Data preprocessing, model building and evaluation
- Joblib – Saving and loading the trained model
- Jupyter Notebook – Development environment

## Project Workflow

### 1. Data Loading and Understanding

- Loaded the dataset using Pandas.
- Examined the first few records and dataset dimensions.
- Analyzed numerical and categorical variables.
- Checked missing values and unique categories.

### 2. Data Cleaning

- Converted `TotalCharges` into a numerical data type.
- Handled blank and invalid values by converting them into missing values.
- Removed records with missing `TotalCharges`.
- Checked for and removed exact duplicate records if present.

### 3. Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand customer behavior and identify patterns related to churn.

Key analysis included:

- Customer churn distribution.
- Numerical feature distributions using histograms.
- Boxplots to compare numerical features across churn categories.
- Group-wise descriptive statistics using mean, median and standard deviation.
- Outlier identification using the Interquartile Range (IQR) method.
- Categorical churn-rate analysis.
- Pearson correlation analysis using a heatmap.

### 4. Feature Engineering

Created five additional features from the existing customer information to capture meaningful patterns.

**1. NumAddOnServices**

Counts the number of subscribed add-on services marked as "Yes" across:

- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

**2. HasInternetService**

A binary feature indicating whether a customer has an internet connection.

- 1 – DSL or Fiber optic
- 0 – No internet service

**3. TenureGroup**

Categorizes customers according to their tenure:

- New: 0–12 months
- Medium: 13–24 months
- Long: 25–48 months
- Very Long: 49–72 months

**4. MonthlyChargeCategory**

Groups customers according to monthly charges:

- Low: 0–35
- Medium: 35–70
- High: 70–100
- Very High: 100–120

**5. HasFamily**

A binary feature created using Partner and Dependents.

- 1 – If Partner or Dependents is Yes.
- 0 – If both are No.

### 5. Data Preprocessing

Separated the dataset into independent variables (X) and the target variable (y).

Used a stratified train-test split:

- Training Data: 70%
- Testing Data: 30%

Applied separate preprocessing techniques to numerical and categorical features.

**Numerical Features:**
- Missing value handling using median imputation.
- Feature scaling using StandardScaler.

**Categorical Features:**
- Missing value handling using most-frequent imputation.
- One-Hot Encoding using OneHotEncoder.

Used ColumnTransformer and Pipeline to combine preprocessing and model training into a structured workflow.

### 6. Model Building

Implemented and compared two classification algorithms.

**Logistic Regression**
- Used as the baseline classification model.
- Applied class_weight='balanced' to account for class imbalance.
- Set max_iter=1000.

**Random Forest Classifier**
- Used an ensemble of 300 decision trees.
- Applied class_weight='balanced'.
- Captures non-linear relationships and feature interactions.

### 7. Model Evaluation

Evaluated model performance using the following classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

Also plotted the ROC curve to examine the trade-off between True Positive Rate and False Positive Rate.

### 8. Cross-Validation

Applied 5-Fold Stratified Cross-Validation to the Logistic Regression pipeline.

Evaluated the model using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Average Precision

Calculated the mean and standard deviation across folds to assess performance consistency.

### 9. Model Comparison and Saving

Compared Logistic Regression and Random Forest using their test-set evaluation metrics, with ROC-AUC used to sort the comparison.

Saved the trained Logistic Regression pipeline, including its preprocessing steps, using Joblib.

**Saved Model:** `telecom_churn_logistic_pipeline.pkl`

## Project Structure

```text
Telecom-Customer-Churn-Prediction/
│
├── Telco_Customer_Churn.csv
├── main1.ipynb
├── telecom_churn_logistic_pipeline.pkl
└── README.md
```

## Installation and Execution

**1. Clone the repository**

```bash
git clone <your-github-repository-url>
```

**2. Navigate to the project directory**

```bash
cd Telecom-Customer-Churn-Prediction
```

**3. Install the required libraries**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

**4. Launch Jupyter Notebook**

```bash
jupyter notebook
```

**5. Open and run**

Open `main1.ipynb` and execute the notebook cells in sequence.

Keep `Telco_Customer_Churn.csv` in the same directory as the notebook.

## Model Deployment / Prediction

The saved Logistic Regression pipeline can be loaded using Joblib:

```python
import joblib

model = joblib.load("telecom_churn_logistic_pipeline.pkl")

predictions = model.predict(new_customer_data)
probabilities = model.predict_proba(new_customer_data)[:, 1]
```

Here, `new_customer_data` should contain the same input feature columns used during model training.

## Conclusion

This project demonstrates an end-to-end Machine Learning workflow for telecom customer churn prediction, covering data cleaning, exploratory data analysis, feature engineering, preprocessing, classification, model evaluation and model saving.

The comparison of Logistic Regression and Random Forest provides a way to assess different approaches to the customer churn classification problem.

**Future Improvements**
- Hyperparameter tuning using GridSearchCV.
- Feature importance analysis and feature selection.
- Decision threshold optimization.
- Deployment through Flask or Streamlit.
- Integration with a customer retention dashboard.

---

*Note: Model performance values are intentionally not included because the notebook source does not provide verified executed evaluation results.*
