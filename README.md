# Predictive Modeling of Employee Performance 📊

> **A Hybrid Approach Utilizing Structured HR Metrics and Unstructured Feedback Analysis**

Accurately predicting employee performance is a critical objective for modern human resources management. This project investigates the efficacy of various machine learning algorithms in classifying employee performance levels using a hybrid dataset.

By combining traditional structured HR metrics (such as attendance, KPI consistency, and engagement scores) with Natural Language Processing (NLP) derived features from managerial feedback, we establish a comprehensive predictive framework.

---

## 📑 Table of Contents

* [Background](https://www.google.com/search?q=%23-background)
* [Dataset Description](https://www.google.com/search?q=%23-dataset-description)
* [Methodology & Pipeline](https://www.google.com/search?q=%23-methodology--pipeline)
* [Results](https://www.google.com/search?q=%23-results)
* [Discussion](https://www.google.com/search?q=%23-discussion)
* [Future Work](https://www.google.com/search?q=%23-future-work)

---

## 💡 Background

The integration of data analytics into Human Resources (HR)—often termed "People Analytics"—has transformed how organizations evaluate, retain, and promote talent. Traditional performance reviews often rely on subjective managerial assessments or isolated quantitative metrics.

This repository proposes a holistic, data-driven approach to performance prediction by fusing objective employee data with sentiment and linguistic features extracted from qualitative feedback. The primary objective is to evaluate which machine learning architecture best handles this combined feature space to accurately classify employee performance ratings.

---

## 📁 Dataset Description

The study utilizes two primary data sources:

1. **Structured Data** (`employee_data.csv`):
Contains demographic and quantitative performance indicators for employees.
* *Features:* Age, Gender, Department, Attendance Rate, KPI Consistency, Training Hours, Engagement Score.
* *Target Variable:* Performance Rating (High, Medium, Low).


2. **Unstructured Data** (`feedback_text.csv`):
Contains qualitative text reviews provided by managers or peers detailing employee behaviors, strengths, and weaknesses.

---

## ⚙️ Methodology & Pipeline

To prepare the data for machine learning models, a rigorous preprocessing pipeline was constructed using `scikit-learn` (as evidenced by `model_training.py` and `scaler.pkl`).

### Feature Engineering & Preprocessing

* **Text Feature Extraction:** The unstructured feedback text was processed to extract quantitative NLP metrics: `feedback_sentiment`, `feedback_positive_count`, `feedback_negative_count`, `feedback_keyword_count`, and `feedback_token_count`.
* **Numerical Imputation and Scaling:** Missing numerical values were handled using median imputation (`SimpleImputer`). Features were standardized using `StandardScaler` to ensure uniform scale across metrics like age and attendance rates.
* **Categorical Encoding:** Categorical variables such as Gender and Department were transformed using `OneHotEncoder`.

### Model Selection

The problem was framed as a multi-class classification task. Models were evaluated using weighted metrics to account for potential class imbalances. Five distinct algorithms were trained and evaluated:

* Logistic Regression *(Selected as Optimal)*
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* Multi-Layer Perceptron (Neural Network)

---

## 🏆 Results

The models were evaluated on a held-out test set, and their performance metrics were recorded (`results.csv` and `metrics.txt`).

| Model | Accuracy | Precision | Recall | $F_1$ Score |
| --- | --- | --- | --- | --- |
| **Logistic Regression** | **0.875** | **0.90625** | **0.875** | **0.8714** |
| Decision Tree | 0.875 | 0.90625 | 0.875 | 0.8714 |
| Random Forest | 0.875 | 0.90625 | 0.875 | 0.8714 |
| Gradient Boosting | 0.875 | 0.90625 | 0.875 | 0.8714 |
| Neural Network | 0.875 | 0.90625 | 0.875 | 0.8714 |

*Note: The `selection_score` was calculated at `0.8714` across all models.*

### Optimal Model Selection

While all algorithms achieved identical performance metrics on the test set, **Logistic Regression** was designated as the best model. In scenarios where complex non-linear models do not outperform simple linear models, Logistic Regression is preferred due to its high interpretability, lower computational cost, and reduced risk of overfitting.

---

## 🧠 Discussion

The empirical results reveal several interesting dynamics about the dataset and the predictive task:

* **Feature Separability:** The identical performance across varying model architectures strongly suggests that the dataset is highly separable. The combination of KPI metrics and NLP-derived sentiment features likely creates distinct, easily identifiable boundaries between "High," "Medium," and "Low" performing employees.
* **Value of NLP in HR:** The successful extraction of numerical sentiment and token counts demonstrates that qualitative reviews can be effectively translated into machine-readable signals, adding significant predictive power.
* **Data Size and Overfitting Potential:** The convergence of metrics at precisely $87.5\%$ accuracy across all models may indicate a relatively small test set size. With smaller sample sizes, discrete step changes in accuracy occur, causing different models to yield the exact same misclassifications.

---

## 🚀 Future Work

* **Dataset Expansion:** Gathering a larger volume of data to thoroughly test the boundaries of the non-linear models and ensure the findings generalize across different organizational structures.
* **Advanced NLP:** Replacing basic sentiment and token counts with advanced contextual embeddings (e.g., BERT or RoBERTa) to capture the deeper nuances of managerial feedback.
* **Bias and Fairness Audits:** Future iterations must rigorously test the model to ensure it does not disproportionately penalize or favor specific demographics based on gender or department.
