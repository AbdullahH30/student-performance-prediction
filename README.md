# 🎓 Student Performance Prediction System

> An end-to-end Machine Learning system that predicts student academic performance
> using 15 behavioural, demographic, and academic features deployed as an
> interactive Flask web application with REST API.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.45-purple)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [REST API](#-rest-api)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🔍 Overview

This project demonstrates the **complete machine learning lifecycle**  from raw data through
deployment within a single, well-structured codebase. It is built to portfolio-grade standard
for graduate school applications (MSc AI / Data Science) and professional employer review.

**Key highlights:**

| Feature | Detail |
|---|---|
| Dataset | 2,000 synthetic student records (research-distribution-matched) |
| Target | 3-class: Low / Medium / High academic performance |
| Models | 9 classifiers compared with hyperparameter tuning |
| Best model | Logistic Regression — **84% test accuracy, 0.92 ROC-AUC** |
| Explainability | SHAP values, permutation importance, feature importance |
| Deployment | Flask web app + REST JSON API |
| Frontend | Bootstrap 5, glassmorphism design, dark/light mode |

---

## 🎯 Objectives

1. Build a modular, production-quality ML pipeline.
2. Compare nine diverse classifiers under identical conditions.
3. Apply Explainable AI (SHAP) to surface model reasoning.
4. Deploy as an interactive web application with REST API.
5. Demonstrate best practices: PEP 8, type hints, docstrings, logging.

---

## 📁 Project Structure

```
Student-Performance-Prediction/
│
├── app/                        # Flask web application
│   ├── app.py                  # Application factory
│   ├── routes.py               # All URL routes + REST API
│   ├── model_loader.py         # Thread-safe artefact caching
│   ├── templates/              # Jinja2 HTML templates (7 pages)
│   └── static/
│       ├── css/style.css       # Glassmorphism stylesheet
│       └── js/main.js          # Theme toggle, animations
│
├── data/
│   ├── generate_data.py        # Synthetic dataset generator
│   └── student_performance.csv # Generated dataset (2,000 rows)
│
├── models/                     # Saved artefacts (post-training)
│   ├── trained_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── reports/                    # Generated plots & metrics (post-pipeline)
│   ├── *.png                   # 17 visualisation plots
│   ├── test_metrics.json
│   ├── model_comparison.json
│   └── shap_importance.json
│
├── src/                        # Core ML modules
│   ├── utils.py                # Shared helpers, paths, serialisation
│   ├── preprocessing.py        # Full preprocessing pipeline
│   ├── eda.py                  # EDA plot generation (9 plots)
│   ├── feature_engineering.py  # 7 engineered features
│   ├── train.py                # 9 models + hyperparameter tuning
│   ├── evaluate.py             # Metrics + evaluation plots
│   ├── explainability.py       # SHAP analysis
│   └── predict.py              # Single / batch inference
│
├── pipeline.py                 # Master orchestration script
├── wsgi.py                     # Production WSGI entry-point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The dataset is generated to closely mirror real-world student performance research distributions.

| Column | Type | Description |
|---|---|---|
| `gender` | Binary | Male / Female |
| `age` | Numeric | 15–18 years |
| `school_type` | Binary | Public / Private |
| `distance_from_home` | Ordinal | Near / Moderate / Far |
| `parental_education` | Ordinal | None / High School / College / Postgraduate |
| `family_support` | Ordinal | Low / Medium / High |
| `internet_access` | Binary | Yes / No |
| `study_hours` | Numeric | 0–12 h/day |
| `attendance` | Numeric | 0–100 % |
| `absences` | Numeric | Days missed per year |
| `previous_grades` | Numeric | 0–100 score |
| `sleep_hours` | Numeric | 4–10 h/night |
| `extracurricular` | Binary | Yes / No |
| `tutoring` | Binary | Yes / No |
| `motivation_level` | Ordinal | Low / Medium / High |
| **`performance`** | **Target** | **Low / Medium / High** |

~3% missing values are artificially introduced in `sleep_hours`, `family_support`,
and `parental_education` to exercise the imputation pipeline.

---

## 🧠 ML Pipeline

### Step 1 – Preprocessing
- Median / mode imputation for missing values
- Winsorisation (z = 3.5) for outliers
- Ordinal and binary encoding
- `StandardScaler` fitted only on training data
- Stratified 80/10/10 train/val/test split

### Step 2 – EDA (9 plots)
Target distribution, attendance KDE, grade histogram, boxplots, correlation heatmap, pairplot, categorical feature analysis, missing value chart.

### Step 3 – Feature Engineering (7 features)

| Feature | Formula | Rationale |
|---|---|---|
| `study_efficiency` | study_hours / (attendance/100) | Effective study adjusted for presence |
| `attendance_ratio` | attendance / 100 | Normalised to [0,1] |
| `academic_engagement` | 0.6·study + 0.4·attendance | Single engagement score |
| `support_index` | family_support + tutoring + internet | Cumulative support resources |
| `avg_prev_perf` | previous_grades / 100 | Scale-normalised prior achievement |
| `sleep_study_balance` | study_hours × (1 − |sleep − 7.5| / 7.5) | Captures cognitive readiness |
| `risk_score` | absences × (1−study) × (1−grades) | Multiplicative at-risk signal |

### Step 4 – Model Training

Nine classifiers trained with 5-fold stratified CV hyperparameter search:

| Model | Search Type |
|---|---|
| Logistic Regression | GridSearchCV |
| Decision Tree | GridSearchCV |
| Random Forest | RandomizedSearchCV |
| Extra Trees | RandomizedSearchCV |
| Gradient Boosting | RandomizedSearchCV |
| XGBoost | RandomizedSearchCV |
| SVM | GridSearchCV |
| KNN | GridSearchCV |
| Naive Bayes | GridSearchCV |

### Step 5 – Evaluation
Accuracy, Precision, Recall, F1 (macro + weighted), ROC-AUC (OvR), confusion matrix, learning curve, permutation importance.

### Step 6 – Explainable AI
SHAP LinearExplainer / KernelExplainer, summary beeswarm, bar chart, and waterfall plots.

---

## 📈 Model Performance

| Model | CV Acc | Val Acc | F1 (weighted) |
|---|---|---|---|
| **Logistic Regression** ✓ | 0.8629 | **0.8750** | **0.8670** |
| SVM | 0.8607 | 0.8600 | 0.8508 |
| XGBoost | 0.8550 | 0.8400 | 0.8207 |
| Gradient Boosting | 0.8500 | 0.8500 | 0.8317 |
| Naive Bayes | 0.8500 | 0.8400 | 0.8108 |
| Random Forest | 0.8443 | 0.8250 | 0.7758 |
| Extra Trees | 0.8314 | 0.8200 | 0.7645 |
| Decision Tree | 0.8200 | 0.8100 | 0.7938 |
| KNN | 0.8193 | 0.8450 | 0.8028 |

**Test set (best model — Logistic Regression):**

| Metric | Score |
|---|---|
| Accuracy | 84.0% |
| F1 (weighted) | 82.5% |
| F1 (macro) | 61.6% |
| ROC-AUC (OvR) | **91.9%** |

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/AbdullahHussain/Student-Performance-Prediction.git
cd Student-Performance-Prediction

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full ML pipeline (generates data, trains models, saves artefacts)
python pipeline.py

# 5. Start the Flask web server
flask --app app/app.py run --debug
# or for production:
gunicorn --bind 0.0.0.0:5000 wsgi:application
```

Open your browser at **http://localhost:5000**

---

## 🖥 Usage

| Page | URL | Description |
|---|---|---|
| Home | `/` | Dashboard with stats and model leaderboard |
| About | `/about` | Project overview and author info |
| Dataset | `/dataset` | Feature descriptions and data preview |
| Model | `/performance` | Full evaluation metrics and plots |
| Visualizations | `/visualizations` | All EDA and XAI plots (lightbox) |
| Predict | `/predict` | Interactive student prediction form |
| Contact | `/contact` | Author links and setup guide |
| Health | `/health` | API health check |

---

## 🔌 REST API

### Health Check

```http
GET /health
```

**Response:**
```json
{ "status": "ok", "models_ready": true }
```

### Predict

```http
POST /api/predict
Content-Type: application/json
```

**Request body:**
```json
{
  "gender": "Female",
  "age": 16,
  "school_type": "Public",
  "distance_from_home": "Near",
  "parental_education": "College",
  "family_support": "High",
  "internet_access": "Yes",
  "study_hours": 7.0,
  "attendance": 88.0,
  "absences": 3,
  "previous_grades": 78.0,
  "sleep_hours": 7.5,
  "extracurricular": "Yes",
  "tutoring": "No",
  "motivation_level": "High"
}
```

**Response:**
```json
{
  "predicted_class": "High",
  "probabilities": {
    "Low": 0.0000,
    "Medium": 0.2078,
    "High": 0.7922
  },
  "confidence": 0.7922
}
```

---

## 🔮 Future Improvements

- [ ] Train on a real large-scale dataset (PISA, NAEP, UCI)
- [ ] Add deep learning baseline (TabNet / MLP)
- [ ] Implement LIME alongside SHAP for local explanations
- [ ] Add student longitudinal tracking (time-series features)
- [ ] Docker containerisation + CI/CD pipeline
- [ ] User authentication and prediction history
- [ ] Multi-language support (Urdu / Arabic)
- [ ] Export prediction reports as PDF

---

## 👤 Author

**Abdullah Hussain**  


---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using Python · Scikit-learn · XGBoost · SHAP · Flask · Bootstrap 5*
