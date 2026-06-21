# 🏥 Hospital Emergency Department — Triage & Admission Prediction

> **Can we predict whether a patient will be admitted or discharged at the moment they walk into the emergency room?**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightblue?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi)
![Status](https://img.shields.io/badge/Status-Completed-green)

---

## 📌 Project Overview

This end to end data science project analyses **560,486 emergency department visits** from Yale-New Haven Health System (2014–2017) to:

- Predict hospital admission at triage before a doctor sees the patient.
- Identify peak congestion periods and patient flow patterns.
- Detect demographic disparities in admission rates
- Deliver actionable recommendations to reduce ED overcrowding

**Dataset:** [Hospital Triage and Patient History Data — Kaggle](https://kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data)  
**Original Paper:** Hong, Haimovich & Taylor (2018) — *Predicting hospital admission at emergency department triage using machine learning*, PLOS ONE  
**Track:** Stackron Academy — April Healthcare Case Study

---

## 🧭 Table of Contents

- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Key Findings](#-key-findings)
- [Model Results](#-model-results)
- [Equity Analysis](#-equity-analysis)
- [Recommendations](#-recommendations)
- [Limitations](#-limitations)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Tools Used](#-tools-used)
- [Author](#-author)

---

## 🏥 Business Problem

Emergency departments worldwide face a growing crisis, patients arrive in large volumes, triage nurses must assess severity in real time, and resource allocation decisions must be made instantly. When these systems fail, patients wait too long, health outcomes deteriorate, and clinical staff burn out.

**This project addresses four core questions:**

1. When are peak congestion periods and how should staffing respond?
2. Which patient profiles drive the highest admission burden?
3. Can we predict admission accurately enough to support triage decisions?
4. Are there demographic disparities in how patients are admitted?

**Stakeholders:** Hospital Administrators | Medical Staff | Health Ministries | Patients

---

## 📂 Dataset

| Property | Detail |
|---|---|
| **Source** | Yale-New Haven Health System via Kaggle |
| **Period** | March 2014 – July 2017 |
| **Total Visits** | 560,486 patient encounters |
| **Total Features** | 972 variables per visit |
| **Target Variable** | `disposition` — ADMIT or DISCHARGE |
| **Triage System** | Emergency Severity Index (ESI) 1–5 |

### ESI Triage Levels

| ESI Level | Urgency | Admission Rate |
|---|---|---|
| ESI-1 | Immediate — life-threatening | 85.6% |
| ESI-2 | Emergent — high risk | 55.0% |
| ESI-3 | Urgent — stable, needs resources | 29.1% |
| ESI-4 | Less Urgent | 2.2% |
| ESI-5 | Non-Urgent | 0.4% |

---

## 🔬 Approach

### 1. Data Cleaning
- Dropped rows with missing ESI values (~2,457 rows) unknown ESI is medically meaningless
- Dropped rows with missing age values (~11 rows)
- Applied median imputation to missing vital signs
- Treated missing chief complaint flags as 0 (complaint not present)
- Checked and removed duplicate records
- Replaced extreme outlier vitals (e.g. BP of 0 or 500) with NaN then imputed

### 2. Feature Engineering
Seven new features were created to improve model performance:

| Feature | Source | Purpose |
|---|---|---|
| `hour_of_day` | `arrivalhour_bin` | Peak congestion analysis |
| `age_group` | `age` | Age-specific admission patterns |
| `is_high_acuity` | `esi` | Binary urgency flag (ESI ≤ 2) |
| `english_speaker` | `lang` | Equity analysis feature |
| `has_prior_admission` | `n_admissions` | Re-admission risk predictor |
| `vital_abnormality_count` | All 6 vitals | Composite clinical risk score |
| `visit_season` | `arrivalmonth` | Seasonal health patterns |

### 3. Exploratory Data Analysis
- Heatmap analysis of patient volume by hour × day of week
- ESI level distribution and admission rates
- Vital signs comparison between admitted and discharged patients
- Demographic disparity analysis across gender, ethnicity, language, and insurance
- Chief complaint frequency and admission association analysis
- Prior visit history vs admission rate analysis

### 4. Machine Learning Models
Four binary classifiers were trained and compared:
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- Deep Neural Network (MLP)

---

## 📊 Key Findings

### Peak Congestion
- **Monday 11:00–14:00** is the single busiest period with **22,737 visits**
- The **11–14 hour bin** is consistently the most congested across every day of the week
- Early morning **(03–06)** is the quietest period — approximately 4–5x fewer visits than peak

### ESI & Admission Patterns
- **ESI 1 patients are admitted 85.6% of the time** — nearly 9 in 10 critical patients require inpatient care
- **ESI 4 and 5 patients are almost always discharged** — representing a major opportunity to redirect low acuity patients to walk in clinics
- Admission rate drops sharply and consistently from ESI 1 → ESI 5, confirming triage scoring accuracy

### Vital Signs
- All 6 vital signs showed **statistically significant differences** between admitted and discharged patients (Mann-Whitney U, p < 0.05)
- Admitted patients consistently showed **lower oxygen saturation, higher respiratory rate, higher heart rate**, and more unstable blood pressure
- **58.9% of patients** arrived with all vitals within normal range; **41.1%** had at least one abnormal vital

### Demographics
- **Self-pay patients have a 95.1% admission rate** — far above all other insurance groups, this suggest uninsured patients only visit the ED when critically ill
- **65+ patients are admitted 55.3% of the time** vs only 10.2% for patients aged 18–24
- **Non-English speakers are admitted at 25.9%** vs 30.1% for English speakers ,a 4.2 point gap across 45,545 patients
- **White or Caucasian patients (35.9%)** and **American Indian or Alaska Native patients (30.1%)** have the highest racial admission rates

### Chief Complaints
- **Shortness of breath** is the strongest predictor of admission (correlation 0.161)
- **Altered mental status** is second (0.128), confusion at triage is a serious red flag
- **Alcohol intoxication** tops the ESI severity chart (0.165) despite rarely resulting in admission
- **Motor vehicle crashes and sore throat** are negatively correlated with admission, most treated and discharged

### Prior History
- Patients with **3+ prior admissions have a 63% admission rate** vs 21% for first-time patients
- **31% of all patients** have at least one prior admission, flagging these at check in could significantly accelerate triage

---

## 🤖 Model Results

| Model | AUC | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|
| **Random Forest** | 0.8179 | 0.8250 | **0.8726** | **0.8482** | 0.50 |
| Neural Network | 0.8317 | 0.5155 | 0.7484 | 0.6105 | 0.30 |
| Logistic Regression | 0.8261 | 0.4843 | 0.8131 | 0.6070 | 0.50 |
| XGBoost | 0.8265 | 0.4841 | 0.7725 | 0.5952 | 0.35 |

### Recommended Model — Random Forest
Despite the Neural Network achieving the highest AUC (0.8317), **Random Forest is the recommended model for clinical deployment** because:
- Highest F1 score **(0.8482)** — best real world performance
- Fewest missed admissions **(201 false negatives)** — safest for patients
- Strong precision **(0.8250)** — fewer false alarms
- No threshold tuning required

### Top Predictors (XGBoost Feature Importance)

| Rank | Feature | Importance | Contribution |
|---|---|---|---|
| 1 | ESI Score | 0.4862 | 48.6% |
| 2 | Prior Admission History | 0.3105 | 31.1% |
| 3 | Age | 0.0382 | 3.8% |
| 4 | Language (English) | 0.0248 | 2.5% |
| 5 | Vital Abnormality Count | 0.0236 | 2.4% |

> **ESI score and prior admission history together drive 80% of the model's decisions**This confirms that what the triage nurse records in the first 3 minutes is the most powerful admission signal available.

---

## ⚖️ Equity Analysis

| Demographic | Finding | Implication |
|---|---|---|
| **Insurance** | Self-pay: 95.1% vs Medicaid: 20.9% | Uninsured patients arrive in extreme crisis only |
| **Language** | English: 30.1% vs Other: 25.9% | Non-English speakers may face communication barriers |
| **Race** | Range: 21% to 36% across racial groups | Warrants further investigation into care pathways |
| **Gender** | Male: 30.7% vs Female: 28.9% | Small but consistent gap |
| **Age** | 65+: 55.3% vs 18-24: 10.2% | Age is a strong independent predictor |

The most concerning finding is the **self-pay admission rate of 95.1%**,  this is not a clinical pattern but a financial access problem. Patients without insurance delay care until they have no choice but to be admitted.

---

## 💡 Recommendations

| # | Recommendation | Supporting Data | Stakeholder |
|---|---|---|---|
| 1 | **Deploy Random Forest model as triage decision support** | F1: 0.8482, Recall: 87.3% — catches most admissions early | Medical Staff |
| 2 | **Increase staffing on Monday 11:00–14:00** | 22,737 visits — highest single period in entire dataset | Administrators |
| 3 | **Redirect ESI 4 & 5 patients to walk-in clinics** | 2.2% and 0.4% admission rates — ED resources wasted | Operations Team |
| 4 | **Investigate and address barriers for self-pay patients** | 95.1% admission rate — financial access is a public health emergency | Health Ministry |
| 5 | **Flag patients with prior admissions at check-in** | 63% admission rate for patients with 3+ prior admissions | Triage Nurses |
| 6 | **Implement language support services at triage** | 4.2 point admission gap between English and non-English speakers | Equity Officers |

---

## ⚠️ Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **Single health system (Yale)** | Results may not generalise to other hospitals | Recommend external validation on different hospital systems |
| **No explicit wait time column** | Cannot directly model time-to-treatment | Used arrival patterns and ESI as proxies for congestion |
| **Missing ESI values (~2,457 rows dropped)** | Excluded patients may differ systematically | Reported % dropped; consistent with original paper approach |
| **Historical data (2014–2017)** | Pre-COVID patterns may not reflect current ED behaviour | Note temporal limitation; recommend retraining on post-2020 data |
| **De-identified data** | Cannot link patients to outcomes beyond admission | Model predicts admission only and not clinical outcome or survival |
| **Class imbalance (29.7% admitted)** | Risk of over predicting discharge | Applied class_weight='balanced' for Logistic Regression |

---

## 📁 Project Structure

```
hospital-triage-waittime-analysis/
├── README.md                        ← You are here
├── data/
│   ├── raw/                         ← Kaggle dataset link (file too large to upload)
│   └── processed/
│       └── hospital_cleaned.csv     ← Cleaned dataset with engineered features
├── notebooks/
│   ├── 01_EDA.ipynb                 ← Exploratory Data Analysis + visualisations
│   ├── 02_Feature_Engineering.ipynb ← Feature engineering steps
│   ├── 03_Modelling.ipynb           ← Model building, evaluation, comparison
│   └── 04_Equity_Analysis.ipynb     ← Demographic fairness analysis
├── models/
│   ├── random_forest.pkl            ← Best performing model
│   ├── xgboost.pkl                  ← XGBoost model
│   ├── logistic_regression.pkl      ← Baseline model
│   ├── neural_network.pkl           ← DNN model
│   └── scaler.pkl                   ← Feature scaler
├── dashboard/
│   └── operations_dashboard.pbix    ← Power BI operations dashboard
├── reports/
│   └── final_report.pdf             ← PDF summary of insights
└── requirements.txt                 ← Python dependencies
```

---

## ⚙️ How to Run

### Prerequisites
- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/MjDAnalyst/hospital-triage-waittime-analysis.git
cd hospital-triage-waittime-analysis

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
xgboost>=1.7
joblib>=1.3
streamlit>=1.25
scipy>=1.11
```

### Run the Notebooks
```bash
# Start Jupyter
jupyter notebook

# Run in order:
# 1. notebooks/01_EDA.ipynb
# 2. notebooks/02_Feature_Engineering.ipynb
# 3. notebooks/03_Modelling.ipynb
# 4. notebooks/04_Equity_Analysis.ipynb
```


### Dataset
Download the dataset from Kaggle:
[Hospital Triage and Patient History Data](https://kaggle.com/datasets/maalona/hospital-triage-and-patient-history-data)

Place the downloaded file at: `data/raw/5v_cleandf.rdata`

---

## 🛠 Tools Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10 |
| **Data Manipulation** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Statistical Testing** | SciPy (Mann-Whitney U) |
| **Model Persistence** | Joblib |
| **Dashboard** | Power BI|
| **Environment** | Jupyter Notebook, Anaconda |
| **Version Control** | Git, GitHub |

---

## 👩‍💻 Author

**Mujidat Fadeyi**  
Data Analyst | MSc Chemistry | Python • Power BI • Excel

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)]
(www.linkedin.com/in/fadeyi-mujidat)
[![GitHub](https://img.shields.io/badge/GitHub-MjDAnalyst-black?logo=github)](https://github.com/MjDAnalyst/MjDAnalyst)

---

*Built as part of the Stackron Academy — April Healthcare Track*  
*Dataset: Yale-New Haven Health System | Study Period: 2014–2017*
