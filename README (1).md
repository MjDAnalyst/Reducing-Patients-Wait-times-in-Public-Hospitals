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

- Predict hospital admission at triage before a doctor sees the patient
- Identify peak congestion periods and patient flow patterns
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
| **Analysed After Cleaning** | 255,308 records |
| **Total Features** | 972 variables per visit |
| **Model Features Used** | 12 engineered and clinical features |
| **Target Variable** | `disposition` — Admit or Discharge |
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
- Dropped rows with missing ESI values (~2,457 rows) — unknown ESI is medically meaningless
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
- Vital signs comparison between admitted and discharged patients (Mann-Whitney U test)
- Demographic disparity analysis across gender, ethnicity, language and insurance
- Chief complaint frequency and admission association analysis
- Prior visit history vs admission rate analysis

### 4. Machine Learning Models
Four binary classifiers trained and compared on 12 features across 255,308 records:
- Logistic Regression (baseline — with class_weight='balanced')
- Random Forest
- XGBoost (no class weight — handles imbalance naturally)
- Deep Neural Network / MLP

---

## 📊 Key Findings

### Peak Congestion
- **Monday 11:00–14:00** is the single busiest period with **22,737 visits**
- The **11–14 hour bin** is consistently the most congested across every day of the week
- Early morning **(03–06)** is the quietest period, approximately 4–5x fewer visits than peak
- **Season has minimal impact** — all four seasons fall within 1 percentage point of each other on admission rates

### ESI & Admission Patterns
- **ESI 1 patients are admitted 85.6% of the time** This means nearly 9 in 10 critical patients require inpatient care
- **ESI 4 and 5 patients are almost always discharged** This represents a major opportunity to redirect low-acuity patients to walk in clinics
- Admission rate drops sharply and consistently from ESI 1 to ESI 5, confirming triage scoring accuracy

### Vital Signs
- All 6 vital signs showed **statistically significant differences** between admitted and discharged patients (Mann-Whitney U, p < 0.05)
- Admitted patients consistently showed **lower oxygen saturation, higher respiratory rate, higher heart rate** and more unstable blood pressure
- **58.9% of patients** arrived with all vitals within normal range; **41.1%** had at least one abnormal vital

### Demographics
- **Self-pay patients have a 95.1% admission rate**  far above all other insurance groups, suggesting uninsured patients only visit the ED when critically ill
- **65+ patients are admitted 55.3% of the time** vs only 10.2% for patients aged 18–24
- **Non-English speakers are admitted at 25.9%** vs 30.1% for English speakers, a 4.2 point gap across 45,545 patients
- **White or Caucasian patients (35.9%)** and **American Indian or Alaska Native patients (30.1%)** have the highest racial admission rates

### Chief Complaints
- **Shortness of breath** is the strongest predictor of admission (correlation 0.161)
- **Altered mental status** is second (0.128) confusion at triage is a serious red flag
- **Alcohol intoxication** tops the ESI severity chart (0.165) despite rarely resulting in admission
- **Motor vehicle crashes and sore throat** are negatively correlated with admission, most treated and discharged

### Prior History
- Patients with **3+ prior admissions have a 63% admission rate** vs 21% for first-time patients
- **31% of all patients** have at least one prior admission. flagging these at check in could significantly accelerate triage

---

## 🤖 Model Results

### AUC Scores — All Models

| Model | AUC | Status |
|---|---|---|
| **Neural Network** | **0.8416** | 🥇 Best AUC |
| **XGBoost** | **0.8404** | 🥈 Very close |
| **Logistic Regression** | **0.8402** | 🥉 Strong baseline |
| **Random Forest** | **0.8197** | 4th |

### Full Evaluation — Default Threshold (0.50)

| Model | Precision | Recall | F1 | Missed Admissions |
|---|---|---|---|---|
| **Logistic Regression** | 0.5549 | **0.8031** | **0.6563** | **933** |
| Neural Network | 0.6552 | 0.5885 | 0.6201 | 1,950 |
| XGBoost | **0.6632** | 0.5664 | 0.6110 | 2,055 |
| Random Forest | 0.6236 | 0.5628 | 0.5916 | 2,072 |

### After Threshold Tuning

| Model | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|
| **Neural Network** | 0.5683 | 0.7854 | **0.6595** | **0.35** |
| **XGBoost** | 0.5544 | **0.8128** | 0.6592 | **0.30** |
| **Logistic Regression** | 0.5549 | 0.8031 | 0.6563 | 0.50 |
| **Random Forest** | 0.6236 | 0.5628 | 0.5916 | 0.50 |

### Deployment Recommendation

| Priority | Best Model | Threshold | Reason |
|---|---|---|---|
| **Never miss admission** | ✅ XGBoost | 0.30 | Highest recall 81.3% |
| **Best overall balance** | ✅ Neural Network | 0.35 | Highest F1 0.6595 |
| **Simplest to deploy** | ✅ Logistic Regression | 0.50 | No tuning needed, best recall at default |

### Top Predictors — XGBoost Feature Importance

| Rank | Feature | Importance | Contribution |
|---|---|---|---|
| 1 | **ESI Score** | 0.7159 | 71.6% |
| 2 | **Prior Admission History** | 0.1181 | 11.8% |
| 3 | **Age** | 0.0619 | 6.2% |
| 4 | **n_admissions** | 0.0179 | 1.8% |
| 5 | **Systolic BP** | 0.0156 | 1.6% |
| 6 | **English Speaker** | 0.0143 | 1.4% |
| 7 | **Diastolic BP** | 0.0133 | 1.3% |
| 8 | **Heart Rate** | 0.0115 | 1.2% |
| 9 | **Respiratory Rate** | 0.0109 | 1.1% |
| 10 | **Oxygen Saturation** | 0.0105 | 1.1% |
| 11 | **Vital Abnormality Count** | 0.0102 | 1.0% |
| 12 | **is_high_acuity** | 0.0000 | 0.0% |

> **ESI alone accounts for 71.6% of the model's decisions** This confirms that the triage nurse's initial severity score is the single most powerful admission predictor available at triage time.

---

## ⚖️ Equity Analysis

| Demographic | Finding | Implication |
|---|---|---|
| **Insurance** | Self-pay: 95.1% vs Medicaid: 20.9% | Uninsured patients arrive in extreme crisis only |
| **Language** | English: 30.1% vs Other: 25.9% | Non-English speakers may face communication barriers |
| **Race** | Range: 21% to 36% across racial groups | Warrants further investigation into care pathways |
| **Gender** | Male: 30.7% vs Female: 28.9% | Small but consistent gap |
| **Age** | 65+: 55.3% vs 18–24: 10.2% | Age is a strong independent predictor |

The most concerning finding is the **self-pay admission rate of 95.1%** — this is not a clinical pattern but a financial access problem. Patients without insurance delay care until they have no choice but to be admitted.

---

## 💡 Recommendations

| # | Recommendation | Supporting Data | Stakeholder |
|---|---|---|---|
| 1 | **Deploy XGBoost at threshold 0.30 as triage decision support** | Recall 81.3% — catches most admitted patients at triage | Medical Staff |
| 2 | **Increase staffing on Monday 11:00–14:00** | 22,737 visits highest single period in entire dataset | Administrators |
| 3 | **Redirect ESI 4 & 5 patients to walk-in clinics** | 2.2% and 0.4% admission rates, ED resources wasted | Operations Team |
| 4 | **Investigate and address barriers for self-pay patients** | 95.1% admission rate, financial access is a public health emergency | Health Ministry |
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
| **De-identified data** | Cannot link patients to outcomes beyond admission | Model predicts admission only — not clinical outcome or survival |
| **Class imbalance (30.9% admitted)** | Risk of over-predicting discharge | Applied class_weight='balanced' for Logistic Regression; natural handling for tree models |
| **Temperature column unavailable** | triage_vital_temp and temp_last were empty in processed dataset | Removed from feature set; temperature captured indirectly via vital_abnormality_count |

---

## 📁 Project Structure

```
hospital-triage-waittime-analysis/
├── README.md                        <- 
├── data/
│   ├── raw/                         <- Kaggle dataset link (file too large to upload)
│   └── processed/
│       └── hospital_cleaned.csv     <- Cleaned dataset with engineered features
├── notebooks/
│   ├── 01_EDA.ipynb                 <- Exploratory Data Analysis + visualisations
│   ├── 02_Feature_Engineering.ipynb <- Feature engineering steps
│   ├── 03_Modelling.ipynb           <- Model building, evaluation, comparison
│   └── 04_Equity_Analysis.ipynb     <- Demographic fairness analysis
├── models/
│   ├── logistic_regression.pkl      <- Best recall at default threshold
│   ├── xgboost.pkl                  <- Best recall after tuning (threshold 0.30)
│   ├── neural_network.pkl           <- Best F1 after tuning (threshold 0.35)
│   ├── random_forest.pkl            <- Baseline ensemble model
│   └── scaler.pkl                   <- Feature scaler
├── dashboard/
│   └── operations_dashboard.pbix    <- Power BI operations dashboard
├── reports/
│   └── final_report.pdf             <- PDF summary of insights
└── requirements.txt                 <- Python dependencies
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
| **Dashboard** | Power BI, Streamlit |
| **Environment** | Jupyter Notebook, Anaconda |
| **Version Control** | Git, GitHub |

---

## 👩‍💻 Author

**Mujidat Fadeyi**  
Data Analyst | MSc Chemistry | Python • Power BI • Excel

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/your-linkedin)
[![GitHub](https://img.shields.io/badge/GitHub-MjDAnalyst-black?logo=github)](https://github.com/MjDAnalyst/MjDAnalyst)

---

*Built as part of the Stackron Academy — April Healthcare Track*  
*Dataset: Yale-New Haven Health System | Study Period: 2014–2017*
