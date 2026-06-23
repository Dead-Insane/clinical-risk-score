# 🩺 Clinical Diabetes Risk Scorer

An end-to-end clinical machine learning project that predicts diabetes risk 
using real CDC NHANES blood test and demographic data, with SHAP explainability 
and an interactive Streamlit risk assessment tool.

---

## 🎯 Problem Statement

Type 2 diabetes is often diagnosed late because early indicators in routine 
blood work go unnoticed. This project builds a clinical decision-support tool that:

- Predicts diabetes risk from **routine CBC + demographic data**
- Explains **why** a patient is flagged as high risk (SHAP)
- Stratifies patients into **Low / Medium / High risk tiers**
- Provides clinically meaningful recommendations per tier

---

## 🗂️ Dataset

**Source:** [CDC NHANES 2017–2018](https://wwwn.cdc.gov/nchs/nhanes/)
*(National Health and Nutrition Examination Survey — real US population health data)*

| File | Content |
|---|---|
| CBC_J | Complete Blood Count (WBC, RBC, Hemoglobin, Hematocrit, MCV, Platelets) |
| DEMO_J | Demographics (Age, Gender, BMI) |
| DIQ_J | Diabetes diagnosis (target variable) |

Merged dataset: **8,366 participants** → **7,354 after data quality filtering**

---

## 📁 Project Structure
clinical-risk-scorer/

│

├── data/                         # Charts and visual outputs

├── notebooks/

│   ├── 01_eda.ipynb              # Load, merge, explore NHANES data

│   ├── 02_preprocessing.ipynb   # Cleaning, feature engineering

│   └── 03_modeling.ipynb        # Modeling, SHAP explainability

├── src/

│   ├── model.pkl                 # Saved Logistic Regression model

│   ├── scaler.pkl                # Saved feature scaler

│   ├── explainer.pkl             # Saved SHAP explainer

│   └── feature_columns.pkl      # Saved feature alignment

├── app/

│   └── streamlit_app.py         # Interactive clinical risk scorer

├── requirements.txt

└── README.md
---

## 🔍 Key Findings from EDA

- **Diabetes prevalence: 10.42%** in the studied population — significant class imbalance (~9:1)
- **Age** is strongly associated with diabetes risk — risk rises sharply after 45
- **BMI** shows a clear positive relationship with diabetes — clinically validated
- **Hemoglobin/Hematocrit** show measurable differences between diabetic and non-diabetic groups
- **WBC** is mildly elevated in diabetic patients, consistent with chronic low-grade inflammation

---

## ⚙️ Methodology

### Missing Data Strategy
- **CBC columns** (10.17% missing, all missing together): rows dropped
  → All 6 blood values missing simultaneously indicates the participant 
  skipped the blood draw entirely — imputing 6 correlated clinical values 
  at once would be medically meaningless
- **BMI** (4.39% missing): imputed with median — safe for a single, 
  independently-missing column

### Feature Engineering (Domain-Driven)
| Feature | Clinical Rationale |
|---|---|
| `Age_BMI_interaction` | Composite metabolic syndrome signal |
| `Anemia_flag` | Hemoglobin < 12 g/dL — linked to diabetic nephropathy |
| `High_WBC_flag` | WBC > 11 — marker of chronic inflammation, a T2D comorbidity |

### Class Imbalance
- Diabetes rate ~10% → handled via `class_weight='balanced'` 
  (chosen over SMOTE for production-appropriate simplicity)

---

## 🤖 Models Trained & Compared

| Model | Accuracy | Recall (Diabetic) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.733 | **0.872** | **0.850** |
| Random Forest | 0.893 | 0.122 | 0.857 |
| XGBoost | 0.854 | 0.415 | 0.833 |

**Best Model: Logistic Regression**

Random Forest achieved 89.3% accuracy but caught only **12.2%** of actual 
diabetic patients — a textbook imbalanced-data trap where a model optimizes 
for the majority class and becomes clinically useless. Logistic Regression 
was chosen for catching **87.2% of true diabetic cases**, since missing a 
diabetic patient carries far greater clinical cost than a false alarm.

---

## 🔬 Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to make every prediction interpretable:

- **Age and BMI** are the dominant global risk drivers
- `Age_BMI_interaction` (engineered feature) contributed meaningfully to model output
- `Anemia_flag` and `High_WBC_flag` added clinically relevant secondary signal
- Each patient prediction includes an individual **SHAP waterfall plot** in the 
  Streamlit app, showing exactly which lab values pushed their risk score up or down

---

## 🚦 Patient Risk Stratification (Test Set, n=1,471)

| Risk Tier | Patients | % |
|---|---|---|
| 🟢 Low Risk (<30%) | 752 | 51.1% |
| 🟡 Medium Risk (30–60%) | 299 | 20.3% |
| 🔴 High Risk (>60%) | 420 | 28.5% |

**Nearly 1 in 3 patients in this sample fall into the high-risk tier** — 
demonstrating the real-world screening value of a tool like this.

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/clinical-risk-scorer.git
cd clinical-risk-scorer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download NHANES data**

Download from [CDC NHANES 2017-2018](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&CycleBeginYear=2017):
- `CBC_J.XPT`, `DEMO_J.XPT`, `DIQ_J.XPT` → save to `data/`

**4. Run notebooks in order**

notebooks/01_eda.ipynb

notebooks/02_preprocessing.ipynb

notebooks/03_modeling.ipynb

**5. Launch the Streamlit app**
```bash
cd app
streamlit run streamlit_app.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Deployment | Streamlit |
| Version Control | Git, GitHub |

---

## ⚠️ Disclaimer

This tool is built for educational and portfolio purposes using public NHANES 
data. It is **not a substitute for professional medical diagnosis**. All clinical 
recommendations shown in the app are illustrative, not actionable medical advice.

---

## 👤 Author

**Divya Prakash**  
Biotechnology background → Data Science | Lab Technician experience informs 
domain-driven feature engineering in this project  
[LinkedIn](https://linkedin.com/in/divya-prakash-09w) | 
[GitHub](https://github.com/Dead-Insane)