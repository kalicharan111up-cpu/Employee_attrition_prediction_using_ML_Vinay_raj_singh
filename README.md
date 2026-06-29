# Employee_attrition_prediction_using_ML_Vinay_raj_singh
End-to-end ML project predicting employee attrition on the IBM HR dataset — EDA, model comparison (Logistic Regression/Random Forest/Gradient Boosting), and HR-ready business recommendations.

Employee Attrition Prediction — IBM HR Analytics

End-to-end data science project that analyzes 1,470 employee records to understand why
employees leave and builds a model to flag who's likely to leave next — with the goal of
giving an HR team specific, data-backed levers to pull.


Every company loses employees — but losing the wrong ones at the wrong time is expensive in
hiring, training, and lost productivity. This project walks through the full pipeline: explore
the data → clean it → find patterns → build and compare models → evaluate them → translate the
results into plain-language HR recommendations.




📊 Dataset

Source: IBM HR Analytics Employee Attrition & Performance
(WA_Fn-UseC_-HR-Employee-Attrition.csv)


1,470 employees, 35 original columns
Target column: Attrition (Yes / No)
16.12% attrition rate (237 left / 1,233 stayed) — an imbalanced target, handled explicitly
in the modeling stage rather than ignored



🗂️ Project Structure

.
├── attrition_analysis.ipynb     # Full notebook: EDA → cleaning → modeling → evaluation → insights
├── HR-Employee-Attrition.csv    # Dataset
├── summary.pdf / summary.docx   # 1-page, non-technical summary for HR leadership
└── README.md


🔧 Pipeline / Methodology


Data Loading & Exploration — shape, target balance, numeric vs. categorical breakdown
Cleaning & Preprocessing

Dropped constant/non-predictive columns: EmployeeNumber, Over18, StandardHours, EmployeeCount
Encoded target Attrition → 1/0
One-hot encoded remaining categorical features (Department, JobRole, MaritalStatus, BusinessTravel, etc.)
Scaled numeric features with StandardScaler — fit only on the training split to avoid
data leakage from the test set



Exploratory Data Analysis — attrition broken down by department, job role, income,
work-life balance, and tenure
Model Building — 80/20 train-test split (stratified), 3 models trained:

Logistic Regression (class_weight='balanced')
Random Forest (class_weight='balanced')
Gradient Boosting (no native class_weight support → balanced sample weights via
compute_sample_weight, fit with sample_weight=)



Evaluation — Precision, Recall, F1, ROC-AUC, confusion matrices for all 3 models
Visualization — attrition-by-department/role bars, income boxplot, confusion matrix
heatmap, feature importance chart, ROC curve comparison
Business Insights — translated model output into plain-language HR recommendations



🏆 Model Comparison

ModelPrecisionRecallF1-ScoreROC-AUCLogistic Regression ✅0.3450.6170.4430.798Gradient Boosting0.4070.4680.4360.779Random Forest0.5000.0850.1450.769

Best model: Logistic Regression — best Recall, F1, and ROC-AUC, and it's the most
explainable to a non-technical HR audience. Random Forest has the highest precision but misses
~91% of actual leavers (Recall 0.085), making it a poor fit for this use case despite balanced
class weights. For attrition prediction, catching at-risk employees (Recall) matters more than
avoiding false alarms.


🔑 Top Predictors of Attrition

Ranked by absolute logistic regression coefficient:


Job Role: Laboratory Technician
OverTime: Yes
Business Travel: Frequent
Total Working Years (more experience → lower risk)
Job Level
Job Role: Sales Representative
Business Travel: Rarely
Education Field: Life Sciences (lower risk)
Years Since Last Promotion
Department: Sales


Notably, raw Monthly Income does not make the top 10 — pay correlates with attrition but is
not an independent top driver once overtime, travel, role, and tenure are accounted for.


💡 Key Findings


Overtime is the strongest single red flag — employees working overtime leave at 30.5%
vs 10.4% for those who don't (~3x higher).
Attrition is front-loaded in tenure — 34.9% of departures happen in year one,
21.3% in year two, dropping to single digits after year 10.
Sales Representative is the highest-risk role at 39.8% attrition — more than double the
company average.
Sales is the highest-risk department (20.6%), ahead of HR (19.0%) and R&D (13.8%).
Equity matters: employees with zero stock options leave at 24.4%, vs 7–9% for
those with any stock options.
Pay alone doesn't explain attrition — leavers earn ~30% less on average, but income isn't
a top-10 model driver.



✅ HR Recommendations


Audit and cap overtime in Sales and Lab Technician teams — chronic overtime usually means
understaffing, so fix it with hiring or workload redistribution, not policy alone.
Build a "First 24 Months" retention program — proactive stay interviews at 90 days and 18
months, plus earlier stock-option eligibility for new hires, since that's the highest-risk
tenure window.



⚠️ Limitations


Single historical snapshot of 1,470 employees — not longitudinal, real-time company data.
Imbalanced target (only 16% attrition) limits how much signal the model can learn from the
minority class.
Best model's Precision is just 0.345 — roughly two-thirds of employees flagged "at risk"
will actually stay. This should be used as a screening signal to prioritize conversations,
not a verdict on any individual, and should be retrained/validated periodically as the
workforce and policies change.



🛠️ Tech Stack


Python — pandas, numpy
scikit-learn — LogisticRegression, RandomForestClassifier, GradientBoostingClassifier,
StandardScaler, train_test_split, metrics
Visualization — matplotlib, seaborn



▶️ Running This Project

bashgit clone <repo-url>
cd <repo-folder>
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook attrition_analysis.ipynb

Run all cells top to bottom — the notebook is self-contained and reads
HR-Employee-Attrition.csv from the same directory.


📄 License

This project uses the publicly available IBM HR Analytics Employee Attrition dataset for
educational/portfolio purposes.
