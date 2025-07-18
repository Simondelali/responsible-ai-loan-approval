# Loan Approval Prediction and Fairness Evaluation

This project explores the use of Machine Learning (ML) models for loan approval predictions and extends it into a **Responsible AI case study**, focusing on bias detection and fairness mitigation across sensitive attributes (e.g., gender).

## 📌 Project Goals

- Build a **standard ML classifier** to predict loan approval outcomes.
- Evaluate **fairness** across demographic groups using **Fairlearn**.
- Apply **bias mitigation** techniques to improve fairness while maintaining model accuracy.
- Integrate both **predictive performance** and **ethical AI practices**.

## 📁 Project Structure

- responsible-ai-loan-approval/
  - notebooks/
    - 1-loan-approval-prediction.ipynb
    - 2-fairness-evaluation-and-mitigation.ipynb
  - data/
    - dataset.csv
  - images/
    - fairness-metrics.png
  - requirements.txt
  - README.md

## 📊 Methodology

- **Phase 1**: Train and evaluate a standard ML classifier (Logistic Regression/Decision Tree).
- **Phase 2**: Measure fairness metrics using **Fairlearn**.
- **Phase 3**: Apply fairness mitigation (e.g., reweighing, threshold optimization).

## 💻 Tools & Technologies

- Python 3.x
- scikit-learn
- Fairlearn
- Jupyter Notebook
- Matplotlib / Seaborn
- Pandas / NumPy

## ✅ Setup Instructions

# Clone repo

- git clone https://github.com/Simondelali/responsible-ai-loan-approval.git
- cd responsible-ai-loan-approval

# Create virtual environment

- python3 -m venv venv
- source venv/bin/activate

# Install dependencies

pip install -r requirements.txt

# Launch Jupyter Notebook

jupyter notebook

## ✅ Phase 1: Loan Approval Prediction Summary

The best-performing model was **Gaussian Naive Bayes (78.86% accuracy)**, followed by Random Forest (76.4%), K-Nearest Neighbors (75.6%), and Decision Tree (73.9%).

### 📊 Key Highlights:

- **Best Model**: Gaussian Naive Bayes (78.86% accuracy)
- **F1-Score**: 85.87%
- **Observation**: High accuracy for approvals, lower performance on rejections, indicating potential fairness concerns.

### 📈 Conclusion:

The baseline prediction model performs well overall but shows bias towards loan approvals, motivating the need for fairness analysis in **Phase 2**.

## ✅ Phase 2: Fairness Evaluation and Responsible AI Auditing

Fairness was assessed using **Fairlearn** with multiple group fairness metrics to capture both **selection fairness** and **outcome fairness** without applying artificial bias modifications.

### 📈 Key Fairness Results:

- **Overall Accuracy:** ~84.5%
- **Demographic Parity Difference (DPD):** 0.0963 – ✅ _Good (Balanced access rates across gender)_
- **Equalized Odds Difference (EOD):** 0.0808 – ✅ _Good (Balanced error rates across gender)_
- **Recall (Sensitivity):** 100% for female applicants
- **Precision:** 77% for females, 85% for males

### 📌 Summary:

The fairness audit demonstrates that the model maintains **fair decision-making** across gender groups in both access and prediction outcomes, following responsible AI guidelines suitable for ethical use in financial decision systems.
