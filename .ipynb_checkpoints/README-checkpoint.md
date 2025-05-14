# Loan Approval Prediction and Fairness Evaluation

This project explores the use of Machine Learning (ML) models for loan approval predictions and extends it into a **Responsible AI case study**, focusing on bias detection and fairness mitigation across sensitive attributes (e.g., gender).

## 📌 Project Goals
- Build a **standard ML classifier** to predict loan approval outcomes.
- Evaluate **fairness** across demographic groups using **Fairlearn**.
- Apply **bias mitigation** techniques to improve fairness while maintaining model accuracy.
- Integrate both **predictive performance** and **ethical AI practices**.

## 🚀 Project Structure

responsible-ai-loan-approval/
│
├── notebooks/
│ ├── 1-loan-approval-prediction.ipynb
│ └── 2-fairness-evaluation-and-mitigation.ipynb
│
├── data/
│ └── dataset.csv
│
├── images/
│ └── fairness-metrics.png
├── requirements.txt
└── README.md


## 📊 Methodology
- **Phase 1**: Train and evaluate a standard ML classifier (Logistic Regression/Decision Tree).
- **Phase 2**: Measure fairness metrics using **Fairlearn**.
- **Phase 3**: Apply fairness mitigation (e.g., reweighing, threshold optimization).
- **Phase 4** *(Optional)*: Document findings in a blog post.

## 💻 Tools & Technologies
- Python 3.x
- scikit-learn
- Fairlearn
- Jupyter Notebook
- Matplotlib / Seaborn
- Pandas / NumPy

## ✅ Setup Instructions

# Clone repo
git clone https://github.com/Simondelali/responsible-ai-loan-approval.git
cd responsible-ai-loan-approval

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
