# 🎛️ The Black-Box Auditor: Enterprise ML Governance & Live XAI Engine

[(!https://blackbox-auditor.streamlit.app/)]

An interactive Model Governance and Explainable AI (XAI) diagnostic dashboard designed to audit black-box ensemble classifiers. This system evaluates real-world credit risk pipelines for algorithmic transparency, demographic fairness, and statutory compliance with the **EU AI Act**.

---

##  Key Features

*   ** Live Risk Evaluation Simulator:** An interactive sandbox letting users manipulate demographic and economic inputs (Age, Loan Amount, Duration) to watch the model compute predictions and execute localized SHAP feature attributions in real time.
*   ** Model Transparency Diagnostics (SHAP):** Combines global feature importance bar charts (across historical testing pools) with individual local waterfall plots to display exactly *why* a decision was reached.
*   ** Ethical AI & Fairness Tracking:** Automatically segments data to monitor the **Disparate Impact Ratio** and check for statistical parity variances between protected demographic groups (`Gender_Male` cohorts) to expose underlying historical training bias.
*   ** EU AI Act Regulatory Compliance Engine:** Bridges tech law with data science via a technical validation mapping layer that audits system safeguards against Chapter II statutory mandates for High-Risk AI Systems.

---

##  Architecture & Dataset

*   **Core Backend Pipeline:** Built using `scikit-learn`'s `RandomForestClassifier` ensemble pipeline.
*   **Production Data Source:** Fetches the real open-source **UCI German Credit Dataset** (1,000 historical consumer banking profiles) directly via OpenML to eliminate synthetic data bias.
*   **Explainability Infrastructure:** Leverages optimized `shap.TreeExplainer` layers configured to run on dynamic matrix slices for instant in-browser calculations.

---

##  Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/haz4rl/blackbox-auditor.git](https://github.com/haz4rl/blackbox-auditor.git)
   cd blackbox-auditor
