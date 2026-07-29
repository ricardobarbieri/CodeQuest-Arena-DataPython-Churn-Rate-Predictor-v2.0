🎯 **CodeQuest Arena — Churn Rate Predictor v2.0**

Data Science pipeline in Python (Pandas + Scikit-Learn) to predict subscription cancellation (churn), with a graphical user interface (GUI) for visual execution. Developed as a reference solution (10/10 answer key) for the DataPython challenge of the CodeQuest Arena competition.

---

### 📌 About the Project

This project implements a complete and reproducible Machine Learning pipeline to predict which customers are most likely to cancel a service (churn), using a dataset in the style of the classic Telco Customer Churn. It was built to be simultaneously:

* **Functional** — trains, evaluates, and saves a real model.
* **Didactic** — each step is commented explaining the *why*, not just the *how*.
* **Visual** — includes a simple graphical interface to monitor the execution without needing the terminal.

---

### ✨ Features

* 🧹 **Data cleaning** (handling missing values with median imputation).
* 🔤 **Automatic encoding** of categorical variables.
* 🤖 **Comparative training** between Logistic Regression and Random Forest.
* 📊 **Evaluation using multiple metrics:** Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
* 🧩 **Confusion matrix** and ranking of the most important variables.
* 💾 **Automatic export** of the best model (`.pkl`).
* 🖥️ **Graphical interface** (Tkinter) to run the pipeline with a single click.

---

### 📁 Repository Structure

```text
Churn Rate Predictor v2.0/
│
├── generate_dataset.py     # Generates the synthetic dataset telco_churn.csv
├── churn_pipeline.py       # ML Pipeline (cleaning, training, evaluation, saving)
├── churn_gui.py            # Graphical interface that runs the pipeline
├── telco_churn.csv         # Generated dataset (after running generate_dataset.py)
├── best_churn_model.pkl    # Saved trained model (generated after running the pipeline)
└── README.md

```

---

### ⚙️ Requirements

* Python 3.9 or higher
* Libraries: `pandas`, `scikit-learn`, `joblib` (Tkinter comes with standard Python)

---

### 🚀 Installation and Usage

```bash
# 1. Clone the repository
git clone https://github.com/your-user/codequest-arena-churn.git
cd codequest-arena-churn

# 2. Install dependencies
pip install pandas scikit-learn joblib

# 3. Generate the dataset (run only once)
python generate_dataset.py

# 4. Execute the pipeline via terminal...
python churn_pipeline.py

# ...or via graphical interface
python churn_gui.py

```

**Expected Output (terminal)**

```text
[CLEANING] 23 missing values found in 'TotalCharges'
[CLEANING] Imputed with the median = 2078.85
[ENCODING] Encoded categorical columns: [...]

=== Logistic Regression ===
Accuracy : 0.6775
Precision : 0.5902
Recall   : 0.4768
F1-Score : 0.5275
AUC-ROC  : 0.7145
...

>>> Best model (by F1-Score): Logistic Regression

[INTERPRETABILITY] Top 5 most important variables (Random Forest):
MonthlyCharges    0.203848
TotalCharges      0.201140
tenure            0.176385
...

[PERSISTENCE] Winning model saved to: best_churn_model.pkl

```

---

### 🧠 Methodology

| Step | Technique Used | Justification |
| --- | --- | --- |
| **Missing value handling** | Median imputation | More robust to outliers than the mean. |
| **Categorical variables** | `LabelEncoder` | Converts text to numbers for the models. |
| **Variable scaling** | `StandardScaler` (only in Logistic Regression) | Distance/gradient-based models are sensitive to scale; trees are not. |
| **Train/test split** | 80/20 with `stratify=y` | Maintains the proportion of churners in both sets (imbalanced data). |
| **Best model selection** | Highest F1-Score | More robust than isolated accuracy in imbalanced problems. |
| **Reproducibility** | Fixed `random_state=42` in all steps | Ensures identical results on every run. |

---

### 📏 Evaluation Criteria (Competition)

| Criterion | Weight |
| --- | --- |
| Pipeline executes without errors | 20% |
| Correct handling of nulls and categoricals | 20% |
| Correct metrics (not just accuracy) | 25% |
| Comparison between 2 models | 15% |
| Code organization/readability | 10% |
| Model correctly saved | 10% |

---

### 🔮 Possible Extensions

* [ ] Class balancing (`class_weight="balanced"` or SMOTE)
* [ ] Cross-validation (`cross_val_score`)
* [ ] Hyperparameter tuning (`GridSearchCV`)
* [ ] One-Hot Encoding for variables without a natural order
* [ ] Package the GUI as an executable (`.exe`) with PyInstaller

---

### 🐞 Troubleshooting

| Error | Cause | Solution |
| --- | --- | --- |
| `FileNotFoundError: telco_churn.csv` | Dataset was not generated | Run `python generate_dataset.py` first. |
| `ModuleNotFoundError` | Missing dependency | Run `pip install pandas scikit-learn joblib`. |
| `TypeError: numpy string dtypes...` | Pandas version incompatible with `include=["object","str"]` | Use `include=["object"]` in `select_dtypes`. |
| `python: command not found` | Python is not in the PATH | Try `python3` instead of `python`. |

---

### 📄 License

This project is licensed under the MIT License — feel free to use, study, and adapt.

### 👤 Author

Developed by Ricardo Barbieri for the CodeQuest Arena — DataPython challenge.
