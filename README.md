# 🎓 Placement Prediction System: ML Audit, Model Analytics & Interactive Dashboard

A machine learning analytics platform and interactive web application for student placement prediction, built using **Logistic Regression**, **Scikit-Learn**, **Plotly**, and **Streamlit**.

The project features a verified machine learning pipeline, data leakage prevention, cross-validation threshold tuning, and an academic analytics dashboard designed for university project demonstrations and viva examinations.

---

## 📌 1. Project Overview & Workflow

Student placement prediction is formulated as a supervised binary classification problem:
* `0` → **Not Placed** (Negative Class)
* `1` → **Placed** (Positive Class)

### End-to-End Pipeline Architecture

```text
               placement.csv (10,000 Records)
                            │
               Remove 'College_ID' (Identifier)
               Target Encoding ('No': 0, 'Yes': 1)
               One-Hot Encoding ('Internship_Experience')
                            │
               Stratified Train/Test Split (80/20, seed=42)
               ┌────────────┴────────────┐
               ▼                         ▼
         X_train (8,000)           X_test (2,000 Held-Out)
               │                         │
     StandardScaler (fit_transform)      │
               │                         │
     Logistic Regression (L2)            │
     5-Fold Stratified CV on Train       │
     (Optimal Threshold Tuning = 0.36)   │
               │                         │
               ├─────────────────────────┤ (transform only)
               ▼                         ▼
     Trained Model & Scaler ──────> Evaluation Suite:
                                    • Confusion Matrix (TN, FP, FN, TP)
                                    • Accuracy, Precision, Recall, F1
                                    • ROC Curve & ROC-AUC (0.9451)
                                    • PR Curve & PR-AUC (0.7793)
                                    • Threshold Sensitivity Sweep
                                         │
                                         ▼
                            Interactive Streamlit Dashboard
```

---

## 🔍 2. Phase One & Two — ML Technical Audit & Findings

During the technical audit of the original codebase (`app.py`, `placement.ipynb`, saved `.pkl` files), the following key findings were identified and resolved:

### A. The Decision Threshold Bug (Loop Iteration Variable Leak)
* **Issue Discovered**: In `placement.ipynb` cell 398, a for-loop was executed to evaluate thresholds:
  ```python
  for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
      ...
  joblib.dump(threshold, "threshold.pkl")  # Leaked loop variable = 0.80!
  ```
  Consequently, `threshold.pkl` stored `0.80`. When `app.py` loaded this file, it applied `probability >= 0.80`, causing:
  - **Severe Recall Collapse**: Test recall dropped from **58.73%** (at 0.50) to **31.63%** (at 0.80).
  - **Excessive False Negatives**: 227 placed students out of 332 were misclassified as "Not Placed" (68.4% missed opportunity rate).
* **Fix Applied**: 
  - Synchronized `threshold.pkl` to the standard default of `0.50` (or `0.36` for CV F1-optimal).
  - Implemented an interactive **Threshold Sensitivity Controller** in the Streamlit dashboard, allowing real-time switching between Default (`0.50`), CV-Tuned (`0.36`), and Conservative (`0.80`), keeping all metrics and confusion matrices 100% consistent with the active threshold.

### B. Prevention of Data Leakage
* **Verification**: `StandardScaler` is fitted strictly on `X_train` (`fit_transform`) and only applied (`transform`) to `X_test` and single student inferences.
* **Threshold Tuning Isolation**: Decision threshold tuning was conducted strictly on `X_train` using 5-fold stratified cross-validation (reaching optimal F1 at `0.36`), leaving the 2,000-record test set completely untouched.

### C. Collinearity & The Dummy Variable Trap
* **Finding**: `Internship_Experience` was one-hot encoded into both `Internship_Experience_No` and `Internship_Experience_Yes`, which are perfectly collinear ($No + Yes = 1$). In unregularized regression this causes singularity, but the scikit-learn Logistic Regression L2 penalty ($C=1.0$) handles it smoothly. Standardized coefficients for both reflect equal and opposite negligible weights ($\pm 0.0074$).

### D. Dataset Nature & Scientific Limitations
* **Provenance**: The dataset contains 10,000 records across 100 college IDs (`CLG0001` - `CLG0100`).
* **Statistical Characteristics**: Zero missing values and zero duplicates. Features like IQ, CGPA, and Communication Skills exhibit smooth, statistically generated distributions with clear correlations, while `Internship_Experience` (16.77% vs 16.32% placement rate) and `Academic_Performance` exhibit near-zero correlation with placement.
* **Disclosure**: This is an educational benchmark/synthetic dataset. High evaluation scores demonstrate algorithm learning capability but do not guarantee real-world campus recruitment outcomes.

---

## 📊 3. Model Evaluation & Ground-Truth Performance

All metrics below are computed from the verified held-out test set (2,000 students) and 5-fold cross-validation on training data.

### Overall Performance Summary

| Metric | Default Threshold (`0.50`) | CV-Tuned F1-Optimal (`0.36`) | Conservative (`0.80`) | Random Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **90.35%** | 90.25% | 88.10% | 83.41% (Majority) |
| **Precision** | **77.69%** | 70.09% | 90.52% | 16.59% |
| **Recall (Sensitivity)** | **58.73%** | **71.69%** | 31.63% | 100.0% (Trivial) |
| **F1-Score** | **0.6690** | **0.7088** | 0.4688 | 0.2846 |
| **Specificity (TNR)** | **96.64%** | 93.94% | 99.34% | 0.00% |
| **ROC-AUC** | **0.9451** | **0.9451** | **0.9451** | 0.5000 |
| **PR-AUC (Avg Precision)** | **0.7793** | **0.7793** | **0.7793** | 0.1659 |

### Held-Out Confusion Matrix Breakdown (Test Set, n = 2,000)

#### At Default Threshold (`0.50`):
```text
                     Predicted: Not Placed (0)    Predicted: Placed (1)
Actual: Not Placed (0)         1612 (TN)                   56 (FP)
Actual: Placed (1)              137 (FN)                  195 (TP)
```
* **True Negatives (TN = 1612, 80.6%)**: Correctly identified unplaced candidates.
* **False Positives (FP = 56, 2.8%)**: Type I Error — Students predicted as placed who were not.
* **False Negatives (FN = 137, 6.85%)**: Type II Error — Capable students overlooked by the model.
* **True Positives (TP = 195, 9.75%)**: Correctly identified placed candidates.

#### At CV-Tuned Threshold (`0.36`):
```text
                     Predicted: Not Placed (0)    Predicted: Placed (1)
Actual: Not Placed (0)         1567 (TN)                  101 (FP)
Actual: Placed (1)               94 (FN)                  238 (TP)
```
* Boosts Recall from **58.73% to 71.69%** (detecting 43 more placed students) with balanced F1 of **0.7088**.

---

## 📈 4. Feature Importance & Model Coefficients

Because all features are standardized using `StandardScaler` ($\mu=0, \sigma=1$), the model coefficients ($\beta_i$) represent the change in log-odds of placement per **one standard deviation** increase in that feature:

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{i=1}^n \beta_i \left(\frac{x_i - \mu_i}{\sigma_i}\right)$$

| Feature Name | Display Name | Standardized $\beta_i$ | Odds Ratio ($e^{\beta_i}$) | Impact Category |
| :--- | :--- | :---: | :---: | :--- |
| `Communication_Skills` | Communication Skills Score | **+1.8756** | **6.52** | Primary Positive Driver |
| `IQ` | Student IQ Score | **+1.6252** | **5.08** | Strong Positive Driver |
| `CGPA` | Cumulative GPA | **+1.4834** | **4.41** | Strong Academic Driver |
| `Projects_Completed` | Number of Projects | **+1.1784** | **3.25** | Practical Skills Driver |
| `Prev_Sem_Result` | Previous Semester GPA | **+0.3339** | **1.40** | Moderate Positive Driver |
| `Internship_Experience_Yes` | Internship (Yes) | **+0.0074** | 1.01 | Negligible in this dataset |
| `Internship_Experience_No` | Internship (No) | **-0.0074** | 0.99 | Negligible in this dataset |
| `Academic_Performance` | Faculty Academic Rating | **-0.0108** | 0.99 | Negligible in this dataset |
| `Extra_Curricular_Score` | Extra-Curricular Score | **-0.0440** | 0.96 | Negligible in this dataset |
| **Model Intercept ($\beta_0$)** | — | **-3.6564** | 0.026 | Reflects ~16.6% base rate |

---

## 🖥️ 5. Dashboard Architecture & Visualizations

The web application is structured into five logical, graph-rich tabs:

1. **📊 Executive Overview**:
   - 6 KPI metric cards (Total Records, Imbalance %, Test Accuracy, F1-score, ROC-AUC, PR-AUC).
   - Interactive Plotly Donut Chart of class distribution.
   - Comprehensive Metric Summary Bar Chart comparing test-set scores against baselines.
   - Callout box explaining the Accuracy Paradox in imbalanced datasets.

2. **🔍 Exploratory Data Analysis (EDA)**:
   - Interactive feature distribution (Histogram + Box plot overlay) segmented by placement outcome.
   - Feature-vs-Target grouped box plots showing median shifts.
   - Full 8x8 Pearson Correlation Heatmap with interactive tooltips.
   - Categorical analysis of Internship Experience vs placement rates.
   - Expandable dataset statistical summary table (mean, std, percentiles).

3. **🎯 Confusion Matrix & Model Performance (High Priority)**:
   - Real-time decision threshold slider (0.10 to 0.90) and preset buttons (Default 0.50, F1-Optimal 0.36, Conservative 0.80).
   - Interactive Confusion Matrix Heatmap displaying exact counts and cell percentages.
   - Detailed breakdown cards for TN, FP, FN, and TP with campus recruitment implications.
   - Interactive ROC Curve with operating point marker and diagonal baseline (AUC = 0.9451).
   - Precision-Recall Curve with baseline and operating point marker (AP = 0.7793).
   - Dynamic Threshold Sensitivity line plot (Precision, Recall, F1, Accuracy vs Threshold).
   - Scikit-Learn Classification Report table.

4. **📈 Feature Importance & Interpretability**:
   - Standardized coefficient horizontal bar chart with color-coded positive/negative contributions.
   - Odds Ratio table per standard deviation.
   - Mathematical formulation and viva explanation of logit transformation and feature scaling.

5. **🔮 Student Placement Predictor**:
   - Two-column input layout for academic and practical attributes.
   - 4 quick-test preset profiles: "Star Candidate", "Average Candidate", "Needs Improvement", "High CGPA/Low Comm".
   - Real-time placement outcome card with circular probability gauge.
   - Feature contribution waterfall chart showing how each specific student score shifted the prediction.
   - Actionable student guidance and viva disclaimer.

---

## 🚀 6. Setup and Execution Instructions

### Prerequisites
- macOS / Linux / Windows
- Python 3.9+ (Python 3.11 recommended)

### Step 1: Clone or Navigate to Directory
```bash
cd "placement project work/placementmlproject"
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🎓 7. Comprehensive Viva Voce Preparation Guide

Below are ten essential viva questions formulated around this exact implementation:

### Q1. Why did you choose Logistic Regression instead of a complex ensemble like Random Forest or XGBoost?
> **Answer**: Placement prediction here is a binary classification problem where interpretability is paramount. Logistic Regression outputs calibrated posterior probabilities $P(Y=1|X) = \sigma(z)$ rather than just hard labels. It allows direct inspection of feature weights through standardized coefficients and odds ratios, ensuring the decision process is completely transparent and defensible during academic audit. Furthermore, with an ROC-AUC of 0.9451, linear decision boundaries with regularized L2 loss already capture the primary variance in this dataset without risk of overfitting.

### Q2. How did you split the dataset and why did you use stratification?
> **Answer**: We used an 80/20 train/test split (8,000 training, 2,000 test) with `random_state=42` for exact reproducibility. Stratification (`stratify=y`) was essential because the target class is imbalanced (83.41% Not Placed vs 16.59% Placed). Without stratification, random sampling could yield disproportionately fewer positive cases in the test set, introducing sampling bias and corrupting evaluation metrics.

### Q3. Why is Feature Scaling necessary for Logistic Regression, and how did you prevent data leakage?
> **Answer**: Logistic Regression optimizes weights using gradient-based solvers with an L2 regularization penalty ($\frac{1}{2}\lambda \|\mathbf{w}\|^2$). If features have different scales (e.g., IQ ranges from 40 to 160 while CGPA ranges from 0 to 10), unscaled weights are penalized unequally, distorting feature importance. StandardScaler normalizes each feature to $\mu=0, \sigma=1$. To prevent data leakage, `StandardScaler` was fitted strictly on `X_train` (`fit_transform`) and subsequently used only to `transform` `X_test` and inference data, ensuring test statistics never leaked into training.

### Q4. What is a Classification Decision Threshold, and how did you select it?
> **Answer**: The decision threshold $\theta$ is the probability cutoff where $\hat{y} = 1$ if $P(Y=1|X) \ge \theta$, otherwise $0$. The default threshold of 0.50 assumes equal cost for false positives and false negatives. In imbalanced datasets, 0.50 can result in suboptimal recall. We tuned the threshold strictly on training data using 5-fold Stratified Cross-Validation to maximize F1-score, identifying $\theta = 0.36$. This boosted test recall from 58.73% to 71.69% while keeping the final test set untouched during tuning.

### Q5. Walk me through the Confusion Matrix and its four quadrants.
> **Answer**: On our held-out test set of 2,000 students at default threshold 0.50:
> - **True Negatives (TN = 1612)**: Correctly predicted unplaced students.
> - **False Positives (FP = 56, Type I Error)**: Unplaced students falsely predicted as placed.
> - **False Negatives (FN = 137, Type II Error)**: Placed students missed by the model.
> - **True Positives (TP = 195)**: Correctly predicted placed students.

### Q6. Why can Accuracy be misleading for this dataset? Which metrics did you prioritize?
> **Answer**: Accuracy measures $\frac{TP + TN}{\text{Total}}$. In our dataset, 83.41% of students are Not Placed. A naive dummy classifier that predicts "Not Placed" for every student would achieve an 83.41% accuracy score despite having a recall of 0.0%. Therefore, we prioritize **Precision** (77.69%), **Recall** (58.73%), **F1-Score** (0.6690 at 0.50 / 0.7088 at 0.36), and **ROC-AUC** (0.9451).

### Q7. What do ROC-AUC and Precision-Recall (PR-AUC) curves reveal?
> **Answer**: 
> - The **ROC Curve** plots True Positive Rate vs False Positive Rate across all thresholds. Our ROC-AUC is **0.9451**, indicating a 94.5% probability that the model ranks a randomly chosen placed student higher than an unplaced one.
> - The **PR Curve** plots Precision vs Recall. In imbalanced problems, PR curves are more informative because they focus on the minority class without being inflated by a large number of True Negatives. Our model achieves an Average Precision of **0.7793** against a random baseline of 0.1659.

### Q8. How do you interpret the model coefficients?
> **Answer**: Because inputs are standard-scaled, the coefficient represents the change in log-odds per standard deviation change. The exponential $e^{\beta_i}$ gives the odds ratio:
> - `Communication_Skills` ($\beta = +1.88, e^\beta = 6.52$): A $1\sigma$ increase multiplies the odds of placement by 6.52.
> - `IQ` ($\beta = +1.63, e^\beta = 5.08$) and `CGPA` ($\beta = +1.48, e^\beta = 4.41$) are the next strongest drivers.
> - `Internship_Experience` and `Academic_Performance` have coefficients near 0, indicating negligible predictive value in this dataset.

### Q9. What is Data Leakage and how did your pipeline guard against it?
> **Answer**: Data leakage occurs when information from outside the training dataset is used to create or tune the model. Common forms include scaling the full dataset before splitting, imputing missing values globally, or tuning hyperparameters/thresholds on the final test set. We strictly guarded against leakage by splitting the raw dataset first, fitting the scaler and model only on `X_train`, and conducting all threshold tuning inside a 5-fold cross-validation loop on `X_train`.

### Q10. What are the ethical and real-world limitations of this placement prediction model?
> **Answer**: 
> 1. **Synthetic Data Artifacts**: The dataset is a benchmark sample where certain variables (like internship experience) show zero correlation, which diverges from real-world recruitment dynamics.
> 2. **Algorithmic Bias & Self-Fulfilling Prophecies**: Using ML predictions to allocate college placement training could deny opportunities to students flagged as "unlikely to be placed", creating a feedback loop.
> 3. **Non-Causal Associations**: Logistic regression identifies statistical correlations, not causation; scoring high on communication skills correlates with placement but does not guarantee company interview selection without aptitude test clearing and domain interview performance.

---

## 📄 License & Attribution
Developed for academic demonstration and machine learning audit purposes in B.Tech Computer Science & Engineering.
