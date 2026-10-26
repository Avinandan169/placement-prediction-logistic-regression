"""
ml_engine.py — Core Machine Learning & Statistical Evaluation Engine
Placement Prediction System

Provides reproducible data preprocessing, rigorous train/test evaluation,
cross-validation threshold tuning, and model interpretability utilities.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

FEATURE_NAMES = [
    'IQ',
    'Prev_Sem_Result',
    'CGPA',
    'Academic_Performance',
    'Extra_Curricular_Score',
    'Communication_Skills',
    'Projects_Completed',
    'Internship_Experience_No',
    'Internship_Experience_Yes'
]

FEATURE_DISPLAY_NAMES = {
    'IQ': 'IQ Score',
    'Prev_Sem_Result': 'Previous Semester GPA',
    'CGPA': 'Cumulative GPA (CGPA)',
    'Academic_Performance': 'Academic Performance Rating',
    'Extra_Curricular_Score': 'Extra-Curricular Score',
    'Communication_Skills': 'Communication Skills Score',
    'Projects_Completed': 'Projects Completed',
    'Internship_Experience_No': 'Internship (No)',
    'Internship_Experience_Yes': 'Internship (Yes)'
}


def load_raw_dataset(csv_path="placement.csv"):
    """Load the original CSV dataset."""
    return pd.read_csv(csv_path)


def preprocess_data(df_raw):
    """
    Preprocess data consistently:
    1. Drop College_ID (arbitrary identifier).
    2. Map target 'Placement': 'No' -> 0, 'Yes' -> 1.
    3. One-hot encode Internship_Experience.
    4. Ensure column ordering matches FEATURE_NAMES.
    """
    df = df_raw.drop(columns=['College_ID']).copy()
    df['Placement'] = df['Placement'].map({'No': 0, 'Yes': 1})
    df_encoded = pd.get_dummies(df, columns=['Internship_Experience'], dtype=int)
    
    # Ensure all required features are present
    for col in FEATURE_NAMES:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    X = df_encoded[FEATURE_NAMES]
    y = df_encoded['Placement']
    return df, X, y


def get_stratified_split(X, y, test_size=0.2, random_state=42):
    """Create reproducible stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def compute_cv_threshold(X_train, y_train, n_splits=5, random_state=42):
    """
    Tune threshold strictly using 5-Fold Stratified Cross-Validation on X_train.
    Prevents data leakage into the test set.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y_train))
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)
        
        fold_model = LogisticRegression(max_iter=1000, random_state=random_state)
        fold_model.fit(X_tr_sc, y_tr)
        oof_probs[val_idx] = fold_model.predict_proba(X_val_sc)[:, 1]
        
    cv_roc_auc = roc_auc_score(y_train, oof_probs)
    
    # Grid search for F1-maximizing threshold
    threshold_grid = np.linspace(0.10, 0.90, 81)
    best_f1 = -1.0
    best_th = 0.50
    
    for th in threshold_grid:
        preds = (oof_probs >= th).astype(int)
        score = f1_score(y_train, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = th
            
    return {
        'best_threshold': round(best_th, 2),
        'best_f1': best_f1,
        'cv_roc_auc': cv_roc_auc,
        'oof_probs': oof_probs
    }


def train_production_pipeline(X_train, y_train):
    """
    Fits StandardScaler and LogisticRegression on X_train.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return scaler, model


def evaluate_predictions_at_threshold(y_true, y_probs, threshold=0.50):
    """
    Calculates detailed classification metrics and confusion matrix
    at a specific decision threshold.
    """
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = rec  # Recall is sensitivity
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value
    
    report_dict = classification_report(
        y_true, y_pred, target_names=['Not Placed', 'Placed'],
        output_dict=True, zero_division=0
    )
    
    return {
        'threshold': threshold,
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'total': len(y_true),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'npv': npv,
        'classification_report': report_dict,
        'y_pred': y_pred
    }


def compute_evaluation_suite(model, scaler, X_test, y_test):
    """
    Computes complete held-out evaluation:
    - Probability estimates
    - ROC curve & AUC
    - Precision-Recall curve & Average Precision
    - Threshold sensitivity sweep
    """
    X_test_scaled = scaler.transform(X_test)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    # Threshold sweep
    thresholds_sweep = np.linspace(0.05, 0.95, 37)
    sweep_records = []
    for th in thresholds_sweep:
        yp = (y_probs >= th).astype(int)
        c = confusion_matrix(y_test, yp, labels=[0, 1])
        sweep_records.append({
            'Threshold': th,
            'Accuracy': accuracy_score(y_test, yp),
            'Precision': precision_score(y_test, yp, zero_division=0),
            'Recall': recall_score(y_test, yp, zero_division=0),
            'F1-Score': f1_score(y_test, yp, zero_division=0),
            'True Positives': c[1, 1],
            'False Negatives': c[1, 0],
            'False Positives': c[0, 1],
            'True Negatives': c[0, 0]
        })
    sweep_df = pd.DataFrame(sweep_records)
    
    return {
        'y_probs': y_probs,
        'roc': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': roc_thresholds,
            'auc': roc_auc
        },
        'pr': {
            'precisions': precisions,
            'recalls': recalls,
            'thresholds': pr_thresholds,
            'auc': pr_auc,
            'baseline': float(np.mean(y_test))
        },
        'sweep_df': sweep_df
    }


def get_feature_coefficients(model, scaler):
    """
    Returns standardized Logistic Regression coefficients with interpretation.
    """
    weights = model.coef_[0]
    records = []
    for feat, w, mean, scale in zip(FEATURE_NAMES, weights, scaler.mean_, scaler.scale_):
        impact = "Strong Positive" if w > 1.0 else (
            "Moderate Positive" if w > 0.2 else (
                "Neutral / Negligible" if abs(w) <= 0.2 else "Negative"
            )
        )
        records.append({
            'Feature': feat,
            'Display Name': FEATURE_DISPLAY_NAMES.get(feat, feat),
            'Standardized Coefficient': float(w),
            'Abs Coefficient': abs(float(w)),
            'Odds Ratio (per 1 SD)': float(np.exp(w)),
            'Impact Direction': "Favors Placement" if w > 0 else "Reduces Placement Odds",
            'Impact Category': impact,
            'Training Mean': float(mean),
            'Training Std': float(scale)
        })
    df_coef = pd.DataFrame(records).sort_values(by='Abs Coefficient', ascending=False)
    return df_coef


def predict_single_student(model, scaler, input_dict, threshold=0.50):
    """
    Predict placement for a single student.
    input_dict must contain all raw student fields.
    """
    internship = input_dict.get('Internship_Experience', 'No')
    internship_no = 1 if internship == 'No' else 0
    internship_yes = 1 if internship == 'Yes' else 0
    
    row = pd.DataFrame([{
        'IQ': float(input_dict['IQ']),
        'Prev_Sem_Result': float(input_dict['Prev_Sem_Result']),
        'CGPA': float(input_dict['CGPA']),
        'Academic_Performance': float(input_dict['Academic_Performance']),
        'Extra_Curricular_Score': float(input_dict['Extra_Curricular_Score']),
        'Communication_Skills': float(input_dict['Communication_Skills']),
        'Projects_Completed': float(input_dict['Projects_Completed']),
        'Internship_Experience_No': internship_no,
        'Internship_Experience_Yes': internship_yes
    }])[FEATURE_NAMES]
    
    row_scaled = scaler.transform(row)
    probability = float(model.predict_proba(row_scaled)[0, 1])
    is_placed = int(probability >= threshold)
    
    # Calculate feature contributions (scaled value * coefficient)
    contributions = []
    for feat, x_sc, coef in zip(FEATURE_NAMES, row_scaled[0], model.coef_[0]):
        contributions.append({
            'Feature': feat,
            'Display Name': FEATURE_DISPLAY_NAMES.get(feat, feat),
            'Raw Value': row[feat].iloc[0],
            'Scaled Z-Score': round(x_sc, 2),
            'Contribution': round(x_sc * coef, 3)
        })
    contrib_df = pd.DataFrame(contributions).sort_values(by='Contribution', ascending=False)
    
    # Qualitative insights
    strengths = []
    improvements = []
    for _, item in contrib_df.iterrows():
        if item['Contribution'] > 0.5:
            strengths.append(f"{item['Display Name']} (+{item['Contribution']:.2f})")
        elif item['Contribution'] < -0.3:
            improvements.append(f"{item['Display Name']} ({item['Contribution']:.2f})")
            
    return {
        'probability': probability,
        'is_placed': is_placed,
        'threshold': threshold,
        'contrib_df': contrib_df,
        'strengths': strengths,
        'improvements': improvements,
        'scaled_features': row_scaled[0]
    }
