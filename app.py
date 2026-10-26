"""
app.py — Placement Prediction System
An interactive machine learning application for student placement prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

import ml_engine

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal CSS for clean card styling and whitespace
st.markdown("""
<style>
    /* Metric Card Styling */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    /* Result Cards */
    .result-box-placed {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        margin-top: 12px;
    }
    .result-box-unplaced {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        margin-top: 12px;
    }
    .result-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
    }
    .result-sub {
        font-size: 1rem;
        margin-top: 6px;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data & Model Pipeline Caching
# ---------------------------------------------------------
@st.cache_data
def get_dataset():
    df_raw = ml_engine.load_raw_dataset("placement.csv")
    df_clean, X, y = ml_engine.preprocess_data(df_raw)
    return df_raw, df_clean, X, y


@st.cache_resource
def get_pipeline(X, y):
    X_train, X_test, y_train, y_test = ml_engine.get_stratified_split(X, y)
    scaler, model = ml_engine.train_production_pipeline(X_train, y_train)
    eval_suite = ml_engine.compute_evaluation_suite(model, scaler, X_test, y_test)
    coef_df = ml_engine.get_feature_coefficients(model, scaler)
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'model': model,
        'eval_suite': eval_suite,
        'coef_df': coef_df
    }


df_raw, df_clean, X, y = get_dataset()
pipeline = get_pipeline(X, y)

X_test = pipeline['X_test']
y_test = pipeline['y_test']
scaler = pipeline['scaler']
model = pipeline['model']
eval_suite = pipeline['eval_suite']
coef_df = pipeline['coef_df']

# Default threshold for production evaluation
DEFAULT_THRESHOLD = 0.50
eval_metrics = ml_engine.evaluate_predictions_at_threshold(
    y_test, eval_suite['y_probs'], threshold=DEFAULT_THRESHOLD
)


# ---------------------------------------------------------
# Compact Header
# ---------------------------------------------------------
st.title("Placement Prediction System")
st.caption("An interactive machine learning application for student placement prediction.")

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Minimal Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("About")
    st.caption("Student Placement Prediction System using Logistic Regression.")
    st.markdown("---")
    st.caption("**Model**: Logistic Regression (L2)")
    st.caption("**Features**: 9 standardized attributes")
    st.caption("**Dataset**: 10,000 student records")


# ---------------------------------------------------------
# Main Navigation (3 Simple Sections)
# ---------------------------------------------------------
tab_overview, tab_analysis, tab_predict = st.tabs([
    "Overview",
    "Model Analysis",
    "Predict Placement"
])


# =========================================================
# 1. OVERVIEW
# =========================================================
with tab_overview:
    # 4 Compact KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_records = len(df_raw)
    placed_total = int(np.sum(df_clean['Placement']))
    placement_rate = (placed_total / total_records) * 100
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Records</div>
            <div class="kpi-value">{total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Placement Rate</div>
            <div class="kpi-value">{placement_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Test Accuracy</div>
            <div class="kpi-value">{eval_metrics['accuracy'] * 100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Test F1-Score</div>
            <div class="kpi-value">{eval_metrics['f1']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    
    # Overview Visualizations (2 Charts Side-by-Side)
    chart_c1, chart_c2 = st.columns(2)
    
    with chart_c1:
        st.markdown("##### Placement Distribution")
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Not Placed (0)', 'Placed (1)'],
            values=[total_records - placed_total, placed_total],
            hole=0.55,
            marker_colors=['#94A3B8', '#10B981'],
            textinfo='label+percent',
            hovertemplate="<b>%{label}</b>: %{value:,} students (%{percent})<extra></extra>"
        )])
        fig_donut.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False
        )
        st.plotly_chart(fig_donut, width="stretch")
        st.caption("Distribution of the target variable showing 83.4% unplaced and 16.6% placed students.")
        
    with chart_c2:
        st.markdown("##### CGPA Distribution by Placement Outcome")
        fig_cgpa = px.histogram(
            df_clean,
            x='CGPA',
            color=df_clean['Placement'].map({0: 'Not Placed', 1: 'Placed'}),
            barmode='overlay',
            opacity=0.75,
            color_discrete_map={'Not Placed': '#94A3B8', 'Placed': '#10B981'},
            labels={'color': 'Outcome'}
        )
        fig_cgpa.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="CGPA",
            yaxis_title="Student Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cgpa, width="stretch")
        st.caption("Comparison showing that placed students consistently concentrate in higher CGPA ranges.")


# =========================================================
# 2. MODEL ANALYSIS
# =========================================================
with tab_analysis:
    st.markdown("##### Model Evaluation & Performance Analysis")
    st.caption("Evaluation results computed on the held-out test set of 2,000 students.")
    
    # 1. Confusion Matrix & 2. ROC Curve side-by-side
    row1_c1, row1_c2 = st.columns(2)
    
    with row1_c1:
        st.markdown("##### 1. Confusion Matrix")
        cm = eval_metrics['confusion_matrix']
        total_test = eval_metrics['total']
        
        annot_text = [
            [f"<b>TN: {cm[0,0]:,}</b><br>({cm[0,0]/total_test*100:.1f}%)",
             f"<b>FP: {cm[0,1]:,}</b><br>({cm[0,1]/total_test*100:.1f}%)"],
            [f"<b>FN: {cm[1,0]:,}</b><br>({cm[1,0]/total_test*100:.1f}%)",
             f"<b>TP: {cm[1,1]:,}</b><br>({cm[1,1]/total_test*100:.1f}%)"]
        ]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted: Not Placed', 'Predicted: Placed'],
            y=['Actual: Not Placed', 'Actual: Placed'],
            text=annot_text,
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=False,
            hoverinfo='none'
        ))
        fig_cm.update_layout(
            height=320,
            margin=dict(t=10, b=20, l=20, r=20),
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig_cm, width="stretch")
        st.caption("Displays correct and incorrect classifications on 2,000 held-out test records at the default 0.50 threshold.")
        
    with row1_c2:
        st.markdown("##### 2. Receiver Operating Characteristic (ROC) Curve")
        roc_data = eval_suite['roc']
        fig_roc = go.Figure()
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='#CBD5E1', dash='dash'),
            name='Random Guessing'
        ))
        fig_roc.add_trace(go.Scatter(
            x=roc_data['fpr'], y=roc_data['tpr'],
            mode='lines',
            line=dict(color='#0284C7', width=2.5),
            name=f"Logistic Regression (AUC = {roc_data['auc']:.4f})",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
        ))
        fig_roc.update_layout(
            height=320,
            margin=dict(t=10, b=20, l=20, r=20),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_roc, width="stretch")
        st.caption(f"Measures classification ability across all thresholds (ROC-AUC = {roc_data['auc']:.4f}).")
        
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    # 3. Correlation Heatmap & 4. Feature Coefficients side-by-side
    row2_c1, row2_c2 = st.columns(2)
    
    with row2_c1:
        st.markdown("##### 3. Correlation Heatmap")
        corr_cols = ['CGPA', 'Prev_Sem_Result', 'IQ', 'Communication_Skills', 'Projects_Completed', 'Extra_Curricular_Score', 'Academic_Performance', 'Placement']
        corr_matrix = df_clean[corr_cols].corr().round(2)
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            zmin=-0.1, zmax=1.0
        )
        fig_corr.update_layout(
            height=340,
            margin=dict(t=10, b=20, l=20, r=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_corr, width="stretch")
        st.caption("Pearson correlation between numerical features and placement outcome. Communication Skills, CGPA, and IQ show the highest correlation.")
        
    with row2_c2:
        st.markdown("##### 4. Model Coefficients (Feature Importance)")
        fig_coef = px.bar(
            coef_df,
            x='Standardized Coefficient',
            y='Display Name',
            orientation='h',
            color='Standardized Coefficient',
            color_continuous_scale=['#EF4444', '#E2E8F0', '#10B981'],
            color_continuous_midpoint=0,
            text='Standardized Coefficient'
        )
        fig_coef.update_layout(
            height=340,
            margin=dict(t=10, b=20, l=20, r=20),
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Standardized Coefficient",
            yaxis_title="",
            coloraxis_showscale=False
        )
        fig_coef.update_traces(texttemplate='%{x:+.2f}', textposition='outside')
        st.plotly_chart(fig_coef, width="stretch")
        st.caption("Fitted Logistic Regression coefficients. Features with positive values increase the log-odds of placement.")
        
    # Optional compact classification report
    with st.expander("View Detailed Classification Report"):
        rep = eval_metrics['classification_report']
        rep_df = pd.DataFrame({
            'Class': ['Not Placed (0)', 'Placed (1)', 'Macro Average', 'Weighted Average'],
            'Precision': [rep['Not Placed']['precision'], rep['Placed']['precision'], rep['macro avg']['precision'], rep['weighted avg']['precision']],
            'Recall': [rep['Not Placed']['recall'], rep['Placed']['recall'], rep['macro avg']['recall'], rep['weighted avg']['recall']],
            'F1-Score': [rep['Not Placed']['f1-score'], rep['Placed']['f1-score'], rep['macro avg']['f1-score'], rep['weighted avg']['f1-score']],
            'Support': [rep['Not Placed']['support'], rep['Placed']['support'], rep['macro avg']['support'], rep['weighted avg']['support']]
        })
        st.dataframe(rep_df.round(3), width="stretch", hide_index=True)


# =========================================================
# 3. PREDICT PLACEMENT
# =========================================================
with tab_predict:
    st.markdown("##### Enter Student Information")
    st.caption("Fill in student details below and click Predict Placement to generate an outcome.")
    
    # Session state for form reset
    if 'form_key' not in st.session_state:
        st.session_state.form_key = 0
        
    # Form layout with 2 columns
    with st.form(f"prediction_form_{st.session_state.form_key}"):
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            val_iq = st.number_input("IQ Score", min_value=50, max_value=160, value=100, step=1, help="Student IQ score (typical range: 70–140)")
            val_prev_sem = st.number_input("Previous Semester Result", min_value=0.0, max_value=10.0, value=7.50, step=0.10, format="%.2f")
            val_cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.50, step=0.10, format="%.2f")
            val_acad = st.number_input("Academic Performance (0–10)", min_value=0, max_value=10, value=7, step=1)
            
        with col_f2:
            val_comm = st.number_input("Communication Skills (0–10)", min_value=0, max_value=10, value=6, step=1)
            val_projects = st.number_input("Projects Completed", min_value=0, max_value=10, value=2, step=1)
            val_extra = st.number_input("Extra-Curricular Score (0–10)", min_value=0, max_value=10, value=5, step=1)
            val_internship = st.selectbox("Internship Experience", ["No", "Yes"], index=0)
            
        btn_c1, btn_c2 = st.columns([1, 4])
        with btn_c1:
            submitted = st.form_submit_button("Predict Placement", type="primary", width="stretch")
            
    # Reset button outside form
    if st.button("Reset Form"):
        st.session_state.form_key += 1
        st.rerun()
        
    # Execute prediction when form is submitted
    if submitted:
        student_data = {
            'IQ': val_iq,
            'Prev_Sem_Result': val_prev_sem,
            'CGPA': val_cgpa,
            'Academic_Performance': val_acad,
            'Extra_Curricular_Score': val_extra,
            'Communication_Skills': val_comm,
            'Projects_Completed': val_projects,
            'Internship_Experience': val_internship
        }
        
        prediction_result = ml_engine.predict_single_student(
            model, scaler, student_data, threshold=DEFAULT_THRESHOLD
        )
        
        prob = prediction_result['probability']
        prob_pct = prob * 100
        is_placed = prediction_result['is_placed']
        
        if is_placed == 1:
            st.markdown(f"""
            <div class="result-box-placed">
                <p class="result-title" style="color: #166534;">Likely to be Placed</p>
                <p class="result-sub">Estimated Placement Probability: <b>{prob_pct:.1f}%</b></p>
                <small style="color: #64748B;">This prediction is a model-estimated probability based on historical student patterns, not an absolute guarantee.</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-unplaced">
                <p class="result-title" style="color: #991B1B;">Unlikely to be Placed</p>
                <p class="result-sub">Estimated Placement Probability: <b>{prob_pct:.1f}%</b></p>
                <small style="color: #64748B;">This prediction is a model-estimated probability based on historical student patterns, not an absolute guarantee.</small>
            </div>
            """, unsafe_allow_html=True)
