import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import pickle

# Force wide layout structure for data density
st.set_page_config(layout="wide", page_title="The Black-Box Auditor")

st.title("🎛️ The Black-Box Auditor")
st.caption("Enterprise Model Governance & Explainable AI (XAI) Live Diagnostic Dashboard")
st.divider()

# Cache data loading pipelines to keep rendering fast
#@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    explainer = shap.TreeExplainer(model)
    return model, X_test, y_test, explainer

try:
    model, X_test, y_test, explainer = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model artifacts missing. Run `python train_model.py` in your terminal first.")
    st.stop()

# Multi-view navigation tabs
tab1, tab2, tab3 = st.tabs([
    "🔮 Real-Time Explainability (SHAP)", 
    "⚖️ Ethical AI & Fairness Tracking", 
    "🇪🇺 EU AI Act Regulatory Compliance"
])

# ─── TAB 1: EXPLAINABLE AI & REAL-TIME AUDITING ───
with tab1:
    st.header("Model Transparency Diagnostics")
    
    # ── REAL-TIME INTERACTIVE SIMULATOR PLAYGROUND ──
    st.subheader("⚡ Live Risk Evaluation Simulator")
    st.write("Modify properties manually below to witness the model calculate risk and draw individual explanations in real time.")
    
    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
    with sim_col1:
        input_age = st.slider("Client Age", 18, 90, 35)
    with sim_col2:
        input_amount = st.number_input("Requested Loan Amount (DM)", min_value=250, max_value=20000, value=3000, step=250)
    with sim_col3:
        input_duration = st.slider("Loan Duration Term (Months)", 4, 72, 24)
    with sim_col4:
        input_gender = st.selectbox("Demographic Profile Group", options=["Male Profile", "Female Profile"])

    # Package input array exactly to model specifications
    live_record = pd.DataFrame([{
        'Age': float(input_age),
        'Credit_Amount': float(input_amount),
        'Duration_Months': float(input_duration),
        'Gender_Male': 1.0 if "Male" in input_gender else 0.0
    }])
    
    # Execute calculations dynamically on interaction loop
    live_pred = model.predict(live_record)[0]
    live_proba = model.predict_proba(live_record)[0][1]
    
    p_col1, p_col2 = st.columns([1, 2])
    with p_col1:
        st.write("#### Decision Result:")
        if live_pred == 1:
            st.success(f"✅ CREDIT APPROVED\n\n(Confidence Score: {live_proba:.1%})")
        else:
            st.error(f"🚨 CREDIT DENIED\n\n(Confidence Score: {(1-live_proba):.1%})")
            
    with p_col2:
        st.write("#### Dynamic Local Feature Attribution Plot:")
        live_shap = explainer(live_record)
        fig_live, ax_live = plt.subplots(figsize=(7, 2.5))
        shap.plots.waterfall(live_shap[0, :, 1], show=False)
        plt.tight_layout()
        st.pyplot(fig_live)
        
    st.divider()
    
    # ── GLOBAL ATTRIBUTION EVALUATION ──
    col1, col2 = st.columns(2)
    
    @st.cache_data
    def get_global_shap_sample():
        return explainer(X_test.head(40))
        
    with col1:
        st.subheader("Global Feature Importance (Real Pool)")
        st.write("Identifies weights influencing system behaviors universally across the whole testing database partition.")
        
        with st.spinner("Processing global matrix layers..."):
            global_shap = get_global_shap_sample()
            fig, ax = plt.subplots(figsize=(6, 4))
            shap.summary_plot(global_shap[:, :, 1], X_test.head(40), plot_type="bar", show=False)
            plt.tight_layout()
            st.pyplot(fig)
        
    with col2:
        st.subheader("Historical Batch Record Explorer")
        row_idx = st.number_input("Select Historical Entry Index Row to Parse:", min_value=0, max_value=len(X_test)-1, value=5)
        st.write(f"**Target Features Matrix File for Index {row_idx}:**")
        st.dataframe(X_test.iloc[[row_idx]])
        
        with st.spinner(f"Computing context explanations..."):
            single_row = X_test.iloc[[row_idx]]
            local_shap = explainer(single_row)
            
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            shap.plots.waterfall(local_shap[0, :, 1], show=False)
            plt.tight_layout()
            st.pyplot(fig2)

# ─── TAB 2: ETHICAL AI & FAIRNESS ───
with tab2:
    st.header("Fairness & Disparate Impact Auditing")
    st.write("Evaluating systemic performance variance between demographic groups extracted from historical profiles.")
    
    predictions = model.predict(X_test)
    eval_df = X_test.copy()
    eval_df['Predictions'] = predictions
    
    male_selection_rate = eval_df[eval_df['Gender_Male'] == 1]['Predictions'].mean()
    female_selection_rate = eval_df[eval_df['Gender_Male'] == 0]['Predictions'].mean()
    disparate_impact_ratio = female_selection_rate / (male_selection_rate if male_selection_rate > 0 else 1)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Male Selection Acceptance Rate", f"{male_selection_rate:.2%}")
    m2.metric("Female Selection Acceptance Rate", f"{female_selection_rate:.2%}")
    m3.metric("Disparate Impact Ratio Index", f"{disparate_impact_ratio:.2f}", 
              delta="Compliant (>= 0.80)" if disparate_impact_ratio >= 0.80 else "Action Required (Bias Detected)")

# ─── TAB 3: REGULATORY COMPLIANCE ───
with tab3:
    st.header("EU AI Act Compliance Mapping Engine")
    st.info("Under Chapter II (High-Risk AI Systems), automated evaluation tools evaluating human credit risks or resource capabilities are subject to strict statutory verification parameters.")
    
    st.write("### Technical Obligation Audit Checklist")
    q1 = st.checkbox("Art 9: Continuous Risk Management System implemented throughout structural life cycles?")
    q2 = st.checkbox("Art 10: Data Governance validation protocols strictly run for bias checking?")
    q3 = st.checkbox("Art 12: Automated logging mechanisms configured for operational traceability metrics?")
    q4 = st.checkbox("Art 14: Human-in-the-loop technological oversight configuration panels deployed?")
    
    if st.button("Generate Compliance Audit Certification"):
        if q1 and q2 and q3 and q4:
            st.success("🎉 PASSED: System deployment pipeline satisfies technical audit compliance mandates under the EU AI Act classification framework.")
        else:
            st.error("🚨 NON-COMPLIANT: Vital governance processes are incomplete. Production deployment state locked.")
