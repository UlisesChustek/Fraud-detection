import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# UI Configuration
st.set_page_config(page_title="Sentinel | Business Impact", page_icon="💰", layout="wide")

st.title("🛡️ Sentinel Fraud Prevention: Business Impact")
st.markdown("### Translating Model Accuracy into Financial Value")

# Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('models/sentinel_model.pkl')
    features = pd.read_csv('data/features_data.csv')
    # Load original data to get the real transaction amounts for ROI calculation
    raw = pd.read_csv('data/data.csv') 
    return model, features, raw

model, features, raw = load_assets()

# Predictions logic
X = features.drop('Fraud_Label', axis=1)
y_actual = features['Fraud_Label']
y_pred = model.predict(X)

# Business Metrics Calculation
results = pd.DataFrame({
    'Actual': y_actual,
    'Predicted': y_pred,
    'Amount': raw['Transaction_Amount']
})

# Financial Metrics
total_fraud_volume = results[results['Actual'] == 1]['Amount'].sum()
detected_fraud = results[(results['Actual'] == 1) & (results['Predicted'] == 1)]['Amount'].sum()
missed_fraud = total_fraud_volume - detected_fraud
detection_rate = (detected_fraud / total_fraud_volume) * 100

# User Interface Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Total Savings (ROI)", f"${detected_fraud:,.2f}", "Fraud Blocked")

with col2:
    st.metric("📉 Detection Rate", f"{detection_rate:.1f}%", "Overall Efficiency")

with col3:
    st.metric("⚠️ Uncaught Loss", f"${missed_fraud:,.2f}", "-Potential Risk", delta_color="inverse")

st.divider()

# ROI Visualization
st.subheader("Financial Performance Analysis")
fig_data = pd.DataFrame({
    'Status': ['Stopped (Savings)', 'Missed (Loss)'],
    'Amount': [detected_fraud, missed_fraud]
})
fig = px.bar(fig_data, x='Status', y='Amount', color='Status', 
             color_discrete_map={'Stopped (Savings)': '#00CC96', 'Missed (Loss)': '#EF553B'},
             text_auto='.2s')
st.plotly_chart(fig, use_container_width=True)

# Technical Insight
with st.expander("🔍 Note on Feature Importance"):
    st.write("The model prioritizes behavioral patterns. High variance in 'Transaction Distance' or 'Amount' often leads to higher weight in decision trees. Current importance levels are being monitored for future pruning.")

st.markdown("---")
st.caption("Developed by Ulises Chustek | Data Processing Specialist")