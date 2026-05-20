import os
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import timedelta

# 1. Page Configuration
st.set_page_config(page_title="Energy Fleet Stress Engine", page_icon="🛢️", layout="wide")

# 2. Paths and Endpoints
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(base_dir, "data", "processed", "master_feature_set.parquet")
API_URL = "http://127.0.0.1:8000/predict"

# --- UI Header ---
st.title("🛢️ Energy Fleet Stress Engine")
st.markdown("""
**Asset Class:** WTI Crude Oil | **Forecast Horizon:** 7 Days  
*Powered by Deep Sequence Modeling (PyTorch TFT), Suez/Hormuz Vessel Tracking, and OPEC NLP Sentiment.*
""")
st.divider()

# --- Backend Communication ---
@st.cache_data(ttl=3600) # Cache for 1 hour to keep UI lightning fast
def load_historical_data():
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        # Grab the last 30 days for context
        return df[['DCOILWTICO']].tail(30)
    return pd.DataFrame()

def fetch_prediction():
    try:
        response = requests.post(API_URL)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        st.error("API Connection Error. Is the FastAPI backend running?")
    return None

# --- Main Dashboard Logic ---
hist_df = load_historical_data()
forecast_data = fetch_prediction()

if forecast_data and not hist_df.empty:
    
    # Extract prediction data
    median_price = forecast_data["predictions"]["median_expected_price"]
    p10_floor = forecast_data["confidence_intervals"]["p10_pessimistic_floor"][0]
    p90_ceiling = forecast_data["confidence_intervals"]["p90_optimistic_ceiling"][0]
    
    # Calculate future dates
    last_date = hist_df.index[-1]
    future_date = last_date + timedelta(days=7)
    
    # 3. Top Level KPIs (Executive Summary)
    col1, col2, col3 = st.columns(3)
    col1.metric("P10 Risk Floor", f"${p10_floor:.2f}", delta="High Supply / Calm", delta_color="inverse")
    col2.metric("Median Forecast (Day 7)", f"${median_price:.2f}", delta="Expected Trajectory", delta_color="off")
    col3.metric("P90 Risk Ceiling", f"${p90_ceiling:.2f}", delta="Supply Shock / Panic", delta_color="normal")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. The Confidence Cone (Plotly Visual Engine)
    fig = go.Figure()

    # Plot historical reality
    fig.add_trace(go.Scatter(
        x=hist_df.index, y=hist_df['DCOILWTICO'],
        mode='lines', name='Historical WTI Price',
        line=dict(color='white', width=2)
    ))

    # Plot the P90 Ceiling (Upper Bound)
    fig.add_trace(go.Scatter(
        x=[last_date, future_date], y=[hist_df['DCOILWTICO'].iloc[-1], p90_ceiling],
        mode='lines', line=dict(width=0), showlegend=False, name='Upper Bound'
    ))

    # Plot the P10 Floor (Lower Bound) - fill to the P90 line to create the cone
    fig.add_trace(go.Scatter(
        x=[last_date, future_date], y=[hist_df['DCOILWTICO'].iloc[-1], p10_floor],
        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 65, 54, 0.2)', 
        name='80% Confidence Interval (P10-P90)'
    ))

    # Plot the Median Prediction
    fig.add_trace(go.Scatter(
        x=[last_date, future_date], y=[hist_df['DCOILWTICO'].iloc[-1], median_price],
        mode='lines', name='Median Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title="WTI Crude Oil 7-Day Forecast Trajectory",
        xaxis_title="Timeline",
        yaxis_title="Price per Barrel ($)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.caption(f"Last API Sync: {last_date.strftime('%Y-%m-%d')} | Engine: PyTorch Forecasting | Backend: FastAPI")
    
else:
    st.warning("Awaiting data connection... Please ensure both the dataset exists and the FastAPI backend is running.")