import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Supply Chain Stress Engine",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3 {color: #FAFAFA;}
    .metric-card {
        background-color: #1E2127;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00E676;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- Backend API Connection ---
# When deployed, we will swap this out for your Render.com URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

@st.cache_data(ttl=3600) # Cache the API response for 1 hour to prevent spamming your backend
def fetch_prediction():
    try:
        response = requests.post(API_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Backend API is currently unreachable. Please ensure it is running. Error: {e}")
        return None

# --- UI Layout ---
st.title("🚢 Global Supply Chain Stress Engine")
st.markdown("Real-time Baltic Dry Index (Copper Proxy) forecasting powered by Deep Sequence Modeling (PyTorch TFT), AIS Vessel Tracking, and NLP Sentiment.")
st.markdown("---")

# Fetch Data
data = fetch_prediction()

if data:
    preds = data["predictions"]
    intervals = data["confidence_intervals"]
    
    # --- Top Metrics Row ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Forecast Horizon</h3>
            <h2>{data['forecast_horizon_days']} Days Ahead</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #29B6F6;">
            <h3>Median Prediction (Expected)</h3>
            <h2>${preds['median_expected_price']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #FF5252;">
            <h3>Risk Bounds (P10 - P90)</h3>
            <h2>${intervals['p10_pessimistic_floor'][0]} - ${intervals['p90_optimistic_ceiling'][0]}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- Interactive Plotly Chart ---
    st.subheader("📊 Forecast Trajectory & Confidence Cone")
    
    # Create a clean, interactive chart showing the prediction bounds
    fig = go.Figure()

    # Define fake "recent past" dates for visual continuity
    today = datetime.now()
    future_date = today + timedelta(days=data['forecast_horizon_days'])

    # Add the Confidence Cone (Shaded Area)
    fig.add_trace(go.Scatter(
        x=[today, future_date, future_date, today],
        y=[preds['median_expected_price'], intervals['p90_optimistic_ceiling'][0], intervals['p10_pessimistic_floor'][0], preds['median_expected_price']],
        fill='toself',
        fillcolor='rgba(41, 182, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='80% Confidence Interval',
        showlegend=True
    ))

    # Add the Median Prediction Line
    fig.add_trace(go.Scatter(
        x=[today, future_date],
        y=[preds['median_expected_price'], preds['median_expected_price']],
        mode='lines+markers',
        line=dict(color='#00E676', width=4, dash='dot'),
        marker=dict(size=10, symbol='diamond'),
        name='Deep Learning Median Forecast'
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        xaxis=dict(showgrid=False, title="Timeline"),
        yaxis=dict(showgrid=True, gridcolor='#333333', title="Proxy Asset Price ($)"),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Engine: PyTorch Forecasting | Backend: FastAPI")