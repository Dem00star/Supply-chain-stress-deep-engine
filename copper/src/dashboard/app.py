import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import torch
from pytorch_forecasting import TemporalFusionTransformer
import warnings
warnings.filterwarnings('ignore')

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


# --- 1. Load the Model into Streamlit ---
# Use cache_resource so the heavy model only loads ONCE per server boot, not on every user click
@st.cache_resource
def load_production_model():
    # Resolve the path to where the model is stored in your repo
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "best_tft_model.ckpt")
    
    if not os.path.exists(model_path):
        return None
        
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load PyTorch model: {e}")
        return None

# --- 2. Load the Pre-Calculated Data ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "processed", "master_feature_set.parquet")
    
    if not os.path.exists(data_path):
        return None
        
    df = pd.read_parquet(data_path)
    return df

# --- 3. Prep Data for PyTorch Forecasting ---
# (We bring over the logic from your CopperDatasetBuilder here so we don't need the external class)
def prep_dataframe(df):
    import numpy as np
    df = df.copy()
    # Reset index so 'Date' becomes a column if it was the index
    if df.index.name == 'Date' or 'Date' not in df.columns:
        df = df.reset_index()
    
    df['time_idx'] = np.arange(len(df))
    df['group_id'] = "copper_market"
    
    # Categorical features for seasonality
    df['day_of_week'] = df['Date'].dt.dayofweek.astype(str).astype("category")
    df['month'] = df['Date'].dt.month.astype(str).astype("category")
    
    # Ensure target is float
    df['copper_close'] = df['copper_close'].astype(float)
    return df

# --- UI Layout ---
st.title("🚢 Global Supply Chain Stress Engine")
st.markdown("Real-time Baltic Dry Index (Copper Proxy) forecasting powered by Deep Sequence Modeling (PyTorch TFT), AIS Vessel Tracking, and NLP Sentiment.")
st.markdown("---")

# --- Execute Inference Flow ---
model = load_production_model()
raw_data = load_data()

if model is None:
    st.error("Model checkpoint not found. Ensure `models/best_tft_model.ckpt` is committed to GitHub.")
elif raw_data is None:
    st.error("Processed data not found. Ensure `data/processed/master_feature_set.parquet` is committed to GitHub.")
else:
    with st.spinner('Running AI Inference Engine...'):
        try:
            # 1. Prep data format
            prepped_df = prep_dataframe(raw_data)
            
            # 2. Setup the "dummy future" row that the TFT architecture requires for prediction
            last_time_idx = prepped_df['time_idx'].max()
            future_df = prepped_df.iloc[-1:].copy()
            future_df['time_idx'] = last_time_idx + 1
            # Concat the past context and the empty future slot
            prediction_df = pd.concat([prepped_df, future_df], ignore_index=True)
            
            # 3. Predict!
            raw_predictions = model.predict(prediction_df, mode="quantiles", return_x=False)
            
            # 4. Extract Quantiles
            # Depending on how many quantiles you trained with, the index changes. 
            # Assuming 7 quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
            preds_list = raw_predictions[0][0].tolist() 
            
            p10 = round(preds_list[1], 2)
            median = round(preds_list[3], 2)
            p90 = round(preds_list[5], 2)
            
            forecast_horizon = 7 # Adjust if you trained on a different horizon
            
            # --- Top Metrics Row ---
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Forecast Horizon</h3>
                    <h2>{forecast_horizon} Days Ahead</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #29B6F6;">
                    <h3>Median Prediction (Expected)</h3>
                    <h2>${median}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #FF5252;">
                    <h3>Risk Bounds (P10 - P90)</h3>
                    <h2>${p10} - ${p90}</h2>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)

            # --- Interactive Plotly Chart ---
            st.subheader("📊 Forecast Trajectory & Confidence Cone")
            
            fig = go.Figure()

            # Grab the last 30 days of actual data to plot history
            historical_df = prepped_df.tail(30)
            
            # Plot historical reality
            fig.add_trace(go.Scatter(
                x=historical_df['Date'], 
                y=historical_df['copper_close'],
                mode='lines', name='Historical Copper Price',
                line=dict(color='white', width=2)
            ))

            last_date = historical_df['Date'].iloc[-1]
            future_date = last_date + timedelta(days=forecast_horizon)
            last_price = historical_df['copper_close'].iloc[-1]

            # Add the Confidence Cone (Shaded Area)
            fig.add_trace(go.Scatter(
                x=[last_date, future_date, future_date, last_date],
                y=[last_price, p90, p10, last_price],
                fill='toself',
                fillcolor='rgba(41, 182, 246, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='80% Confidence Interval',
                showlegend=True
            ))

            # Add the Median Prediction Line
            fig.add_trace(go.Scatter(
                x=[last_date, future_date],
                y=[last_price, median],
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
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Engine: PyTorch Forecasting | Hosted purely on Streamlit Cloud")

        except Exception as e:
            st.error(f"Inference execution failed. Ensure data schema matches training parameters. Error details: {str(e)}")