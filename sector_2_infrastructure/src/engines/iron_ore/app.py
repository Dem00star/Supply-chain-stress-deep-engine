import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import torch
import warnings

# Use the modern lightning namespace if pytorch-forecasting requires it under the hood, 
# but TFT loads from pytorch_forecasting directly.
from pytorch_forecasting import TemporalFusionTransformer

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Iron Ore Infrastructure Engine",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3, h4 {color: #FAFAFA;}
    .metric-card {
        background-color: #1E2127; padding: 20px; border-radius: 10px;
        border-left: 5px solid #FFCA28; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. Load the Model ---
@st.cache_resource
def load_production_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    model_path = os.path.join(base_dir, "models", "best_iron_ore_tft_model.ckpt")
    if not os.path.exists(model_path): return None
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
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_path = os.path.join(base_dir, "data", "processed", "iron_ore", "iron_ore_feature_set.parquet")
    if not os.path.exists(data_path): return None
    return pd.read_parquet(data_path)

# --- 3. Prep Data ---
def prep_dataframe(df, trained_model):
    import numpy as np
    df = df.copy()
    if df.index.name == 'Date' or 'Date' not in df.columns:
        df = df.reset_index()
    if 'index' in df.columns:
        df = df.rename(columns={'index': 'Date'})
        
    df['time_idx'] = np.arange(len(df))
    
    try:
        encoder = trained_model.dataset_parameters['categorical_encoders']['group_id']
        valid_categories = list(encoder.classes_.keys())
        df['group_id'] = valid_categories[0]
    except Exception:
        df['group_id'] = "iron_ore_market" 
    
    df['day_of_week'] = df['Date'].dt.dayofweek.astype(str).astype("category")
    df['month'] = df['Date'].dt.month.astype(str).astype("category")
    df['BHP_Supply_Proxy'] = df['BHP_Supply_Proxy'].astype(float)
    return df

# --- UI Layout ---
st.title("🏗️ Infrastructure Fleet: Iron Ore Engine")
st.markdown("**Asset Class:** Global Iron Ore (BHP Proxy) | **Forecast Horizon:** 7 Days")
st.markdown("*Probabilistic forecasting via Australian Currency Dynamics, Global Dry Bulk Freight Rates, and Mining Equities.*")
st.markdown("---")

# --- Execute Inference Flow ---
model = load_production_model()
raw_data = load_data()

if model is None or raw_data is None:
    st.error("System initializing or missing model/data files.")
else:
    with st.spinner('Calculating Infrastructure Trajectories...'):
        try:
            prepped_df = prep_dataframe(raw_data, model)
            
            # --- Extract Current State Variables ---
            current_aud_usd = prepped_df['AUD_USD_Exchange'].iloc[-1]
            current_dry_bulk = prepped_df['Dry_Bulk_Freight_Index'].iloc[-1]
            current_dollar = prepped_df.get('US_Dollar_Index', pd.Series([0])).iloc[-1]
            
            # --- System State Dashboard ---
            st.subheader("🌍 Live Sector Macro Variables")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-card"><h3>AUD/USD (Demand Proxy)</h3><h2>${current_aud_usd:.4f}</h2></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card" style="border-left-color: #29B6F6;"><h3>Dry Bulk Freight (Shipping)</h3><h2>{current_dry_bulk:.2f}</h2></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card" style="border-left-color: #FF5252;"><h3>US Dollar Index (Macro)</h3><h2>{current_dollar:.2f}</h2></div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- Inference Execution ---
            last_time_idx = prepped_df['time_idx'].max()
            future_df = prepped_df.iloc[-1:].copy()
            future_df['time_idx'] = last_time_idx + 1
            prediction_df = pd.concat([prepped_df, future_df], ignore_index=True)
            
            raw_predictions = model.predict(prediction_df, mode="quantiles", return_x=False)
            preds_list = raw_predictions[0][0].tolist() 
            
            # Adjust index based on QuantileLoss([0.1, 0.5, 0.9])
            p10, median, p90 = round(preds_list[0], 2), round(preds_list[1], 2), round(preds_list[2], 2)
            forecast_horizon = 7 
            
            # --- Price Forecast ---
            st.subheader(f"📊 {forecast_horizon}-Day Probabilistic Forecast (BHP Proxy)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="P10 Risk Floor", value=f"${p10}")
            with col2:
                st.metric(label="Median Expected", value=f"${median}")
            with col3:
                st.metric(label="P90 Risk Ceiling", value=f"${p90}")

            # --- Charting ---
            fig = go.Figure()
            historical_df = prepped_df.tail(45)
            
            fig.add_trace(go.Scatter(
                x=historical_df['Date'], y=historical_df['BHP_Supply_Proxy'],
                mode='lines', name='Historical Iron Ore Proxy', line=dict(color='white', width=2)
            ))

            last_date = historical_df['Date'].iloc[-1]
            future_date = last_date + timedelta(days=forecast_horizon)
            last_price = historical_df['BHP_Supply_Proxy'].iloc[-1]

            fig.add_trace(go.Scatter(
                x=[last_date, future_date, future_date, last_date],
                y=[last_price, p90, p10, last_price],
                fill='toself', fillcolor='rgba(255, 202, 40, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                name='80% Confidence Interval'
            ))

            fig.add_trace(go.Scatter(
                x=[last_date, future_date], y=[last_price, median],
                mode='lines+markers', line=dict(color='#FFCA28', width=4, dash='dot'),
                marker=dict(size=10, symbol='diamond'), name='Median AI Forecast'
            ))

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FAFAFA'), xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#333333', title="Proxy Asset Price ($)"),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Inference execution failed. Error details: {str(e)}")