import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import torch
import warnings
from pytorch_forecasting import TemporalFusionTransformer

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Copper Infrastructure Engine", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    h1, h2, h3, h4 {color: #FAFAFA;}
    .metric-card {
        background-color: #1E2127; padding: 20px; border-radius: 10px;
        border-left: 5px solid #00E676; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_production_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    model_path = os.path.join(base_dir, "models", "best_copper_tft_model.ckpt")
    if not os.path.exists(model_path): return None
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load PyTorch model: {e}")
        return None

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_path = os.path.join(base_dir, "data", "processed", "copper", "copper_feature_set.parquet")
    if not os.path.exists(data_path): return None
    return pd.read_parquet(data_path)

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
        df['group_id'] = "copper_market" 
    
    df['day_of_week'] = df['Date'].dt.dayofweek.astype(str).astype("category")
    df['month'] = df['Date'].dt.month.astype(str).astype("category")
    df['Copper_Close'] = df['Copper_Close'].astype(float)
    return df

st.title("⚡ Infrastructure Fleet: Copper Engine")
st.markdown("**Asset Class:** Global Copper (COMEX) | **Forecast Horizon:** 7 Days")
st.markdown("*Probabilistic forecasting via Global Mining Equities, Dry Bulk Shipping, and Dollar Liquidity.*")

# --- BIG NAVIGATION BUTTON ---
st.markdown("<br>", unsafe_allow_html=True)
st.link_button(
    "🏗️ Switch to Iron Ore Engine (Sector 2)", 
    "https://supply-chain-stress-deep-engine-1.streamlit.app/", 
    use_container_width=True
)
st.markdown("---")

model = load_production_model()
raw_data = load_data()

if model is None or raw_data is None:
    st.error("System initializing or missing model/data files.")
else:
    with st.spinner('Calculating Infrastructure Trajectories...'):
        try:
            prepped_df = prep_dataframe(raw_data, model)
            
            current_miners = prepped_df['Copper_Miners_Index'].iloc[-1]
            current_health = prepped_df['Global_Mining_Health'].iloc[-1]
            current_dollar = prepped_df.get('US_Dollar_Index', pd.Series([0])).iloc[-1]
            
            st.subheader("🌍 Live Sector Macro Variables")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-card"><h3>Copper Miners ETF</h3><h2>${current_miners:.2f}</h2></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card" style="border-left-color: #29B6F6;"><h3>Global Mining Health</h3><h2>${current_health:.2f}</h2></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card" style="border-left-color: #FF5252;"><h3>US Dollar Index</h3><h2>{current_dollar:.2f}</h2></div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            last_time_idx = prepped_df['time_idx'].max()
            future_df = prepped_df.iloc[-1:].copy()
            future_df['time_idx'] = last_time_idx + 1
            prediction_df = pd.concat([prepped_df, future_df], ignore_index=True)
            
            raw_predictions = model.predict(prediction_df, mode="quantiles", return_x=False)
            preds_list = raw_predictions[0][0].tolist() 
            
            p10, median, p90 = round(preds_list[0], 2), round(preds_list[1], 2), round(preds_list[2], 2)
            forecast_horizon = 7 
            
            st.subheader(f"📊 {forecast_horizon}-Day Probabilistic Forecast (COMEX Copper)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="P10 Risk Floor", value=f"${p10}")
            with col2:
                st.metric(label="Median Expected", value=f"${median}")
            with col3:
                st.metric(label="P90 Risk Ceiling", value=f"${p90}")

            fig = go.Figure()
            historical_df = prepped_df.tail(45)
            
            fig.add_trace(go.Scatter(
                x=historical_df['Date'], y=historical_df['Copper_Close'],
                mode='lines', name='Historical Copper', line=dict(color='white', width=2)
            ))

            last_date = historical_df['Date'].iloc[-1]
            future_date = last_date + timedelta(days=forecast_horizon)
            last_price = historical_df['Copper_Close'].iloc[-1]

            fig.add_trace(go.Scatter(
                x=[last_date, future_date, future_date, last_date],
                y=[last_price, p90, p10, last_price],
                fill='toself', fillcolor='rgba(0, 230, 118, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                name='80% Confidence Interval'
            ))

            fig.add_trace(go.Scatter(
                x=[last_date, future_date], y=[last_price, median],
                mode='lines+markers', line=dict(color='#00E676', width=4, dash='dot'),
                marker=dict(size=10, symbol='diamond'), name='Median AI Forecast'
            ))

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FAFAFA'), xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#333333', title="Price per Pound ($)"),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Inference execution failed. Error details: {str(e)}")