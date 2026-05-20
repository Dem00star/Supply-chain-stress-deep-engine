import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import torch
from pytorch_forecasting import TemporalFusionTransformer
import pandas as pd
from src.api.schemas import ForecastResponse

# Fix path routing to find the models and data directories
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(base_dir, "models", "best_tft_model.ckpt")
data_path = os.path.join(base_dir, "data", "processed", "master_feature_set.parquet")

logger = logging.getLogger("uvicorn.error")

# Global dictionary to hold our loaded model in RAM
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the heavy PyTorch model into memory right when the server starts."""
    logger.info("Starting up API and loading PyTorch TFT Model into memory...")
    try:
        # Load the model weights we just trained
        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        # Put the model in evaluation mode (turns off dropout)
        model.eval()
        ml_models["tft"] = model
        logger.info("Model successfully loaded!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    # Clean up on shutdown
    logger.info("Shutting down API and clearing memory...")
    ml_models.clear()

app = FastAPI(
    title="Supply Chain Stress API",
    description="Live inference engine for the Baltic Dry Index / Copper proxy.",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "online", "model_loaded": "tft" in ml_models}

@app.post("/predict", response_model=ForecastResponse)
async def get_prediction():
    """Runs a forward pass of the neural network on the latest available data."""
    model = ml_models.get("tft")
    if not model:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        # 1. Load the very latest data points
        df = pd.read_parquet(data_path)
        
        # 2. PyTorch Forecasting requires the data to be formatted through the dataset wrapper
        # For this portfolio demo, we'll grab the last valid sequence from our dataframe
        # In a full live production app, we would dynamically build this from today's live API pulls
        from src.models.dataset import BDIDatasetBuilder
        builder = BDIDatasetBuilder()
        df_prepped = builder.prep_dataframe_for_pytorch(df)
        
        # We need a dummy target for the future to satisfy the dataset constraints during inference
        last_time_idx = df_prepped['time_idx'].max()
        future_df = df_prepped.iloc[-1:].copy()
        future_df['time_idx'] = last_time_idx + 1
        prediction_df = pd.concat([df_prepped, future_df], ignore_index=True)

        # 3. Ask the model to predict
        # model.predict returns the 7 quantiles we trained it on
        raw_predictions = model.predict(prediction_df, mode="quantiles", return_x=False)
        
        # Extract the median (50th percentile) as our main prediction
        # and the 10th/90th percentiles for our confidence intervals
        preds = raw_predictions[0][0].tolist() # Grab the first batch, first time step
        
        return ForecastResponse(
            status="success",
            target_variable="copper_proxy",
            forecast_horizon_days=7,
            predictions={
                "median_expected_price": round(preds[3], 4) # The middle of our 7 quantiles
            },
            confidence_intervals={
                "p10_pessimistic_floor": [round(preds[1], 4)],
                "p90_optimistic_ceiling": [round(preds[5], 4)]
            }
        )
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal prediction engine error")