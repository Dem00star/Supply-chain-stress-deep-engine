import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import torch
from pytorch_forecasting import TemporalFusionTransformer
import pandas as pd
from src.api.schemas import EnergyForecastResponse

# Route to the crude_oil specific directories
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(base_dir, "models", "best_energy_tft_model.ckpt")
data_path = os.path.join(base_dir, "data", "processed", "master_feature_set.parquet")

logger = logging.getLogger("uvicorn.error")

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the heavy PyTorch model into memory right when the server starts."""
    logger.info("Starting up Energy API and loading PyTorch TFT Model into memory...")
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        model.eval()
        ml_models["tft"] = model
        logger.info("Energy Model successfully loaded!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    logger.info("Shutting down API and clearing memory...")
    ml_models.clear()

app = FastAPI(
    title="Energy Fleet Stress API",
    description="Live inference engine for WTI Crude Oil and OPEC geopolitics.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "online", "model_loaded": "tft" in ml_models}

@app.post("/predict", response_model=EnergyForecastResponse)
async def get_prediction():
    """Runs a forward pass of the neural network on the latest Energy data."""
    model = ml_models.get("tft")
    if not model:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        # Load the latest oil and geopolitical data
        df = pd.read_parquet(data_path)
        
        # Use the specific Energy Dataset Builder
        from src.models.dataset import EnergyDatasetBuilder
        builder = EnergyDatasetBuilder()
        df_prepped = builder.prep_dataframe_for_pytorch(df)
        
        # Dummy target for future inference constraints
        last_time_idx = df_prepped['time_idx'].max()
        future_df = df_prepped.iloc[-1:].copy()
        future_df['time_idx'] = last_time_idx + 1
        prediction_df = pd.concat([df_prepped, future_df], ignore_index=True)

        raw_predictions = model.predict(prediction_df, mode="quantiles", return_x=False)
        preds = raw_predictions[0][0].tolist() 
        
        return EnergyForecastResponse(
            status="success",
            target_variable="WTI_Crude_Oil",
            forecast_horizon_days=7,
            predictions={
                "median_expected_price": round(preds[3], 2) 
            },
            confidence_intervals={
                "p10_pessimistic_floor": [round(preds[1], 2)],
                "p90_optimistic_ceiling": [round(preds[5], 2)]
            }
        )
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal prediction engine error")