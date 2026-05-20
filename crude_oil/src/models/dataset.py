import os
import pandas as pd
import numpy as np
import logging
from pytorch_forecasting import TimeSeriesDataSet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyDatasetBuilder:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_path = os.path.join(self.base_dir, "data", "processed", "master_feature_set.parquet")
        
    def load_and_prep_data(self):
        logger.info("Loading Synthesized Master Matrix...")
        df = pd.read_parquet(self.data_path)
        
        # Reset index to make Date a column
        df = df.reset_index()
        df = df.rename(columns={'index': 'Date'})
        
        # PyTorch Forecasting requires a sequential integer time index and a group ID
        df['time_idx'] = np.arange(len(df))
        df['group_id'] = "energy_market"
        
        # --- Type Casting for the Neural Network ---
        # 1. Target and Continuous variables must be floats
        continuous_cols = [
            'WTI_Spot', 'Brent_WTI_Spread', 'Suez_Canal_Stress_Idx', 
            'Strait_of_Hormuz_Stress_Idx', 'Malacca_Strait_Stress_Idx'
        ]
        
        # Safely add FRED macro variables if they successfully downloaded
        if 'usd_index' in df.columns:
            continuous_cols.extend(['usd_index', 'risk_free_rate'])
            
        for col in continuous_cols:
            df[col] = df[col].astype(float)
            
        # 2. Categoricals must be strings explicitly
        df['Market_Regime'] = df['Market_Regime'].astype(str).astype('category')
        df['day_of_week'] = df['Date'].dt.dayofweek.astype(str).astype('category')
        df['month'] = df['Date'].dt.month.astype(str).astype('category')
        
        return df, continuous_cols

    def create_dataset(self, df, continuous_cols, max_prediction_length=7, max_encoder_length=60):
        logger.info("Constructing PyTorch TimeSeriesDataSet...")
        
        # We hold out the last 30 days for validation
        training_cutoff = df["time_idx"].max() - 30

        training_dataset = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="WTI_Spot",
            group_ids=["group_id"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["group_id"],
            time_varying_known_categoricals=["day_of_week", "month"],
            time_varying_unknown_categoricals=["Market_Regime"], # The HMM Regime
            time_varying_unknown_reals=continuous_cols,          # Prices, Spreads, and Stress Indexes
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, 
            df, 
            predict=True, 
            stop_randomization=True
        )
        
        logger.info(f"Training parameters established. Encoder: {max_encoder_length} days, Horizon: {max_prediction_length} days.")
        return training_dataset, validation_dataset

if __name__ == "__main__":
    builder = EnergyDatasetBuilder()
    df, cols = builder.load_and_prep_data()
    train, val = builder.create_dataset(df, cols)
    print("Dataset construction successful.")