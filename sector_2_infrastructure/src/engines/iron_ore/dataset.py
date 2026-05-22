import os
import pandas as pd
import numpy as np
import logging
from pytorch_forecasting import TimeSeriesDataSet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IronOreDatasetBuilder:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.data_path = os.path.join(self.base_dir, "data", "processed", "iron_ore", "iron_ore_feature_set.parquet")
        
    def load_and_prep_data(self):
        logger.info("Loading Synthesized Iron Ore Matrix...")
        df = pd.read_parquet(self.data_path)
        
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'Date'})
            
        df['time_idx'] = np.arange(len(df))
        df['group_id'] = "iron_ore_market"
        
        # --- Type Casting ---
        continuous_cols = [
            'BHP_Supply_Proxy', 'RIO_Supply_Proxy', 'AUD_USD_Exchange',
            'Dry_Bulk_Freight_Index', 'Global_Mining_Health', 'Copper_Miners_Index'
        ]
        
        # Add FRED macro variables if they were successfully pulled
        if 'US_Dollar_Index' in df.columns:
            continuous_cols.extend(['US_Dollar_Index', 'US_10Yr_Treasury'])
            
        for col in continuous_cols:
            df[col] = df[col].astype(float)
            
        # Time-based categoricals for seasonality
        df['day_of_week'] = df['Date'].dt.dayofweek.astype(str).astype('category')
        df['month'] = df['Date'].dt.month.astype(str).astype('category')
        
        return df, continuous_cols

    def create_dataset(self, df, continuous_cols, max_prediction_length=7, max_encoder_length=60):
        logger.info("Constructing PyTorch TimeSeriesDataSet for Iron Ore...")
        
        training_cutoff = df["time_idx"].max() - 30

        training_dataset = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="BHP_Supply_Proxy",
            group_ids=["group_id"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["group_id"],
            time_varying_known_categoricals=["day_of_week", "month"],
            time_varying_unknown_reals=continuous_cols,
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
        
        logger.info(f"Training parameters set. Encoder: {max_encoder_length} days, Horizon: {max_prediction_length} days.")
        return training_dataset, validation_dataset

if __name__ == "__main__":
    builder = IronOreDatasetBuilder()
    df, cols = builder.load_and_prep_data()
    train, val = builder.create_dataset(df, cols)
    print("Iron Ore Dataset construction successful.")