import os
import yaml
import logging
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyDatasetBuilder:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.processed_dir = os.path.join(self.base_dir, "data", "processed")
        
        config_path = os.path.join(self.base_dir, 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.target = self.config['data_sources']['fred_series']['target']
        self.max_encoder_length = self.config['tft_hyperparameters']['context_length']
        self.max_prediction_length = self.config['tft_hyperparameters']['prediction_horizon']

    def prep_dataframe_for_pytorch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['time_idx'] = np.arange(len(df))
        df['group_id'] = "energy_market"
        
        # Categorical features for seasonality
        df['day_of_week'] = df.index.dayofweek.astype(str).astype("category")
        df['month'] = df.index.month.astype(str).astype("category")
        
        # Ensure target is float
        df[self.target] = df[self.target].astype(float)
        return df

    def create_dataset(self):
        logger.info("Loading master feature set...")
        data_path = os.path.join(self.processed_dir, "master_feature_set.parquet")
        
        if not os.path.exists(data_path):
            logger.error("Master dataset not found. Run feature_factory.py first.")
            return None, None
            
        df = pd.read_parquet(data_path)
        df = self.prep_dataframe_for_pytorch(df)
        
        logger.info("Building PyTorch TimeSeriesDataSet for Crude Oil...")
        
        # Dynamically grab the VMD columns
        vmd_cols = [col for col in df.columns if 'mode_' in col]
        
        training_dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=f"target_{self.max_prediction_length}d_ahead",
            group_ids=["group_id"],
            min_encoder_length=self.max_encoder_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=self.max_prediction_length,
            max_prediction_length=self.max_prediction_length,
            
            # Static inputs
            static_categoricals=["group_id"],
            
            # Known future inputs (Seasonality)
            time_varying_known_categoricals=["day_of_week", "month"],
            time_varying_known_reals=["time_idx"],
            
            # Unknown future inputs (The specific Energy macro drivers)
            time_varying_unknown_reals=[
                self.target, 
                f"{self.target}_rolling_7d",
                f"{self.target}_rolling_30d",
                "DTWEXBGS",                  # US Dollar Strength
                "WCSSTUS1",                  # US Strategic Petroleum Reserve (Updated!)
                "sentiment_score",           # OPEC/Middle East Panic Score
                "Suez_Canal_vessels_waiting",
                "Strait_of_Hormuz_vessels_waiting"
            ] + vmd_cols,
            
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        logger.info("Dataset successfully built!")
        return training_dataset, df

if __name__ == "__main__":
    builder = EnergyDatasetBuilder()
    ts_dataset, raw_df = builder.create_dataset()