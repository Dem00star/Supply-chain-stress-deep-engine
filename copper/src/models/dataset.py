import os
import logging
import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BDIDatasetBuilder:
    """
    Transforms the Pandas master feature set into a PyTorch Forecasting TimeSeriesDataSet.
    Handles scaling, tensor conversion, and defining the encoder/decoder lengths.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.processed_dir = os.path.join(self.base_dir, "data", "processed")
        
        # Hyperparameters for the Transformer's memory
        self.max_encoder_length = 30  # Look back 30 days into the past
        self.max_prediction_length = 7  # Predict 7 days into the future

    def prep_dataframe_for_pytorch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds necessary tracking columns required by the TFT architecture."""
        df = df.copy()
        
        # PyTorch Forecasting requires an integer time index (not dates)
        df['time_idx'] = np.arange(len(df))
        
        # It also expects to handle multiple series (like 50 different stocks).
        # Since we are forecasting one main global proxy, we give them all the same group ID.
        df['group_id'] = "global_market"
        
        # Extract basic calendar features (known future inputs)
        df['day_of_week'] = df.index.dayofweek.astype(str).astype('category')
        df['month'] = df.index.month.astype(str).astype('category')
        
        return df

    def create_dataset(self):
        """Builds the PyTorch TimeSeriesDataSet object."""
        input_file = os.path.join(self.processed_dir, "master_feature_set.parquet")
        
        if not os.path.exists(input_file):
            logger.error("Master feature set missing. Run feature_factory.py first.")
            return None, None
            
        logger.info("Loading master feature set...")
        df = pd.read_parquet(input_file)
        df = self.prep_dataframe_for_pytorch(df)
        
        # Identify our VMD mode columns dynamically
        vmd_cols = [col for col in df.columns if "mode" in col]
        
        # Define the PyTorch Dataset
        logger.info("Building PyTorch TimeSeriesDataSet...")
        training_dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="target_7d_ahead",
            group_ids=["group_id"],
            
            # Sequence lengths
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            
            # Static inputs (don't change over time)
            static_categoricals=["group_id"],
            
            # Known future inputs (we know tomorrow's day of the week)
            time_varying_known_categoricals=["day_of_week", "month"],
            time_varying_known_reals=["time_idx"],
            
            # Unknown future inputs (we don't know tomorrow's news sentiment or port congestion)
            time_varying_unknown_reals=[
                "copper_close", 
                "copper_rolling_7d",
                "sentiment_score",
                "Shanghai_daily_port_calls"
            ] + vmd_cols,
            
            # We scale the target to help the neural network converge faster
            target_normalizer=None, # We'll let it use the default TorchNormalizer
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        logger.info("Dataset successfully built!")
        return training_dataset, df

if __name__ == "__main__":
    builder = BDIDatasetBuilder()
    ts_dataset, raw_df = builder.create_dataset()
    
    if ts_dataset:
        print("\nPyTorch Dataset Configuration:")
        print(f"Number of samples (sequences) available for training: {len(ts_dataset)}")
        
        # Get a sample batch to prove it's tensor-ready
        dataloader = ts_dataset.to_dataloader(train=True, batch_size=4, num_workers=0)
        x, y = next(iter(dataloader))
        print(f"\nSample Input Tensor Shape (Encoder): {x['encoder_cont'].shape}")
        print("Everything is ready for the neural network!")