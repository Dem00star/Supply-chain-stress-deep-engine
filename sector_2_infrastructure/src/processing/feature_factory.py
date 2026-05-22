import os
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureFactory:
    """
    Merges all raw and processed data streams into a single master timeline.
    Generates time-series features (lags, rolling averages) and the future target variable.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_dir = os.path.join(self.base_dir, "data", "raw")
        self.processed_dir = os.path.join(self.base_dir, "data", "processed")
        
        # Define the forecasting horizon (e.g., predict 7 days into the future)
        self.forecast_horizon = 7

    def _load_dataframes(self):
        """Loads all available data components."""
        dfs = {}
        try:
            # Load Market Data
            market_path = os.path.join(self.raw_dir, "market_data_raw.parquet")
            if os.path.exists(market_path):
                dfs['market'] = pd.read_parquet(market_path)
            
            # Load AIS Data
            ais_path = os.path.join(self.raw_dir, "ais_data_raw.parquet")
            if os.path.exists(ais_path):
                dfs['ais'] = pd.read_parquet(ais_path)
                
            # Load Sentiment Data
            sentiment_path = os.path.join(self.processed_dir, "daily_sentiment.parquet")
            if os.path.exists(sentiment_path):
                # Sentiment has 'Date' as a column, set it to index to match others
                sentiment_df = pd.read_parquet(sentiment_path)
                sentiment_df.set_index('Date', inplace=True)
                dfs['sentiment'] = sentiment_df
                
            # Load VMD Features
            vmd_path = os.path.join(self.processed_dir, "vmd_features.parquet")
            if os.path.exists(vmd_path):
                dfs['vmd'] = pd.read_parquet(vmd_path)
                
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            
        return dfs

    def build_master_dataset(self):
        """Merges data, engineers features, and saves the final ML-ready set."""
        logger.info("Starting feature factory...")
        
        dfs = self._load_dataframes()
        if not dfs:
            logger.error("No data found to merge. Run ingestion scripts first.")
            return None

        # 1. Merge everything on the Date index
        logger.info("Merging datasets...")
        # Start with the market data as the backbone, join the rest
        master_df = dfs.get('market', pd.DataFrame())
        
        for name, df in dfs.items():
            if name != 'market' and not df.empty:
                # Use outer join to ensure we don't drop days where only one signal updated
                master_df = master_df.join(df, how='outer')

        # Forward fill to handle weekends and mismatched reporting days
        master_df = master_df.ffill()

        # 2. Engineer Time-Series Features (Rolling Averages & Lags)
        logger.info("Generating time-series features...")
        
        # Let's create a rolling 7-day average of our target proxy (Copper)
        if 'copper_close' in master_df.columns:
            master_df['copper_rolling_7d'] = master_df['copper_close'].rolling(window=7).mean()
            
        # Create a lag feature for AIS congestion (e.g., Shanghai congestion 3 days ago)
        if 'Shanghai_daily_port_calls' in master_df.columns:
            master_df['shanghai_congestion_lag_3d'] = master_df['Shanghai_daily_port_calls'].shift(3)

        # 3. Create the Target Variable (What we want to predict)
        logger.info(f"Generating target variable ({self.forecast_horizon} days ahead)...")
        # We shift the target column BACKWARDS. 
        # This aligns today's features with the price 7 days from now.
        if 'copper_close' in master_df.columns:
            master_df[f'target_{self.forecast_horizon}d_ahead'] = master_df['copper_close'].shift(-self.forecast_horizon)

        # 4. Clean up
        # Fill missing historical sentiment with 0.0 (Neutral)
        if 'sentiment_score' in master_df.columns:
            master_df['sentiment_score'] = master_df['sentiment_score'].fillna(0.0)
            
        # Only drop rows where our CRITICAL features are missing 
        # (e.g., the very beginning without rolling history, or the very end without a future target)
        critical_cols = []
        if 'copper_rolling_7d' in master_df.columns:
            critical_cols.append('copper_rolling_7d')
        if f'target_{self.forecast_horizon}d_ahead' in master_df.columns:
            critical_cols.append(f'target_{self.forecast_horizon}d_ahead')
            
        if critical_cols:
            master_df.dropna(subset=critical_cols, inplace=True)

        # Save to processed directory
        output_file = os.path.join(self.processed_dir, "master_feature_set.parquet")
        master_df.to_parquet(output_file)
        
        logger.info(f"Successfully created master feature set with {len(master_df.columns)} features and {len(master_df)} rows.")
        logger.info(f"Saved to: {output_file}")
        
        return master_df

if __name__ == "__main__":
    factory = FeatureFactory()
    df = factory.build_master_dataset()
    if df is not None:
        print("\nMaster ML-Ready Dataset Snapshot (Last 5 days):")
        # Print a subset of columns to fit the terminal
        cols_to_show = ['copper_close', 'Shanghai_daily_port_calls', 'sentiment_score', 'copper_mode_1', 'target_7d_ahead']
        available_cols = [c for c in cols_to_show if c in df.columns]
        print(df[available_cols].tail())