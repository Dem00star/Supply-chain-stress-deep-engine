import os
import yaml
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyFeatureFactory:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(self.base_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.target_col = self.config['data_sources']['fred_series']['target']
        self.horizon = self.config['tft_hyperparameters']['prediction_horizon']

    def build_master_dataset(self):
        """Merges all modalities and creates the 7-day-ahead target."""
        logger.info("Starting Feature Factory for Energy Engine...")
        
        # 1. Load all data streams
        market_df = pd.read_parquet(os.path.join(self.data_dir, 'raw', 'market_data_raw.parquet'))
        ais_df = pd.read_parquet(os.path.join(self.data_dir, 'raw', 'ais_data_raw.parquet'))
        
        # Safely load sentiment (in case the NLP scraper hasn't built enough history yet)
        sentiment_path = os.path.join(self.data_dir, 'processed', 'daily_sentiment.parquet')
        if os.path.exists(sentiment_path):
            sentiment_df = pd.read_parquet(sentiment_path)
            sentiment_df.set_index('Date', inplace=True)
        else:
            sentiment_df = pd.DataFrame(index=market_df.index, columns=['sentiment_score']).fillna(0)
            
        vmd_df = pd.read_parquet(os.path.join(self.data_dir, 'processed', 'vmd_features.parquet'))
        
        # 2. Merge everything aligning by Date
        master_df = market_df.join([ais_df, sentiment_df, vmd_df], how='inner')
        master_df.ffill(inplace=True)
        master_df.fillna(0, inplace=True) # Fill any leftover missing sentiment with neutral (0)
        
        # 3. Feature Engineering: Rolling Averages for momentum
        master_df[f'{self.target_col}_rolling_7d'] = master_df[self.target_col].rolling(window=7).mean()
        master_df[f'{self.target_col}_rolling_30d'] = master_df[self.target_col].rolling(window=30).mean()
        
        # 4. Target Generation: What is the price going to be exactly 7 days from now?
        master_df[f'target_{self.horizon}d_ahead'] = master_df[self.target_col].shift(-self.horizon)
        
        # Drop the last 7 rows because we don't know the future yet!
        master_df.dropna(inplace=True)
        
        output_path = os.path.join(self.data_dir, 'processed', 'master_feature_set.parquet')
        master_df.to_parquet(output_path)
        logger.info(f"SUCCESS: Master ML-Ready dataset created with {len(master_df.columns)} features.")
        logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    factory = EnergyFeatureFactory()
    factory.build_master_dataset()