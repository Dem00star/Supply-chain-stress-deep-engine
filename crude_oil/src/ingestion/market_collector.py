import os
import yaml
import logging
import pandas as pd
from fredapi import Fred
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyMarketCollector:
    def __init__(self):
        # 1. Load the Energy-Specific Configuration
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(self.base_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.fred_series = self.config['data_sources']['fred_series']
        
        # 2. Initialize the FRED API Client
        # Make sure your .env file in the crude_oil/ root has your FRED_API_KEY
        self.fred_api_key = os.getenv("FRED_API_KEY")
        if not self.fred_api_key:
            logger.warning("FRED_API_KEY not found in environment. Please set it before pulling data.")
        else:
            self.fred = Fred(api_key=self.fred_api_key)

        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'raw')
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def fetch_macro_energy_data(self, lookback_years=5):
        """Pulls WTI prices, USD strength, and SPR inventory levels."""
        logger.info("Initializing Energy Macro Data Pull from Federal Reserve...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * lookback_years)
        
        raw_dfs = []
        
        for name, series_id in self.fred_series.items():
            try:
                logger.info(f"Fetching {name} (Series ID: {series_id})...")
                series_data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                
                df = pd.DataFrame({'Date': series_data.index, series_id: series_data.values})
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                # Forward fill missing weekend data (standard for macro financial trading)
                df.ffill(inplace=True)
                raw_dfs.append(df)
                
            except Exception as e:
                logger.error(f"Failed to fetch {name} ({series_id}): {e}")

        # Merge all fetched indicators into a single continuous DataFrame
        if raw_dfs:
            master_df = pd.concat(raw_dfs, axis=1)
            master_df.ffill(inplace=True)  # Final sweep for any missing alignment dates
            master_df.dropna(inplace=True)
            
            output_path = os.path.join(self.raw_data_dir, 'market_data_raw.parquet')
            master_df.to_parquet(output_path)
            
            logger.info(f"SUCCESS: Master Energy Macro dataset saved to {output_path}")
            logger.info(f"Dataset Shape: {master_df.shape}")
            return master_df
        else:
            logger.error("No data was fetched. Check API key and network connection.")
            return None

if __name__ == "__main__":
    # Test the pipeline locally
    collector = EnergyMarketCollector()
    collector.fetch_macro_energy_data()