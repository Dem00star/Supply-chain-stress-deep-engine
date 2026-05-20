import os
import logging
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketCollector:
    """
    Ingests daily quantitative financial data.
    Uses the Federal Reserve Economic Data (FRED) API for institutional macro indicators,
    and yfinance specifically for the daily Copper target proxy.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_dir = os.path.join(self.base_dir, "data", "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # Pull API key securely from environment
        self.fred_api_key = os.getenv("FRED_API_KEY")
        self.start_date = "2018-01-01"

    def _fetch_target_proxy(self) -> pd.DataFrame:
        """Fetches the daily Copper target variable via yfinance."""
        logger.info("Fetching Daily Copper Futures (Target) from yfinance...")
        copper = yf.download("HG=F", start=self.start_date, progress=False)
        
        # Handle yfinance pandas formatting
        if isinstance(copper.columns, pd.MultiIndex):
            copper_close = copper['Close']['HG=F']
        else:
            copper_close = copper['Close']
            
        df = pd.DataFrame({'copper_close': copper_close})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    def _fetch_macro_indicators(self) -> pd.DataFrame:
        """Fetches macro leading indicators. Uses FRED if key exists, else yfinance fallback."""
        if self.fred_api_key:
            logger.info("FRED API Key detected. Fetching institutional macro data...")
            fred = Fred(api_key=self.fred_api_key)
            
            # SP500 = S&P 500 Daily | DCOILWTICO = Crude Oil WTI Daily
            sp500 = fred.get_series('SP500', observation_start=self.start_date)
            oil = fred.get_series('DCOILWTICO', observation_start=self.start_date)
            
            df = pd.DataFrame({
                'sp500_close': sp500,
                'crude_oil_close': oil
            })
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        else:
            logger.warning("No FRED API Key found. Running 'Portfolio Mode' via yfinance fallback...")
            macro_data = yf.download(["^GSPC", "CL=F"], start=self.start_date, progress=False)
            
            if isinstance(macro_data.columns, pd.MultiIndex):
                sp500_close = macro_data['Close']['^GSPC']
                oil_close = macro_data['Close']['CL=F']
            else:
                sp500_close = macro_data['Close']['^GSPC']
                oil_close = macro_data['Close']['CL=F']
                
            df = pd.DataFrame({
                'sp500_close': sp500_close,
                'crude_oil_close': oil_close
            })
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df

    def fetch_all_assets(self):
        """Merges target and macro data, cleans it, and saves to Parquet."""
        logger.info(f"Starting market data ingestion from {self.start_date}...")
        
        try:
            # 1. Fetch data components
            target_df = self._fetch_target_proxy()
            macro_df = self._fetch_macro_indicators()
            
            # 2. Merge on the date index
            combined_data = target_df.join(macro_df, how="outer")
            
            # 3. Clean and Forward-Fill missing weekend/holiday data
            combined_data = combined_data.ffill().dropna()
            
            # 4. Save to disk
            output_file = os.path.join(self.raw_dir, "market_data_raw.parquet")
            combined_data.to_parquet(output_file)
            
            logger.info(f"Successfully saved market data to {output_file}")
            return combined_data
            
        except Exception as e:
            logger.error(f"Market Ingestion failed: {str(e)}")
            raise

if __name__ == "__main__":
    collector = MarketCollector()
    df = collector.fetch_all_assets()
    print("\nMarket Data Snapshot (FRED + yfinance):")
    print(df.tail())