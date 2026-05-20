import os
import yaml
import logging
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyCurveCollector:
    """
    Ingests WTI, Brent, and Macro rates to calculate the structural
    forces of the oil market (Spreads and Cost of Carry).
    """
    def __init__(self):
        # This resolves to the 'crude_oil/' directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # --- THE FIX: explicitly load the .env from the monorepo root ---
        monorepo_root = os.path.dirname(self.base_dir)
        env_path = os.path.join(monorepo_root, '.env')
        load_dotenv(env_path)
        # ----------------------------------------------------------------
        
        config_path = os.path.join(self.base_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'raw')
        self.financial_cfg = self.config['financial_layer']
        
        # This will now successfully pull the key from the root folder!
        self.fred_api_key = os.getenv("FRED_API_KEY")
        if self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
        else:
            logger.warning("FRED_API_KEY missing. US Dollar and Treasury yields will be bypassed.")

    def fetch_oil_curves(self, start_date="2018-01-01"):
        logger.info("Pulling Global Oil Benchmarks from yfinance...")
        tickers = [
            self.financial_cfg['yfinance_tickers']['wti_front_month'],
            self.financial_cfg['yfinance_tickers']['brent_front_month']
        ]
        
        # Pull futures data
        df_oil = yf.download(tickers, start=start_date, progress=False)['Close']
        
        # Flatten MultiIndex if necessary and rename
        if isinstance(df_oil.columns, pd.MultiIndex):
            df_oil.columns = df_oil.columns.get_level_values(0)
            
        df_oil.rename(columns={
            self.financial_cfg['yfinance_tickers']['wti_front_month']: 'WTI_Spot',
            self.financial_cfg['yfinance_tickers']['brent_front_month']: 'Brent_Spot'
        }, inplace=True)
        
        df_oil.index = pd.to_datetime(df_oil.index).tz_localize(None)
        return df_oil

    def fetch_macro_rates(self, start_date="2018-01-01"):
        logger.info("Pulling Risk-Free Rates and USD Strength from FRED...")
        if not self.fred_api_key:
            return pd.DataFrame()

        dfs = []
        for name, series_id in self.financial_cfg['fred_series'].items():
            try:
                series_data = self.fred.get_series(series_id, observation_start=start_date)
                df = pd.DataFrame({'Date': series_data.index, name: series_data.values})
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to fetch {name}: {e}")

        if dfs:
            macro_df = pd.concat(dfs, axis=1)
            macro_df.ffill(inplace=True)
            return macro_df
        return pd.DataFrame()

    def build_financial_base(self):
        logger.info("Constructing Quantitative Financial Base Layer...")
        
        oil_df = self.fetch_oil_curves()
        macro_df = self.fetch_macro_rates()
        
        # Merge Oil and Macro
        master_financial = oil_df.join(macro_df, how='left')
        master_financial.ffill(inplace=True)
        master_financial.dropna(inplace=True)
        
        # Quantitative Feature: The Brent-WTI Spread
        # A widening spread indicates US oversupply or Eastern hemisphere geopolitical risk
        master_financial['Brent_WTI_Spread'] = master_financial['Brent_Spot'] - master_financial['WTI_Spot']
        
        output_path = os.path.join(self.raw_data_dir, 'financial_curve_raw.parquet')
        master_financial.to_parquet(output_path)
        
        logger.info(f"SUCCESS: Financial Base Layer saved. Shape: {master_financial.shape}")
        return master_financial

if __name__ == "__main__":
    collector = EnergyCurveCollector()
    df = collector.build_financial_base()
    print("\nFinancial Base Layer (Spot, Spreads, and Rates):")
    print(df.tail())