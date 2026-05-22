import os
import logging
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InfrastructureMacroCollector:
    """
    Pulls shared macroeconomic drivers for Sector 2 (Copper, Iron Ore, Coal).
    Focuses on US Dollar strength, Global Demand, and Dry Bulk Shipping Proxies.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Connect to the Master .env file
        monorepo_root = os.path.dirname(self.base_dir)
        env_path = os.path.join(monorepo_root, '.env')
        load_dotenv(env_path)
        
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'raw', 'shared_macro')
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        self.fred_api_key = os.getenv("FRED_API_KEY")
        if self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
        else:
            logger.warning("FRED API key missing. Macro rates will be bypassed.")

    def fetch_global_equities_and_shipping(self, start_date="2018-01-01"):
        logger.info("Pulling Global Infrastructure & Dry Bulk Proxies...")
        
        # BDRY: Breakwave Dry Bulk Shipping ETF (Proxy for Capesize/Panamax freight rates)
        # PICK: iShares MSCI Global Metals & Mining Producers ETF
        # COPX: Global Copper Miners ETF
        tickers = ["BDRY", "PICK", "COPX"]
        
        df = yf.download(tickers, start=start_date, progress=False)['Close']
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.rename(columns={
            "BDRY": "Dry_Bulk_Freight_Index",
            "PICK": "Global_Mining_Health",
            "COPX": "Copper_Miners_Index"
        }, inplace=True)
        
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.ffill(inplace=True)
        return df

    def fetch_fed_macro_data(self, start_date="2018-01-01"):
        logger.info("Pulling US Dollar and Global Liquidity from FRED...")
        if not self.fred_api_key:
            return pd.DataFrame()
            
        series_map = {
            "DTWEXBGS": "US_Dollar_Index", # A strong dollar crushes metal prices
            "DGS10": "US_10Yr_Treasury"    # Risk-free rate affecting capital intensive mining
        }
        
        dfs = []
        for fred_code, column_name in series_map.items():
            try:
                data = self.fred.get_series(fred_code, observation_start=start_date)
                temp_df = pd.DataFrame({'Date': data.index, column_name: data.values})
                temp_df['Date'] = pd.to_datetime(temp_df['Date'])
                temp_df.set_index('Date', inplace=True)
                dfs.append(temp_df)
            except Exception as e:
                logger.error(f"Failed to pull {column_name}: {e}")
                
        if dfs:
            macro_df = pd.concat(dfs, axis=1)
            macro_df = macro_df.resample('B').ffill()
            return macro_df
        return pd.DataFrame()

    def build_sector_foundation(self):
        logger.info("Constructing Sector 2 Shared Macro Foundation...")
        
        market_df = self.fetch_global_equities_and_shipping()
        fed_df = self.fetch_fed_macro_data()
        
        shared_master = market_df.join(fed_df, how='left')
        shared_master.ffill(inplace=True)
        shared_master.dropna(inplace=True)
        
        output_path = os.path.join(self.raw_data_dir, 'infrastructure_macro_base.parquet')
        shared_master.to_parquet(output_path)
        
        logger.info(f"SUCCESS: Shared Macro Foundation saved. Shape: {shared_master.shape}")
        return shared_master

if __name__ == "__main__":
    collector = InfrastructureMacroCollector()
    df = collector.build_sector_foundation()
    print("\nSector 2 Macro Foundation (Latest Data):")
    print(df.tail())