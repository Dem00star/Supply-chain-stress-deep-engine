import os
import logging
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CopperDataEngine:
    """
    Ingests asset-specific physical pricing for Copper
    and merges it with the Sector 2 Shared Macro Foundation.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.shared_macro_path = os.path.join(self.base_dir, 'data', 'raw', 'shared_macro', 'infrastructure_macro_base.parquet')
        
        self.engine_data_dir = os.path.join(self.base_dir, 'data', 'processed', 'copper')
        os.makedirs(self.engine_data_dir, exist_ok=True)

    def fetch_asset_specific_proxies(self, start_date="2018-01-01"):
        logger.info("Pulling Copper Specific Proxies (COMEX Futures)...")
        
        # HG=F: COMEX Copper Futures (The global benchmark for copper pricing)
        df = yf.download(["HG=F"], start=start_date, progress=False)['Close']
        
        # Handle yfinance multi-index if present
        if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # If it returns a Series (single ticker), convert to DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame(name="HG=F")
            
        df.rename(columns={"HG=F": "Copper_Close"}, inplace=True)
        
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.ffill(inplace=True)
        return df

    def synthesize_copper_matrix(self):
        logger.info("Fusing Copper Pricing with Sector 2 Shared Macro...")
        
        asset_df = self.fetch_asset_specific_proxies()
        
        if not os.path.exists(self.shared_macro_path):
            logger.error("Shared Macro file not found. Run macro_collector.py first.")
            return None
            
        shared_df = pd.read_parquet(self.shared_macro_path)
        
        master_df = asset_df.join(shared_df, how='inner')
        master_df.ffill(inplace=True)
        master_df.dropna(inplace=True)
        
        output_path = os.path.join(self.engine_data_dir, 'copper_feature_set.parquet')
        master_df.to_parquet(output_path)
        
        logger.info(f"SUCCESS: Copper Matrix synthesized. Final Shape: {master_df.shape}")
        return master_df

if __name__ == "__main__":
    engine = CopperDataEngine()
    df = engine.synthesize_copper_matrix()
    if df is not None:
        print("\nCopper Training Matrix (Asset Proxy + Shared Macro):")
        cols_to_show = ['Copper_Close', 'Copper_Miners_Index', 'US_Dollar_Index', 'Global_Mining_Health']
        print(df[cols_to_show].tail())