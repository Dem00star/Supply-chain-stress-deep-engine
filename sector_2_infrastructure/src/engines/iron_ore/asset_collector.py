import os
import logging
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IronOreDataEngine:
    """
    Ingests asset-specific physical and currency proxies for Iron Ore
    and merges them with the Sector 2 Shared Macro Foundation.
    """
    def __init__(self):
        # Resolve paths for the new sector-based architecture
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Where to find the shared macro data we already pulled
        self.shared_macro_path = os.path.join(self.base_dir, 'data', 'raw', 'shared_macro', 'infrastructure_macro_base.parquet')
        
        # Where to save this specific engine's data
        self.engine_data_dir = os.path.join(self.base_dir, 'data', 'processed', 'iron_ore')
        os.makedirs(self.engine_data_dir, exist_ok=True)

    def fetch_asset_specific_proxies(self, start_date="2018-01-01"):
        logger.info("Pulling Iron Ore Specific Proxies (Australian Supply & Currency)...")
        
        # BHP: World's largest mining company (Massive Iron Ore exposure)
        # RIO: Rio Tinto (Massive Iron Ore exposure)
        # AUDUSD=X: Australian Dollar to US Dollar (Currency demand proxy)
        tickers = ["BHP", "RIO", "AUDUSD=X"]
        
        df = yf.download(tickers, start=start_date, progress=False)['Close']
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.rename(columns={
            "BHP": "BHP_Supply_Proxy",
            "RIO": "RIO_Supply_Proxy",
            "AUDUSD=X": "AUD_USD_Exchange"
        }, inplace=True)
        
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.ffill(inplace=True)
        return df

    def synthesize_iron_ore_matrix(self):
        logger.info("Fusing Iron Ore Proxies with Sector 2 Shared Macro...")
        
        # 1. Get the asset-specific data
        asset_df = self.fetch_asset_specific_proxies()
        
        # 2. Get the shared sector macro data (China, US Dollar, Dry Bulk Index)
        if not os.path.exists(self.shared_macro_path):
            logger.error("Shared Macro file not found. Run macro_collector.py first.")
            return None
            
        shared_df = pd.read_parquet(self.shared_macro_path)
        
        # 3. Merge them based on the Date index
        master_df = asset_df.join(shared_df, how='inner')
        master_df.ffill(inplace=True)
        master_df.dropna(inplace=True)
        
        # Save the finalized feature set for the PyTorch Meta-Learner
        output_path = os.path.join(self.engine_data_dir, 'iron_ore_feature_set.parquet')
        master_df.to_parquet(output_path)
        
        logger.info(f"SUCCESS: Iron Ore Matrix synthesized. Final Shape: {master_df.shape}")
        return master_df

if __name__ == "__main__":
    engine = IronOreDataEngine()
    df = engine.synthesize_iron_ore_matrix()
    
    if df is not None:
        print("\nIron Ore Training Matrix (Asset Proxies + Shared Macro):")
        # Display a cross-section of the new fused matrix
        cols_to_show = ['BHP_Supply_Proxy', 'AUD_USD_Exchange', 'Dry_Bulk_Freight_Index', 'US_Dollar_Index']
        print(df[cols_to_show].tail())