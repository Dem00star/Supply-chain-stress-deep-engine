import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterDataSynthesizer:
    """
    Fuses the independent Financial, Geopolitical (Regime), and Physical
    (Graph) data pipelines into a single, time-aligned master matrix.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'raw')
        self.processed_data_dir = os.path.join(self.base_dir, 'data', 'processed')

    def fuse_pipelines(self):
        logger.info("Synthesizing multi-dimensional AI layers...")
        
        # Load Layer 2 & 3 (Financial + Regimes)
        fin_path = os.path.join(self.processed_data_dir, 'financial_regimes.parquet')
        df_financial = pd.read_parquet(fin_path)
        
        # Load Layer 1 (Physical Graph Stress)
        logistics_path = os.path.join(self.raw_data_dir, 'chokepoint_stress.parquet')
        df_logistics = pd.read_parquet(logistics_path)
        
        # Align timelines using an inner join on the Date index
        master_df = df_financial.join(df_logistics, how='inner')
        
        # Forward-fill any minor gaps (like missing weekend shipping data vs active commodity trading)
        master_df.ffill(inplace=True)
        master_df.dropna(inplace=True)
        
        # Save the finalized training matrix
        output_path = os.path.join(self.processed_data_dir, 'master_feature_set.parquet')
        master_df.to_parquet(output_path)
        
        logger.info(f"SUCCESS: Master Feature Matrix compiled. Final Shape: {master_df.shape}")
        return master_df

if __name__ == "__main__":
    synthesizer = MasterDataSynthesizer()
    df = synthesizer.fuse_pipelines()
    
    print("\nMeta-Learner Training Matrix (Latest 5 Days):")
    # Display a cross-section of the different layers
    cols_to_show = ['WTI_Spot', 'Brent_WTI_Spread', 'Market_Regime', 'Suez_Canal_Stress_Idx']
    print(df[cols_to_show].tail())