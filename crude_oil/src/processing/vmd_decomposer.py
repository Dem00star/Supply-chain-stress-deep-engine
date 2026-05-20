import os
import yaml
import logging
import pandas as pd
import numpy as np
from vmdpy import VMD

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyVMD:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(self.base_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.raw_market_path = os.path.join(self.base_dir, 'data', 'raw', 'market_data_raw.parquet')
        self.processed_dir = os.path.join(self.base_dir, 'data', 'processed')
        self.target_col = self.config['data_sources']['fred_series']['target']
        self.vmd_params = self.config['vmd_parameters']

    def decompose(self):
        """Applies VMD to extract the true underlying macro trend of Crude Oil."""
        logger.info("Loading raw WTI Crude Oil prices...")
        df = pd.read_parquet(self.raw_market_path)
        
        signal = df[self.target_col].values
        
        logger.info(f"Applying VMD (K={self.vmd_params['K']}) to {self.target_col}...")
        u, u_hat, omega = VMD(
            signal, 
            self.vmd_params['alpha'], 
            self.vmd_params['tau'], 
            self.vmd_params['K'], 
            self.vmd_params['DC'], 
            self.vmd_params['init'], 
            self.vmd_params['tol']
        )
        
        # Create a DataFrame for the extracted Intrinsic Mode Functions (IMFs)
        valid_length = u.shape[1]
        vmd_df = pd.DataFrame(index=df.index[:valid_length])
        for i in range(self.vmd_params['K']):
            vmd_df[f'{self.target_col}_mode_{i+1}'] = u[i, :]
            
        output_path = os.path.join(self.processed_dir, 'vmd_features.parquet')
        vmd_df.to_parquet(output_path)
        logger.info(f"SUCCESS: VMD features saved to {output_path}")

if __name__ == "__main__":
    vmd = EnergyVMD()
    vmd.decompose()