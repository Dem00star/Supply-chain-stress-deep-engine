import os
import yaml
import logging
import numpy as np
import pandas as pd
from vmdpy import VMD

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDecomposer:
    """
    Applies Variational Mode Decomposition (VMD) to strip market noise 
    from systemic signals. Decomposes a target variable into K discrete 
    intrinsic mode functions (IMFs).
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.raw_dir = os.path.join(self.base_dir, "data", "raw")
        self.processed_dir = os.path.join(self.base_dir, "data", "processed")
        
        # Load VMD hyperparameters from the central config
        config_path = os.path.join(self.base_dir, "config", "config.yaml")
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        # VMD specific settings
        # If not explicitly set in yaml, we fall back to standard quant defaults
        self.K = self.config.get("vmd_settings", {}).get("K", 4) # Number of modes to extract
        self.alpha = self.config.get("vmd_settings", {}).get("alpha", 2000) # Bandwidth constraint
        self.tau = 0. # Noise-tolerance (0 = no strict fidelity enforcement)
        self.DC = 0   # No DC part imposed
        self.init = 1 # Initialize omegas uniformly
        self.tol = 1e-7 # Tolerance of convergence

    def decompose_signal(self, series: pd.Series, prefix: str = "vmd") -> pd.DataFrame:
        """Splits a single time-series vector into K discrete frequencies."""
        # VMD requires a clean numpy array with no NaN values
        signal = series.ffill().bfill().values
        
        logger.info(f"Applying VMD (K={self.K}, alpha={self.alpha}) to {series.name}...")
        
        # Execute the VMD algorithm
        # u: decomposed modes, u_hat: spectra of the modes, omega: estimated mode center-frequencies
        u, u_hat, omega = VMD(
            f=signal,
            alpha=self.alpha,
            tau=self.tau, 
            K=self.K,
            DC=self.DC,
            init=self.init,
            tol=self.tol
        )
        
        # Build out a clean structured DataFrame mapped precisely to the original dates
        # Note: vmdpy sometimes truncates odd-length arrays by 1 due to FFT math. 
        # We slice the index to match the exact length of the returned array.
        valid_length = u.T.shape[0]
        mode_df = pd.DataFrame(
            u.T, 
            index=series.index[:valid_length], 
            columns=[f"{prefix}_mode_{i+1}" for i in range(self.K)]
        )
        return mode_df

    def process_target_variable(self):
        """Loads market data, selects a target variable, and decomposes it."""
        input_file = os.path.join(self.raw_dir, "market_data_raw.parquet")
        
        if not os.path.exists(input_file):
            logger.error(f"Missing {input_file}. Run market_collector.py first.")
            return None
            
        df = pd.read_parquet(input_file)
        
        # For this portfolio piece, we will use Copper as our proxy target variable 
        # (Dr. Copper is the ultimate leading indicator of global industrial health).
        target_col = "copper_close"
        
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in raw data.")
            return None
            
        # Decompose the target
        target_series = df[target_col]
        modes_df = self.decompose_signal(target_series, prefix="copper")
        
        # Save to the processed directory
        output_file = os.path.join(self.processed_dir, "vmd_features.parquet")
        modes_df.to_parquet(output_file)
        
        logger.info(f"Successfully saved VMD features to {output_file}")
        return modes_df

if __name__ == "__main__":
    decomposer = MarketDecomposer()
    df = decomposer.process_target_variable()
    if df is not None:
        print("\nVMD Decomposed Modes Snapshot:")
        print(df.tail())