import os
import yaml
import logging
import pandas as pd
import numpy as np
import joblib
from hmmlearn import hmm
import warnings
from sklearn.preprocessing import StandardScaler

# Suppress sklearn deprecation warnings from hmmlearn
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegimeSwitcher:
    """
    Applies a Hidden Markov Model (HMM) to classify the oil market 
    into discrete volatility regimes (e.g., Calm, Panic, Shock).
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(self.base_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'raw')
        self.processed_data_dir = os.path.join(self.base_dir, 'data', 'processed')
        self.model_dir = os.path.join(self.base_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.n_regimes = self.config['geopolitical_layer']['markov_regimes']

    def calculate_features(self, df):
        """Calculates Log Returns and Volatility as HMM inputs."""
        df = df.copy()
        # Log Returns: The continuous compounding rate of return
        df['Log_Return'] = np.log(df['WTI_Spot'] / df['WTI_Spot'].shift(1))
        
        # Volatility: 7-day rolling standard deviation of returns
        df['Volatility_7d'] = df['Log_Return'].rolling(window=7).std()
        
        df.dropna(inplace=True)
        return df

    def fit_predict_regimes(self):
        logger.info("Loading Financial Base Layer...")
        input_path = os.path.join(self.raw_data_dir, 'financial_curve_raw.parquet')
        df = pd.read_parquet(input_path)
        
        logger.info("Calculating Volatility Features for HMM...")
        df = self.calculate_features(df)
        
        # We train the HMM purely on price velocity and volatility
        X_raw = df[['Log_Return', 'Volatility_7d']].values
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        
        logger.info(f"Training Gaussian Hidden Markov Model with {self.n_regimes} Regimes...")
        # covariance_type="full" allows the model to capture complex relationships
        model = hmm.GaussianHMM(n_components=self.n_regimes, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X)
        
        # Predict the hidden states (0, 1, or 2)
        hidden_states = model.predict(X)
        df['Market_Regime'] = hidden_states
        
        # Save the mathematical model so the live API can use it later
        model_path = os.path.join(self.model_dir, 'hmm_regime_model.pkl')
        joblib.dump(model, model_path)
        
        # Save the enriched dataset
        output_path = os.path.join(self.processed_data_dir, 'financial_regimes.parquet')
        df.to_parquet(output_path)
        
        logger.info(f"SUCCESS: Market Regimes classified and saved. Shape: {df.shape}")
        
        # Calculate summary statistics per regime to interpret them
        summary = df.groupby('Market_Regime')[['Log_Return', 'Volatility_7d']].mean()
        print("\n--- Regime Characteristics (Mean Values) ---")
        print("Analyze these to determine which is 'Calm' and which is 'Panic':")
        print(summary)
        
        return df

if __name__ == "__main__":
    switcher = RegimeSwitcher()
    switcher.fit_predict_regimes()