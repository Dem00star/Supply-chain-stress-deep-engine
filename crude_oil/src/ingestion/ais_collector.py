import os
import yaml
import logging
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TankerAISCollector:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(self.base_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.chokepoints = self.config['data_sources'].get('chokepoints', ['Suez_Canal', 'Strait_of_Hormuz'])
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'raw')
        
        # In a live production environment, this would hit the MarineTraffic or IMF PortWatch API
        self.api_key = os.getenv("MARINETRAFFIC_API_KEY")

    def fetch_chokepoint_congestion(self, days_back=30):
        """Pulls maritime congestion metrics for global energy chokepoints."""
        logger.info(f"Targeting Energy Chokepoints: {self.chokepoints}")
        
        # ---------------------------------------------------------
        # PORTFOLIO MOCK: For the sake of having a runnable repository
        # without paying for a $500/mo Maritime API, we generate 
        # synthetically sound structural data to feed the neural network.
        # In production, replace this block with an active requests.get()
        # ---------------------------------------------------------
        
        dates = [datetime.now().date() - timedelta(days=i) for i in range(days_back)]
        data = {'Date': dates}
        
        import numpy as np
        for point in self.chokepoints:
            # Generate baseline vessel counts with some random volatility
            baseline = 50 if point == 'Suez_Canal' else 80
            data[f'{point}_vessels_waiting'] = np.random.normal(baseline, 10, days_back).astype(int)
            
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        output_path = os.path.join(self.raw_data_dir, 'ais_data_raw.parquet')
        df.to_parquet(output_path)
        logger.info(f"SUCCESS: Saved AIS chokepoint data to {output_path}")
        return df

if __name__ == "__main__":
    collector = TankerAISCollector()
    collector.fetch_chokepoint_congestion()