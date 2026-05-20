import os
import yaml
import logging
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhysicalGraphBuilder:
    """
    Constructs a spatial-temporal graph of the global oil supply chain.
    Nodes = Basins/Refineries. Edges = Chokepoints.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(self.base_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'raw')
        self.physical_cfg = self.config['physical_layer']
        
        # We align the graph timeline with your financial data
        financial_path = os.path.join(self.raw_data_dir, 'financial_curve_raw.parquet')
        if os.path.exists(financial_path):
            self.dates = pd.read_parquet(financial_path).index
        else:
            self.dates = pd.date_range(start="2018-01-01", end=datetime.today(), freq='B')

    def build_static_graph(self):
        """Creates the physical mapping of the global oil trade."""
        logger.info("Initializing Directed Graph for Global Maritime Routes...")
        G = nx.DiGraph()
        
        # Add Nodes (Supply & Demand Hubs)
        for node in self.physical_cfg['nodes']:
            if "USA" in node or "Saudi" in node:
                G.add_node(node, type="supply")
            else:
                G.add_node(node, type="demand")
                
        # Add Edges (The Shipping Chokepoints with baseline transit days)
        # Ghawar -> Rotterdam via Suez
        G.add_edge("Ghawar_Saudi_Arabia", "Rotterdam_Netherlands", 
                   chokepoint="Suez_Canal", baseline_transit_days=14)
        
        # Ghawar -> Asia (not explicitly in config, but passing via Malacca)
        G.add_edge("Ghawar_Saudi_Arabia", "Houston_USA", 
                   chokepoint="Strait_of_Hormuz", baseline_transit_days=25)
                   
        # Save the static graph structure
        graph_path = os.path.join(self.raw_data_dir, 'maritime_network.gpickle')
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)
            
        logger.info(f"Graph constructed with {G.number_of_nodes()} hubs and {G.number_of_edges()} critical edges.")
        return G

    def generate_chokepoint_stress(self):
        """
        Generates a time-series stress index for each physical chokepoint.
        In a production $10k/mo Bloomberg setup, this pulls live AIS vessel delays.
        For this engine, we synthesize baseline volatility with historical shock alignment.
        """
        logger.info("Calculating Spatial-Temporal Edge Weights (Chokepoint Stress)...")
        np.random.seed(42)
        
        df_stress = pd.DataFrame(index=self.dates)
        
        for chokepoint in self.physical_cfg['edges']:
            # Base stress is a mean-reverting random walk representing normal maritime traffic
            base_noise = np.random.normal(loc=1.0, scale=0.05, size=len(self.dates))
            stress_series = pd.Series(base_noise, index=self.dates)
            
            # Smooth the noise to represent slow-moving shipping fleets
            stress_series = stress_series.rolling(window=5, min_periods=1).mean()
            df_stress[f"{chokepoint}_Stress_Idx"] = stress_series

        # Inject historical reality: 2021 Suez Canal Blockage (Ever Given)
        # This teaches the AI what a catastrophic physical blockage looks like
        mask_suez = (df_stress.index >= '2021-03-23') & (df_stress.index <= '2021-04-10')
        df_stress.loc[mask_suez, 'Suez_Canal_Stress_Idx'] *= 3.5 
        
        # Inject historical reality: 2023/2024 Red Sea Houthi Attacks
        mask_red_sea = (df_stress.index >= '2023-11-15')
        df_stress.loc[mask_red_sea, 'Suez_Canal_Stress_Idx'] *= 1.8 

        output_path = os.path.join(self.raw_data_dir, 'chokepoint_stress.parquet')
        df_stress.to_parquet(output_path)
        
        logger.info(f"SUCCESS: Physical logistics matrix saved. Shape: {df_stress.shape}")
        return df_stress

if __name__ == "__main__":
    builder = PhysicalGraphBuilder()
    G = builder.build_static_graph()
    stress_df = builder.generate_chokepoint_stress()
    
    print("\nPhysical Graph Edge Weights (Chokepoint Stress):")
    print(stress_df.tail())