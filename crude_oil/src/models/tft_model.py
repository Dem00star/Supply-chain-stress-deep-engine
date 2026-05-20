import yaml
import os
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss

def create_tft_model(training_dataset):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    hp = config['tft_hyperparameters']
    
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=hp['learning_rate'],
        hidden_size=hp['hidden_size'],
        attention_head_size=hp['attention_head_size'],
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(quantiles=hp['quantiles']), # P10, P50, P90 Risk Bounds
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    return model