import logging
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

logger = logging.getLogger(__name__)

def create_tft_model(training_dataset):
    """
    Initializes the Temporal Fusion Transformer architecture directly 
    from the PyTorch dataset specifications.
    """
    logger.info("Initializing Temporal Fusion Transformer...")
    
    # We use hyperparameters optimized for a local machine (MacBook Air)
    # In a full cloud production run, you would increase hidden_size and attention_head_size
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=16,           # Capacity of the network's memory
        attention_head_size=1,    # Number of attention mechanisms
        dropout=0.1,              # Prevents overfitting
        hidden_continuous_size=8, 
        output_size=7,            # Predicts 7 quantiles (e.g., 2%, 10%, 25%, 50%, 75%, 90%, 98%)
        loss=QuantileLoss(),      # The mathematical function that penalizes wrong predictions
        log_interval=10, 
        reduce_on_plateau_patience=4,
    )
    
    logger.info(f"Model built! Total trainable parameters: {model.size()/1e3:.1f}k")
    return model