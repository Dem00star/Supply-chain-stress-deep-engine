import os
import logging
import warnings
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from dataset import EnergyDatasetBuilder

# Suppress PyTorch Lightning warnings for a clean console
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train(self):
        logger.info("Initializing Energy Fleet Meta-Learner (TFT)...")
        
        # 1. Build DataLoaders
        builder = EnergyDatasetBuilder()
        df, continuous_cols = builder.load_and_prep_data()
        training_dataset, validation_dataset = builder.create_dataset(df, continuous_cols)

        batch_size = 64
        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

        # 2. Define the Neural Network Architecture
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss([0.1, 0.5, 0.9]), # Predicts Risk Floor, Expected Price, and Risk Ceiling
            optimizer="Adam"
        )

        # 3. Setup Callbacks (Early Stopping to prevent overfitting)
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_dir,
            filename="best_energy_tft_model",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )

        # 4. Initialize Cloud-Safe Trainer
        logger.info("Beginning Deep Learning Optimization Phase (Forced CPU for Cloud Compatibility)...")
        trainer = pl.Trainer(
            max_epochs=15,
            accelerator="cpu", # GUARANTEES STREAMLIT COMPATIBILITY
            enable_model_summary=False,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=False
        )

        # 5. Train!
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # PyTorch Lightning appends '-v1.ckpt' to the file, let's rename it cleanly
        ckpt_files = [f for f in os.listdir(self.model_dir) if f.startswith("best_energy_tft_model")]
        if ckpt_files:
            latest_ckpt = os.path.join(self.model_dir, ckpt_files[0])
            clean_ckpt = os.path.join(self.model_dir, "best_energy_tft_model.ckpt")
            os.replace(latest_ckpt, clean_ckpt)
            
        logger.info(f"SUCCESS: Global Energy Model trained and saved to {self.model_dir}/best_energy_tft_model.ckpt")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()