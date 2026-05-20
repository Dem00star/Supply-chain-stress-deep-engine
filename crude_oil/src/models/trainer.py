import os
import shutil
import logging
import torch
import yaml
from datetime import datetime
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from dataset import EnergyDatasetBuilder
from tft_model import create_tft_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyModelTrainer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        config_path = os.path.join(self.base_dir, 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def run_training(self):
        logger.info("Starting ML Training Pipeline for Energy Engine...")
        
        builder = EnergyDatasetBuilder()
        training_dataset, raw_df = builder.create_dataset()
        if not training_dataset: return

        # Validation Split (Last 10%)
        validation_split_idx = int(len(raw_df) * 0.90)
        validation_dataset = training_dataset.from_dataset(
            training_dataset,
            raw_df[validation_split_idx:],
            stop_randomization=True
        )

        batch_size = self.config['tft_hyperparameters']['batch_size']
        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

        model = create_tft_model(training_dataset)

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.models_dir, filename="best_energy_tft_model", monitor="val_loss", mode="min", save_top_k=1
        )

        accelerator = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using compute accelerator: {accelerator.upper()}")
        
        trainer = pl.Trainer(
            max_epochs=10, 
            accelerator=accelerator,
            devices=1,
            callbacks=[early_stop_callback, checkpoint_callback],
            log_every_n_steps=5
        )

        logger.info("Commencing Crude Oil model training loop...")
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        # MLOps: Model Registry Automation
        best_run_path = checkpoint_callback.best_model_path
        if best_run_path:
            archive_dir = os.path.join(self.models_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            production_model_path = os.path.join(self.models_dir, "best_energy_tft_model.ckpt")
            
            if os.path.exists(production_model_path) and best_run_path != production_model_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.move(production_model_path, os.path.join(archive_dir, f"energy_tft_archive_{timestamp}.ckpt"))
                shutil.copy2(best_run_path, production_model_path)
                os.remove(best_run_path)
            elif best_run_path == production_model_path:
                logger.info(f"SUCCESS: Model saved directly to production -> {production_model_path}")
        else:
            logger.warning("Training finished but no best model path was found!")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    trainer_engine = EnergyModelTrainer()
    trainer_engine.run_training()