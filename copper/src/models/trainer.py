import os
import shutil
import logging
import torch
from datetime import datetime
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from dataset import BDIDatasetBuilder
from tft_model import create_tft_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def run_training(self):
        logger.info("Starting ML Training Pipeline...")
        
        # 1. Load the data
        builder = BDIDatasetBuilder()
        training_dataset, raw_df = builder.create_dataset()
        if not training_dataset:
            return

        # 2. Split into Train & Validation (Hold out the last 10% of time for testing)
        validation_split_idx = int(len(raw_df) * 0.90)
        validation_dataset = training_dataset.from_dataset(
            training_dataset,
            raw_df[validation_split_idx:],
            stop_randomization=True
        )

        # 3. Create DataLoaders (Feed data in batches of 32)
        batch_size = 32
        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

        # 4. Initialize the Model
        model = create_tft_model(training_dataset)

        # 5. Set up PyTorch Lightning Callbacks
        # EarlyStopping: Stops training if the model stops improving (saves time & prevents overfitting)
        early_stop_callback = EarlyStopping(
            monitor="val_loss", 
            min_delta=1e-4, 
            patience=5, 
            verbose=True, 
            mode="min"
        )
        
        # ModelCheckpoint: Automatically saves the weights of the best performing epoch
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.models_dir,
            filename="best_tft_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        # 6. Configure the Trainer
        # Detect Apple Silicon (MPS) for GPU acceleration, fallback to CPU
        accelerator = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using compute accelerator: {accelerator.upper()}")
        
        trainer = pl.Trainer(
            max_epochs=10, # Kept low (10) for quick portfolio demonstration. In real life, use 50-100.
            accelerator=accelerator,
            devices=1,
            enable_model_summary=True,
            callbacks=[early_stop_callback, checkpoint_callback],
            log_every_n_steps=5
        )

        # 7. TRAIN!
        logger.info("Commencing model training loop...")
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # 8. MLOps: Model Registry Automation
        best_run_path = checkpoint_callback.best_model_path

        if best_run_path:
            logger.info(f"Training complete! Best run natively saved at: {best_run_path}")
            
            # Define directories
            archive_dir = os.path.join(self.models_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            
            production_model_path = os.path.join(self.models_dir, "best_tft_model.ckpt")
            
            # If a production model already exists and it's NOT the one we just saved
            if os.path.exists(production_model_path) and best_run_path != production_model_path:
                # Archive the old one
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archived_name = f"tft_model_archive_{timestamp}.ckpt"
                archived_path = os.path.join(archive_dir, archived_name)
                
                shutil.move(production_model_path, archived_path)
                logger.info(f"Archived old production model to: {archived_path}")
                
                # Promote the new one to the exact name the API expects
                shutil.copy2(best_run_path, production_model_path)
                
                # Delete the native lightning file (e.g., best_tft_model-v1.ckpt) to keep the folder clean
                os.remove(best_run_path)
                logger.info(f"SUCCESS: Promoted new model to production -> {production_model_path}")
                
            elif best_run_path == production_model_path:
                logger.info(f"SUCCESS: Model saved directly to production -> {production_model_path}")
        else:
            logger.warning("Training finished but no best model path was found!")

if __name__ == "__main__":
    # Ignore some noisy PyTorch Lightning warnings for a cleaner terminal output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    trainer_engine = ModelTrainer()
    trainer_engine.run_training()