#!/usr/bin/env python3
"""
Automated Experiment Runner for BanglaBERT Hate Speech Detection
This script provides a complete automated pipeline for running experiments in Google Colab
with automatic Google Drive mounting, dependency installation, and MLflow logging.
"""

import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any
import torch
import mlflow

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, get_default_config, CONFIG_PRESETS
from utils import setup_environment, get_device, log_system_info
import data
import train

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('experiment.log')
    ]
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Main experiment runner class"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = None
        self.tokenizer = None
        self.texts = None
        self.labels = None
        
    def setup(self) -> bool:
        """Complete setup pipeline"""
        logger.info("🚀 Starting experiment setup...")
        
        # Step 1: Environment setup
        if not setup_environment(self.config):
            logger.error("❌ Environment setup failed")
            return False
        
        # Step 2: Device setup
        self.device = get_device(self.config)
        
        # Step 3: Load and preprocess data
        if not self.load_data():
            logger.error("❌ Data loading failed")
            return False
        
        # Step 4: Setup tokenizer
        if not self.setup_tokenizer():
            logger.error("❌ Tokenizer setup failed")
            return False
        
        logger.info("✅ Experiment setup completed successfully")
        return True
    
    def load_data(self) -> bool:
        """Load and preprocess training data"""
        try:
            logger.info(f"📊 Loading data from {self.config.dataset_path}")
            
            # Check if dataset exists
            if not os.path.exists(self.config.dataset_path):
                logger.error(f"❌ Dataset not found: {self.config.dataset_path}")
                return False
            
            # Load and preprocess data
            self.texts, self.labels = data.load_and_preprocess_data(self.config.dataset_path)
            
            logger.info(f"✅ Data loaded successfully:")
            logger.info(f"   Total samples: {len(self.texts)}")
            logger.info(f"   Text shape: {self.texts.shape}")
            logger.info(f"   Labels shape: {self.labels.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def setup_tokenizer(self) -> bool:
        """Setup BERT tokenizer"""
        try:
            from transformers import BertTokenizer
            
            logger.info(f"🔤 Loading tokenizer: {self.config.model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
            
            logger.info("✅ Tokenizer loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading tokenizer: {str(e)}")
            return False
    
    def run_experiment(self) -> bool:
        """Run the complete experiment"""
        logger.info("🔬 Starting experiment execution...")
        
        try:
            # Start MLflow run
            with mlflow.start_run(run_name=self.get_run_name()) as run:
                logger.info(f"📊 MLflow run started: {run.info.run_id}")
                
                # Log system information
                system_info = log_system_info(self.config)
                mlflow.log_params(system_info)
                
                # Log configuration parameters
                config_dict = self.config.to_dict()
                mlflow.log_params(config_dict)
                
                # Log dataset information
                self.log_dataset_info()
                
                # Run training
                logger.info("🏋️  Starting training...")
                results = train.run_kfold_training(
                    self.config, 
                    self.texts, 
                    self.labels, 
                    self.tokenizer, 
                    self.device
                )
                
                # Log final results
                if results:
                    self.log_final_results(results)
                    logger.info("✅ Experiment completed successfully!")
                    return True
                else:
                    logger.error("❌ Training failed")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error during experiment: {str(e)}")
            return False
    
    def get_run_name(self) -> str:
        """Generate a descriptive run name"""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        gpu_info = "GPU" if self.config.use_gpu and torch.cuda.is_available() else "CPU"
        
        return f"{self.config.author_name}_{self.config.batch_size}batch_{self.config.lr}lr_{self.config.num_epochs}epochs_{gpu_info}_{timestamp}"
    
    def log_dataset_info(self):
        """Log dataset statistics to MLflow"""
        try:
            # Calculate dataset statistics
            hate_speech_dist = {
                "hate": int(sum(self.labels[:, 0])),
                "nonhate": int(len(self.labels) - sum(self.labels[:, 0]))
            }
            
            emotion_dist = {
                "sad": int(sum(self.labels[:, 1] == 0)),
                "angry": int(sum(self.labels[:, 1] == 1)),
                "happy": int(sum(self.labels[:, 1] == 2))
            }
            
            # Log to MLflow
            mlflow.log_metrics({
                "dataset_total_samples": len(self.texts),
                "dataset_hate_speech_ratio": hate_speech_dist["hate"] / len(self.labels),
                "dataset_emotion_sad_ratio": emotion_dist["sad"] / len(self.labels),
                "dataset_emotion_angry_ratio": emotion_dist["angry"] / len(self.labels),
                "dataset_emotion_happy_ratio": emotion_dist["happy"] / len(self.labels)
            })
            
            # Log as artifacts
            import json
            with open("dataset_stats.json", "w") as f:
                json.dump({
                    "hate_speech_distribution": hate_speech_dist,
                    "emotion_distribution": emotion_dist,
                    "total_samples": len(self.texts)
                }, f, indent=2)
            
            mlflow.log_artifact("dataset_stats.json")
            
            logger.info("📊 Dataset statistics logged to MLflow")
            
        except Exception as e:
            logger.error(f"❌ Error logging dataset info: {str(e)}")
    
    def log_final_results(self, results: Dict[str, Any]):
        """Log final experiment results"""
        try:
            # Calculate average metrics across folds
            avg_metrics = {}
            for key in results[0].keys():
                if key.startswith('val_'):
                    avg_metrics[f"avg_{key}"] = sum(r[key] for r in results) / len(results)
            
            # Log average metrics
            mlflow.log_metrics(avg_metrics)
            
            # Log detailed results as artifact
            import json
            with open("experiment_results.json", "w") as f:
                json.dump({
                    "fold_results": results,
                    "average_metrics": avg_metrics,
                    "config": self.config.to_dict()
                }, f, indent=2)
            
            mlflow.log_artifact("experiment_results.json")
            
            # Print summary
            logger.info("📈 Final Results Summary:")
            for metric, value in avg_metrics.items():
                logger.info(f"   {metric}: {value:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error logging final results: {str(e)}")

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Automated BanglaBERT Hate Speech Detection Experiment Runner"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick_test", "production", "research"],
        help="Use a configuration preset"
    )
    
    parser.add_argument(
        "--experiment_type",
        type=str,
        choices=["standard", "fast", "high_performance"],
        default="standard",
        help="Type of experiment to run"
    )
    
    parser.add_argument(
        "--author_name",
        type=str,
        default="colab_experiment",
        help="Author name for experiment tracking"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to dataset file"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs"
    )
    
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU usage"
    )
    
    parser.add_argument(
        "--no_drive",
        action="store_true",
        help="Disable Google Drive usage"
    )
    
    return parser

def main():
    """Main execution function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = ExperimentConfig.load_config(args.config)
    elif args.preset:
        logger.info(f"Using preset configuration: {args.preset}")
        config = ExperimentConfig.from_dict(CONFIG_PRESETS[args.preset])
    else:
        logger.info(f"Using default configuration: {args.experiment_type}")
        config = get_default_config(args.experiment_type)
    
    # Override configuration with command line arguments
    if args.author_name:
        config.author_name = args.author_name
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.epochs:
        config.num_epochs = args.epochs
    if args.no_gpu:
        config.use_gpu = False
    if args.no_drive:
        config.use_google_drive = False
    
    # Create and run experiment
    runner = ExperimentRunner(config)
    
    if runner.setup():
        success = runner.run_experiment()
        if success:
            logger.info("🎉 Experiment completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Experiment failed!")
            sys.exit(1)
    else:
        logger.error("❌ Experiment setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
