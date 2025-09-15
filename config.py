import os
import argparse
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ExperimentConfig:
    """Configuration class for BanglaBERT hate speech detection experiments - Enhanced v2.0 for 93% F1 score"""
    
    # Model parameters
    model_name: str = "sagorsarker/bangla-bert-base"
    model_path: str = "sagorsarker/bangla-bert-base"
    use_enhanced_model: bool = True  # Enable enhanced model architecture
    
    # Training parameters - ENHANCED for 93% F1
    batch_size: int = 32          # Enhanced: Increased from 16 to 32 for better gradient estimation
    learning_rate: float = 2e-5
    num_epochs: int = 30          # Enhanced: Increased from 5 to 30 for full convergence
    num_folds: int = 5
    max_length: int = 256
    dropout: float = 0.3          # Enhanced: Increased from 0.1 to 0.3 for better regularization
    
    # Optimization parameters - ENHANCED
    weight_decay: float = 0.01
    warmup_steps: int = 500
    scheduler_type: str = "cosine"  # Enhanced: Changed from "linear" to "cosine" for better convergence
    
    # Model architecture
    freeze_base: bool = True
    multi_task: bool = True
    
    # Enhanced model architecture parameters
    attention_heads: int = 8       # Number of attention heads for multi-head attention
    cross_attention_heads: int = 4 # Number of attention heads for cross-attention
    stochastic_depth: float = 0.1  # Stochastic depth probability for regularization
    use_gelu_activation: bool = True  # Use GELU activation instead of ReLU
    use_layer_wise_lr: bool = False  # Enable layer-wise learning rate decay
    layer_wise_decay: float = 0.9   # Layer-wise learning rate decay factor
    
    # Data parameters
    dataset_path: str = "data/5_BanEmoHate.csv"
    test_size: float = 0.2
    random_state: int = 42
    
    # MLflow parameters
    mlflow_experiment_name: str = "BanglaBERT_HateSpeech_Detection_Enhanced"
    mlflow_tracking_uri: Optional[str] = None
    mlflow_log_model: bool = True
    mlflow_log_artifacts: bool = True
    
    # Google Drive parameters
    use_google_drive: bool = True
    drive_mount_point: str = "/content/drive"
    mlflow_drive_path: str = "/content/drive/MyDrive/bangla_bert_mlflow"
    
    # Experiment metadata
    author_name: str = "enhanced_experiment"
    experiment_description: str = "Enhanced BanglaBERT fine-tuning for hate speech detection with emotion classification - Target 93% F1"
    
    # Loss weights for multi-task learning - ENHANCED
    hate_speech_loss_weight: float = 0.6
    emotion_loss_weight: float = 0.4
    
    # Early stopping parameters - ENHANCED for longer training
    early_stopping_patience: int = 10  # Enhanced: Increased from 3 to 10 for 30-epoch training
    early_stopping_metric: str = "val_hate_f1"
    
    # GPU parameters
    use_gpu: bool = True
    mixed_precision: bool = True
    
    # Logging parameters
    log_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Set MLflow tracking URI based on environment
        if self.mlflow_tracking_uri is None:
            if self.use_google_drive and os.path.exists(self.drive_mount_point):
                self.mlflow_tracking_uri = f"file://{self.mlflow_drive_path}"
            else:
                self.mlflow_tracking_uri = "./mlruns"
        
        # Validate loss weights
        total_weight = self.hate_speech_loss_weight + self.emotion_loss_weight
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Loss weights sum to {total_weight}, normalizing to 1.0")
            self.hate_speech_loss_weight /= total_weight
            self.emotion_loss_weight /= total_weight
        
        # Validate enhanced model parameters
        if self.use_enhanced_model:
            print(f"Using enhanced model architecture with {self.attention_heads} attention heads")
            if self.stochastic_depth > 0:
                print(f"Enabled stochastic depth with probability {self.stochastic_depth}")
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Default configurations for different experiment types
def get_default_config(experiment_type: str = "standard") -> ExperimentConfig:
    """Get default configuration for different experiment types"""
    
    if experiment_type == "standard":
        return ExperimentConfig()
    
    elif experiment_type == "fast":
        return ExperimentConfig(
            batch_size=32,
            num_epochs=3,
            num_folds=3,
            max_length=128,
            log_steps=50,
            eval_steps=250,
            save_steps=500
        )
    
    elif experiment_type == "high_performance":
        return ExperimentConfig(
            batch_size=8,
            learning_rate=1e-5,
            num_epochs=10,
            num_folds=5,
            max_length=512,
            dropout=0.2,
            freeze_base=False,
            mixed_precision=True
        )
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

# Configuration presets for common use cases
CONFIG_PRESETS = {
    "quick_test": {
        "batch_size": 8,
        "num_epochs": 1,
        "num_folds": 2,
        "max_length": 64,
        "log_steps": 10,
        "eval_steps": 20,
        "save_steps": 50
    },
    "production": {
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 8,
        "num_folds": 5,
        "max_length": 256,
        "dropout": 0.3,
        "freeze_base": True,
        "mixed_precision": True
    },
    "research": {
        "batch_size": 12,
        "learning_rate": 1e-5,
        "num_epochs": 15,
        "num_folds": 10,
        "max_length": 512,
        "dropout": 0.2,
        "freeze_base": False,
        "mixed_precision": True
    },
    "enhanced": {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 30,
        "num_folds": 5,
        "max_length": 256,
        "dropout": 0.3,
        "scheduler_type": "cosine",
        "early_stopping_patience": 10
    }
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="BanglaBERT Hate Speech Detection")
    parser.add_argument("--experiment_type", type=str, default="standard", help="Experiment type (standard, fast, high_performance)")
    parser.add_argument("--config_preset", type=str, default=None, help="Configuration preset (quick_test, production, research, enhanced)")
    args = parser.parse_args()
    if args.config_preset:
        return ExperimentConfig(**CONFIG_PRESETS[args.config_preset])
    else:
        return get_default_config(args.experiment_type)
