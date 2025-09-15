# BanglaBERT Hate Speech Detection - Modular Structure Guide

## 📁 Project Structure

This project has been modularized to provide a clean, maintainable, and scalable structure for running BanglaBERT hate speech detection experiments in Google Colab.

### Core Modules

```
Finetune-Bangla-BERT-on-Bangla-hate-speech-Data/
├── config.py              # Configuration management
├── utils.py               # Utility functions (Drive, MLflow, GPU)
├── data.py                # Data loading and preprocessing
├── model.py               # Model architecture
├── train.py               # Training logic
├── run_experiment.py      # Automated experiment runner
├── colab_automated.py     # Simple Colab script
├── colab_training.py      # Advanced Colab script
├── colab_simple.py        # Manual Colab script
├── COLAB_GUIDE.md         # Detailed Colab guide
├── MODULAR_GUIDE.md       # This guide
├── README.md              # Project README
└── data/
    └── 5_BanEmoHate.csv   # Dataset
```

## 🚀 Quick Start in Google Colab

### Option 1: Simple Automated (Recommended)

```python
# Mount Google Drive and run the experiment
!python colab_automated.py
```

### Option 2: Advanced Automated

```python
# Run with custom configuration
!python colab_training.py
```

### Option 3: Manual Step-by-Step

```python
# Run with manual control
!python colab_simple.py
```

## 📋 Module Descriptions

### 1. `config.py` - Configuration Management

**Purpose**: Centralized configuration management for all experiment parameters.

**Features**:
- Dataclass-based configuration with type hints
- Default configurations for different experiment types
- Configuration presets (quick_test, production, research)
- JSON save/load functionality
- Automatic validation and normalization

**Usage**:
```python
from config import ExperimentConfig, get_default_config

# Use default configuration
config = get_default_config("standard")

# Use preset
config = ExperimentConfig.from_dict(CONFIG_PRESETS["production"])

# Custom configuration
config = ExperimentConfig(
    batch_size=32,
    learning_rate=1e-5,
    num_epochs=10
)
```

### 2. `utils.py` - Utility Functions

**Purpose**: Common utility functions for environment setup, Google Drive mounting, and MLflow configuration.

**Features**:
- Google Drive mounting with error handling
- MLflow setup with Google Drive backup
- Automatic dependency installation
- Environment detection (Colab vs local)
- GPU availability checking
- System information logging

**Key Functions**:
```python
# Mount Google Drive
success = mount_google_drive("/content/drive")

# Setup MLflow with Drive backup
success, mlflow_uri = setup_mlflow_with_drive(config)

# Complete environment setup
success = setup_environment(config)

# Get appropriate device
device = get_device(config)
```

### 3. `data.py` - Data Loading and Preprocessing

**Purpose**: Handle dataset loading, preprocessing, and validation with robust error handling.

**Features**:
- Automatic label validation and correction
- Bangla text cleaning and preprocessing
- Data augmentation support
- NaN handling and missing value imputation
- K-fold cross-validation preparation

**Key Improvements**:
- Fixed CUDA error by ensuring proper integer labels
- Added label validation and range checking
- Automatic correction of invalid labels
- Enhanced text cleaning for Bangla

### 4. `model.py` - Model Architecture

**Purpose**: BanglaBERT model with multi-task learning capabilities.

**Features**:
- Multi-task learning (hate speech + emotion classification)
- Enhanced classifier with multiple layers
- Proper loss calculation with debugging
- Automatic label correction during training
- Layer freezing/unfreezing support

**Key Improvements**:
- Added `multi_task` parameter
- Enhanced label validation in forward pass
- Automatic label clipping for invalid values
- Detailed debugging output

### 5. `train.py` - Training Logic

**Purpose**: Core training logic with k-fold cross-validation and comprehensive metrics.

**Features**:
- K-fold cross-validation
- Comprehensive metrics calculation
- Class weight balancing
- MLflow integration
- Early stopping support

### 6. `run_experiment.py` - Automated Experiment Runner

**Purpose**: Complete experiment automation with setup, execution, and logging.

**Features**:
- End-to-end experiment pipeline
- Automatic environment setup
- Comprehensive logging to MLflow
- Dataset statistics logging
- Results aggregation and reporting
- Command-line interface

**Usage**:
```python
# Basic usage
!python run_experiment.py

# With preset
!python run_experiment.py --preset production

# With custom parameters
!python run_experiment.py --batch_size 32 --epochs 10 --author_name my_experiment

# With configuration file
!python run_experiment.py --config my_config.json
```

## 🎯 Experiment Types

### 1. Standard (Default)
- Balanced configuration for general use
- 5-fold cross-validation
- Moderate training time
- Good performance baseline

### 2. Fast
- Quick testing and debugging
- 3-fold cross-validation
- Reduced sequence length
- Faster execution

### 3. High Performance
- Maximum performance configuration
- 5-fold cross-validation
- Longer sequences
- More epochs
- No layer freezing

## 📊 Configuration Presets

### Quick Test
```python
{
    "batch_size": 8,
    "num_epochs": 1,
    "num_folds": 2,
    "max_length": 64,
    "log_steps": 10,
    "eval_steps": 20,
    "save_steps": 50
}
```

### Production
```python
{
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 8,
    "num_folds": 5,
    "max_length": 256,
    "dropout": 0.3,
    "freeze_base": True,
    "mixed_precision": True
}
```

### Research
```python
{
    "batch_size": 12,
    "learning_rate": 1e-5,
    "num_epochs": 15,
    "num_folds": 10,
    "max_length": 512,
    "dropout": 0.2,
    "freeze_base": False,
    "mixed_precision": True
}
```

## 🔧 Custom Configuration

### Creating Custom Config

```python
from config import ExperimentConfig

# Create custom configuration
config = ExperimentConfig(
    batch_size=24,
    learning_rate=3e-5,
    num_epochs=12,
    num_folds=5,
    max_length=384,
    dropout=0.25,
    freeze_base=False,
    multi_task=True,
    author_name="my_custom_experiment"
)

# Save configuration
config.save_config("my_config.json")

# Load configuration
loaded_config = ExperimentConfig.load_config("my_config.json")
```

### Environment Variables

You can also control behavior via environment variables:

```bash
export USE_GPU=true
export USE_GOOGLE_DRIVE=true
export MLFLOW_EXPERIMENT_NAME="My_Experiment"
export AUTHOR_NAME="my_name"
```

## 📈 MLflow Integration

### Automatic Logging
The system automatically logs:
- System information (GPU, CPU, memory)
- Configuration parameters
- Dataset statistics
- Training metrics per fold
- Average metrics across folds
- Model artifacts
- Results summaries

### Accessing MLflow UI
```python
# Start MLflow UI
!mlflow ui

# Or access directly from Google Drive
# Path: /content/drive/MyDrive/bangla_bert_mlflow
```

### Custom Metrics
You can add custom metrics by modifying the `calculate_metrics` function in `train.py`.

## 🚨 Error Handling and Debugging

### Common Issues and Solutions

1. **CUDA Device-Side Assert Error**
   - Fixed in the modular version
   - Automatic label validation and correction
   - Detailed debugging output

2. **Google Drive Mounting Issues**
   - Automatic fallback to local storage
   - Multiple mounting attempts
   - Clear error messages

3. **Memory Issues**
   - Automatic batch size adjustment
   - Gradient accumulation support
   - Mixed precision training

4. **Dependency Issues**
   - Automatic dependency installation
   - Version pinning for compatibility
   - Fallback to older versions if needed

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Running Multiple Experiments

### Batch Experiments
```python
# Run multiple experiments with different configurations
configs = [
    get_default_config("fast"),
    get_default_config("standard"),
    get_default_config("high_performance")
]

for i, config in enumerate(configs):
    config.author_name = f"batch_experiment_{i}"
    runner = ExperimentRunner(config)
    runner.run_experiment()
```

### Hyperparameter Search
```python
# Grid search example
learning_rates = [1e-5, 2e-5, 3e-5]
batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for bs in batch_sizes:
        config = get_default_config("standard")
        config.learning_rate = lr
        config.batch_size = bs
        config.author_name = f"lr_{lr}_bs_{bs}"
        
        runner = ExperimentRunner(config)
        runner.run_experiment()
```

## 📝 Best Practices

### 1. Configuration Management
- Always use the configuration system
- Save configurations for reproducibility
- Use meaningful experiment names

### 2. Data Management
- Keep dataset in the `data/` directory
- Validate data before running experiments
- Use the provided preprocessing functions

### 3. Experiment Tracking
- Use MLflow for all experiments
- Log custom metrics and artifacts
- Document experiment purposes

### 4. Resource Management
- Monitor GPU memory usage
- Use appropriate batch sizes
- Enable mixed precision when possible

### 5. Error Handling
- Check logs for detailed error messages
- Use the provided debugging tools
- Report issues with system information

## 🎯 Next Steps

1. **Run First Experiment**: Start with the simple automated script
2. **Explore Configuration**: Try different presets and custom settings
3. **Analyze Results**: Use MLflow UI to compare experiments
4. **Optimize**: Run hyperparameter search for best performance
5. **Deploy**: Export the best model for production use

## 📞 Support

For issues and questions:
1. Check the logs in `experiment.log`
2. Review MLflow experiment results
3. Consult the troubleshooting section in `COLAB_GUIDE.md`
4. Enable debug mode for detailed output

---

This modular structure provides a robust, scalable, and maintainable foundation for your BanglaBERT hate speech detection experiments in Google Colab.
