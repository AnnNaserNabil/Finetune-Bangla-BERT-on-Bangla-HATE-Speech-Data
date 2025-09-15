#!/usr/bin/env python3
"""
Simple Automated Colab Script for BanglaBERT Training
This is a simplified version for easy execution in Google Colab
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function for automated Colab execution"""
    
    print("=" * 60)
    print("🚀 BanglaBERT Hate Speech Detection - Automated Colab Runner")
    print("=" * 60)
    
    # Step 1: Mount Google Drive
    print("\n📁 Step 1: Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully!")
    except Exception as e:
        print(f"⚠️  Google Drive mounting failed: {e}")
        print("   Continuing with local storage...")
    
    # Step 2: Install dependencies
    print("\n📦 Step 2: Installing dependencies...")
    try:
        import subprocess
        packages = [
            "transformers==4.44.2",
            "torch==2.4.0", 
            "scikit-learn==1.5.1",
            "pandas==2.2.2",
            "numpy==1.26.4",
            "tqdm==4.66.4",
            "mlflow==2.14.1",
            "emoji==2.12.1"
        ]
        
        for package in packages:
            print(f"   Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        
        print("✅ All dependencies installed successfully!")
        
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return
    
    # Step 3: Check GPU
    print("\n🚀 Step 3: Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name}")
        else:
            print("⚠️  No GPU available, using CPU")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
    
    # Step 4: Setup MLflow with Google Drive
    print("\n📊 Step 4: Setting up MLflow with Google Drive...")
    try:
        import mlflow
        
        # Create MLflow directory on Google Drive
        mlflow_drive_path = "/content/drive/MyDrive/bangla_bert_mlflow"
        os.makedirs(mlflow_drive_path, exist_ok=True)
        
        # Set MLflow tracking URI
        mlflow_uri = f"file://{mlflow_drive_path}"
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Create symlink for easier access
        if not os.path.exists("/content/mlruns"):
            os.symlink(mlflow_drive_path, "/content/mlruns")
        
        # Create experiment
        experiment_name = "BanglaBERT_HateSpeech_Detection"
        try:
            mlflow.create_experiment(experiment_name)
            print(f"✅ Created new MLflow experiment: {experiment_name}")
        except:
            print(f"✅ Using existing MLflow experiment: {experiment_name}")
        
        print(f"✅ MLflow logs will be saved to: {mlflow_drive_path}")
        
    except Exception as e:
        print(f"❌ Error setting up MLflow: {e}")
        return
    
    # Step 5: Upload dataset (if not already present)
    print("\n📄 Step 5: Checking dataset...")
    dataset_path = "data/5_BanEmoHate.csv"
    if not os.path.exists(dataset_path):
        print(f"⚠️  Dataset not found at {dataset_path}")
        print("   Please upload your dataset file to the 'data' directory")
        print("   Expected file: 5_BanEmoHate.csv")
        return
    else:
        print(f"✅ Dataset found at {dataset_path}")
    
    # Step 6: Run experiment
    print("\n🔬 Step 6: Starting experiment...")
    try:
        # Import and run the experiment
        from run_experiment import ExperimentRunner
        from config import get_default_config
        
        # Get configuration
        config = get_default_config("standard")
        config.author_name = "colab_automated"
        config.dataset_path = dataset_path
        
        print(f"📋 Configuration:")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Model: {config.model_name}")
        
        # Create and run experiment
        runner = ExperimentRunner(config)
        
        if runner.setup():
            print("\n🏋️  Starting training...")
            success = runner.run_experiment()
            
            if success:
                print("\n🎉 Experiment completed successfully!")
                print(f"📊 MLflow logs saved to: {mlflow_drive_path}")
                print(f"🔍 View results: !mlflow ui")
            else:
                print("\n❌ Experiment failed!")
        else:
            print("\n❌ Experiment setup failed!")
            
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
