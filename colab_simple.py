#!/usr/bin/env python3
"""
Simple Google Colab Training Script for Enhanced BanglaBERT
This provides a step-by-step approach with manual Google Drive mounting
"""

import os
import sys
import subprocess

def print_header():
    """Print header with instructions"""
    print("=" * 60)
    print("🚀 Enhanced BanglaBERT Hate Speech Detection - Google Colab")
    print("=" * 60)
    print("\n📋 STEP-BY-STEP INSTRUCTIONS:")
    print("\n1️⃣  FIRST: Mount Google Drive (run this in a separate cell):")
    print("   from google.colab import drive")
    print("   drive.mount('/content/drive')")
    print("   !mkdir -p /content/drive/MyDrive/bangla_bert_mlflow")
    print("   !ln -s /content/drive/MyDrive/bangla_bert_mlflow /content/mlruns")
    print("\n2️⃣  SECOND: Install dependencies (run this in a separate cell):")
    print("   !pip install -r requirements.txt")
    print("\n3️⃣  THIRD: Upload your dataset to data/5_BanEmoHate.csv")
    print("\n4️⃣  FOURTH: Run this training script")
    print("=" * 60)

def check_environment():
    """Check basic environment setup"""
    print("🔍 Checking environment...")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        is_colab = True
    except ImportError:
        print("⚠️  Not running in Google Colab")
        is_colab = False
    
    # Check if Google Drive is mounted
    if os.path.exists('/content/drive/MyDrive'):
        print("✅ Google Drive is mounted")
        drive_mounted = True
    else:
        print("❌ Google Drive is NOT mounted")
        print("   Please run: from google.colab import drive; drive.mount('/content/drive')")
        drive_mounted = False
    
    # Check if MLflow directory exists
    if os.path.exists('/content/mlruns'):
        print("✅ MLflow directory is ready")
    else:
        print("⚠️  MLflow directory not found")
        if drive_mounted:
            print("   Creating MLflow directory...")
            os.makedirs('/content/drive/MyDrive/bangla_bert_mlflow', exist_ok=True)
            os.symlink('/content/drive/MyDrive/bangla_bert_mlflow', '/content/mlruns')
            print("✅ MLflow directory created and linked")
        else:
            print("   Creating local MLflow directory...")
            os.makedirs('/content/mlruns', exist_ok=True)
            print("✅ Local MLflow directory created")
    
    return is_colab, drive_mounted

def check_dependencies():
    """Check if required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'pandas', 'numpy', 
        'scikit-learn', 'tqdm', 'mlflow', 'emoji'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Please run: !pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies are installed")
        return True

def check_gpu():
    """Check GPU availability"""
    print("🚀 Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected - training will be slower")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_dataset():
    """Check if dataset exists"""
    print("📁 Checking dataset...")
    
    dataset_path = "data/5_BanEmoHate.csv"
    
    if os.path.exists(dataset_path):
        print(f"✅ Dataset found: {dataset_path}")
        
        # Check dataset size
        import pandas as pd
        try:
            df = pd.read_csv(dataset_path)
            print(f"   Dataset shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Please upload your dataset file to data/5_BanEmoHate.csv")
        print("   You can use:")
        print("   from google.colab import files")
        print("   uploaded = files.upload()")
        print("   !mkdir -p data")
        print("   !mv 5_BanEmoHate.csv data/")
        return False

def run_training():
    """Run the training"""
    print("🎯 Starting training...")
    
    # Build command with optimized parameters
    cmd = [
        sys.executable, 'train.py',
        '--author_name', 'colab_user',
        '--dataset_path', 'data/5_BanEmoHate.csv',
        '--batch', '32',
        '--lr', '2e-5',
        '--epochs', '30',
        '--max_length', '256',
        '--dropout', '0.3',
        '--mlflow_experiment_name', 'Bangla_Hate_Speech_Enhanced_Colab'
    ]
    
    print(f"📋 Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        
        # Show results location
        if os.path.exists('/content/mlruns'):
            print(f"📊 Results saved to: /content/mlruns")
            if os.path.exists('/content/drive/MyDrive'):
                print(f"💾 Also saved to Google Drive: /content/drive/MyDrive/bangla_bert_mlflow")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with return code: {e.returncode}")
        return False

def main():
    """Main function"""
    print_header()
    
    # Check environment
    is_colab, drive_mounted = check_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Check dataset
    if not check_dataset():
        print("\n❌ Please upload the dataset and try again")
        return
    
    # Everything is ready, run training
    print("\n" + "=" * 60)
    print("🚀 ALL CHECKS PASSED - STARTING TRAINING")
    print("=" * 60)
    
    success = run_training()
    
    if success:
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("📊 Check your MLflow logs for detailed results")
        if drive_mounted:
            print("💾 Results are safely saved in Google Drive")
    else:
        print("\n❌ TRAINING FAILED")
        print("🔍 Please check the error messages above")

if __name__ == "__main__":
    main()
