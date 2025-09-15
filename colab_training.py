#!/usr/bin/env python3
"""
Google Colab Training Script for Enhanced BanglaBERT Hate Speech Detection
This script includes Google Drive mounting to save MLflow logs and prevent data loss
"""

import os
import sys
import subprocess
from pathlib import Path

def check_colab_environment():
    """Check if running in Google Colab"""
    # Method 1: Try importing google.colab
    try:
        import google.colab
        return True
    except ImportError:
        pass
    
    # Method 2: Check environment variables
    import os
    colab_env_vars = ['COLAB_GPU', 'DATALAB_SETTINGS_OAUTH', 'GCS_READ_CACHE_BLOCK_SIZE_MB']
    if any(var in os.environ for var in colab_env_vars):
        return True
    
    # Method 3: Check file system structure
    colab_paths = ['/content', '/content/sample_data', '/usr/local/bin/colab']
    if all(os.path.exists(path) for path in colab_paths[:2]):  # At least first two paths
        return True
    
    # Method 4: Check if running in typical Colab Python path
    import sys
    if 'google.colab' in sys.modules:
        return True
    
    return False

def mount_google_drive():
    """Mount Google Drive in Colab environment"""
    if not check_colab_environment():
        print("Not running in Google Colab. Skipping Google Drive mounting.")
        return None
    
    try:
        # Try importing and mounting with better error handling
        try:
            from google.colab import drive
            print(" Mounting Google Drive...")
            
            # Mount with force_remount to handle existing mounts
            drive.mount('/content/drive', force_remount=True)
            
            # Verify mount was successful
            if os.path.exists('/content/drive/MyDrive'):
                print(" Google Drive mounted successfully!")
                
                # Create MLflow directory in Google Drive
                drive_mlflow_path = '/content/drive/MyDrive/bangla_bert_mlflow'
                os.makedirs(drive_mlflow_path, exist_ok=True)
                
                # Create symlink for easier access
                if os.path.exists('/content/mlruns'):
                    if os.path.islink('/content/mlruns'):
                        os.remove('/content/mlruns')
                    else:
                        import shutil
                        shutil.rmtree('/content/mlruns')
                
                os.symlink(drive_mlflow_path, '/content/mlruns')
                
                print(f" MLflow logs will be saved to: {drive_mlflow_path}")
                return drive_mlflow_path
            else:
                print(" Google Drive mount verification failed")
                return None
                
        except Exception as import_error:
            print(f" Error importing google.colab: {import_error}")
            return None
            
    except Exception as e:
        print(f" Error mounting Google Drive: {e}")
        print("  Continuing without Google Drive. MLflow logs will be saved locally.")
        
        # Create local mlruns directory as fallback
        local_mlflow_path = '/content/mlruns'
        os.makedirs(local_mlflow_path, exist_ok=True)
        print(f" MLflow logs will be saved locally to: {local_mlflow_path}")
        
        return None

def install_dependencies():
    """Install required dependencies"""
    print(" Installing dependencies...")
    
    # Install required packages
    packages = [
        'transformers==4.44.2',
        'torch==2.4.0',
        'scikit-learn==1.5.1',
        'pandas==2.2.2',
        'numpy==1.26.4',
        'tqdm==4.66.4',
        'mlflow==2.14.1',
        'emoji==2.12.1'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f" Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f" Failed to install {package}: {e}")
    
    print(" All dependencies installed successfully!")

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f" GPU detected: {gpu_name}")
            return True
        else:
            print("  No GPU detected. Training will be slower.")
            return False
    except ImportError:
        print(" PyTorch not installed. Please install dependencies first.")
        return False

def setup_environment():
    """Setup the training environment"""
    print(" Setting up environment...")
    
    # Check if we're in Colab
    is_colab = check_colab_environment()
    print(f" Running in Google Colab: {is_colab}")
    
    # Mount Google Drive if in Colab
    drive_path = mount_google_drive()
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU
    check_gpu()
    
    return is_colab, drive_path

def download_dataset():
    """Download or prepare the dataset"""
    dataset_path = "data/5_BanEmoHate.csv"
    
    if not os.path.exists(dataset_path):
        print(" Dataset not found. Please upload your dataset file.")
        print(f"Expected path: {dataset_path}")
        print("You can upload files using the Colab file browser or by running:")
        print("from google.colab import files")
        print("uploaded = files.upload()")
        return False
    
    print(f" Dataset found at: {dataset_path}")
    return True

def run_training(author_name="colab_user", **kwargs):
    """Run the enhanced training"""
    print(" Starting enhanced BanglaBERT training...")
    
    # Build command
    cmd = [sys.executable, 'train.py']
    
    # Add arguments
    cmd.extend(['--author_name', author_name])
    cmd.extend(['--dataset_path', 'data/5_BanEmoHate.csv'])
    
    # Default optimized parameters
    default_params = {
        'batch': 32,
        'lr': 2e-5,
        'epochs': 30,
        'max_length': 256,
        'dropout': 0.3,
        'mlflow_experiment_name': 'Bangla_Hate_Speech_Enhanced_Colab'
    }
    
    # Override with user-provided parameters
    default_params.update(kwargs)
    
    # Add parameters to command
    for key, value in default_params.items():
        cmd.extend([f'--{key}', str(value)])
    
    print(f" Training command: {' '.join(cmd)}")
    
    try:
        # Run training
        subprocess.run(cmd, check=True)
        print(" Training completed successfully!")
        
        # Show MLflow logs location
        if os.path.exists('/content/mlruns'):
            print(f" MLflow logs saved to: /content/mlruns")
            if check_colab_environment():
                print(f" Also saved to Google Drive: /content/drive/MyDrive/bangla_bert_mlflow")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f" Training failed: {e}")
        return False

def show_mlflow_ui():
    """Show MLflow UI instructions"""
    print("\n MLflow UI Instructions:")
    print("1. To view MLflow experiments, run:")
    print("   !mlflow ui")
    print("2. Click on the generated link (usually http://localhost:5000)")
    print("3. Or use ngrok for external access:")
    print("   !pip install pyngrok")
    print("   from pyngrok import ngrok")
    print("   ngrok.kill()")
    print("   ngrok_tunnel = ngrok.connect(5000)")
    print("   print('MLflow UI:', ngrok_tunnel.public_url)")

def main():
    """Main function for Colab training"""
    print("=" * 60)
    print(" Enhanced BanglaBERT Hate Speech Detection - Google Colab")
    print("=" * 60)
    
    # Setup environment
    is_colab, drive_path = setup_environment()
    
    # Check dataset
    if not download_dataset():
        print(" Please upload the dataset and try again.")
        return
    
    # Show MLflow instructions
    show_mlflow_ui()
    
    # Run training with default parameters
    print("\n Starting training with optimized parameters...")
    success = run_training(
        author_name="colab_user",
        batch=32,
        lr=2e-5,
        epochs=30,
        max_length=256,
        dropout=0.3
    )
    
    if success:
        print("\n Training completed successfully!")
        print(" Check your MLflow logs for detailed results.")
        if is_colab and drive_path:
            print(f" All logs saved to Google Drive: {drive_path}")
    else:
        print("\n Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
