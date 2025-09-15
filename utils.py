import os
import sys
import subprocess
import mlflow
import logging
from typing import Optional, Tuple
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_colab_environment() -> bool:
    """Check if running in Google Colab environment"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_gpu_available() -> Tuple[bool, str]:
    """Check if GPU is available and return device info"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return True, device_name
    return False, "CPU"

def mount_google_drive(drive_mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive in Colab environment"""
    if not is_colab_environment():
        logger.info("Not running in Colab, skipping Google Drive mounting")
        return False
    
    try:
        from google.colab import drive
        
        # Check if already mounted
        if os.path.exists(drive_mount_point):
            logger.info(f"Google Drive already mounted at {drive_mount_point}")
            return True
        
        logger.info("Mounting Google Drive...")
        drive.mount(drive_mount_point)
        
        # Verify mount was successful
        if os.path.exists(drive_mount_point):
            logger.info(f"✅ Google Drive successfully mounted at {drive_mount_point}")
            return True
        else:
            logger.error(f"❌ Failed to mount Google Drive at {drive_mount_point}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error mounting Google Drive: {str(e)}")
        return False

def setup_mlflow_with_drive(config) -> Tuple[bool, str]:
    """Setup MLflow with Google Drive backup"""
    
    mlflow_uri = config.mlflow_tracking_uri
    
    # If using Google Drive and it's mounted, use Drive path
    if config.use_google_drive and os.path.exists(config.drive_mount_point):
        try:
            # Create MLflow directory on Google Drive
            os.makedirs(config.mlflow_drive_path, exist_ok=True)
            
            # Set MLflow tracking URI to Google Drive
            mlflow_uri = f"file://{config.mlflow_drive_path}"
            
            # Create symlink for easier access
            local_mlflow_path = "/content/mlruns"
            if not os.path.exists(local_mlflow_path):
                os.symlink(config.mlflow_drive_path, local_mlflow_path)
                logger.info(f"Created symlink: {local_mlflow_path} -> {config.mlflow_drive_path}")
            
            logger.info(f"✅ MLflow logs will be saved to Google Drive: {config.mlflow_drive_path}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not setup Google Drive for MLflow: {str(e)}")
            logger.info("Falling back to local MLflow logging")
            mlflow_uri = "./mlruns"
    
    else:
        # Use local MLflow logging
        mlflow_uri = "./mlruns"
        os.makedirs(mlflow_uri, exist_ok=True)
        logger.info(f"📁 MLflow logs will be saved locally: {mlflow_uri}")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)
        if experiment is None:
            mlflow.create_experiment(config.mlflow_experiment_name)
            logger.info(f"📊 Created new MLflow experiment: {config.mlflow_experiment_name}")
        else:
            logger.info(f"📊 Using existing MLflow experiment: {config.mlflow_experiment_name}")
    except Exception as e:
        logger.error(f"❌ Error setting up MLflow experiment: {str(e)}")
        return False, str(e)
    
    return True, mlflow_uri

def install_dependencies() -> bool:
    """Install required dependencies for the project"""
    required_packages = [
        "transformers==4.44.2",
        "torch==2.4.0",
        "scikit-learn==1.5.1",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "tqdm==4.66.4",
        "mlflow==2.14.1",
        "emoji==2.12.1"
    ]
    
    logger.info("📦 Installing dependencies...")
    
    for package in required_packages:
        try:
            logger.info(f"Installing {package}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully installed {package}")
            else:
                logger.error(f"❌ Failed to install {package}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error installing {package}: {str(e)}")
            return False
    
    logger.info("✅ All dependencies installed successfully")
    return True

def check_environment_setup(config) -> Tuple[bool, dict]:
    """Check if environment is properly set up"""
    status = {
        "colab": False,
        "gpu": False,
        "gpu_name": "",
        "drive_mounted": False,
        "mlflow_ready": False,
        "dependencies_installed": False
    }
    
    # Check Colab environment
    status["colab"] = is_colab_environment()
    logger.info(f"🌐 Colab environment: {'Yes' if status['colab'] else 'No'}")
    
    # Check GPU availability
    status["gpu"], status["gpu_name"] = is_gpu_available()
    logger.info(f"🚀 GPU available: {'Yes' if status['gpu'] else 'No'}")
    if status["gpu"]:
        logger.info(f"💾 GPU: {status['gpu_name']}")
    
    # Check Google Drive mounting
    if config.use_google_drive and status["colab"]:
        status["drive_mounted"] = mount_google_drive(config.drive_mount_point)
    else:
        status["drive_mounted"] = True  # Not needed in non-Colab environment
    
    # Check MLflow setup
    mlflow_success, _ = setup_mlflow_with_drive(config)
    status["mlflow_ready"] = mlflow_success
    
    # Check dependencies
    try:
        import transformers, torch, sklearn, pandas, numpy, tqdm, mlflow, emoji
        status["dependencies_installed"] = True
        logger.info("✅ All dependencies are available")
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {str(e)}")
        status["dependencies_installed"] = False
    
    # Overall status
    all_ready = all([
        status["dependencies_installed"],
        status["mlflow_ready"],
        status["drive_mounted"]
    ])
    
    return all_ready, status

def setup_environment(config) -> bool:
    """Complete environment setup"""
    logger.info("🔧 Setting up environment for BanglaBERT training...")
    
    # Install dependencies if needed
    if not check_dependencies():
        logger.info("Installing missing dependencies...")
        if not install_dependencies():
            logger.error("❌ Failed to install dependencies")
            return False
    
    # Check environment setup
    all_ready, status = check_environment_setup(config)
    
    if not all_ready:
        logger.error("❌ Environment setup incomplete")
        logger.error(f"Status: {status}")
        return False
    
    logger.info("✅ Environment setup completed successfully")
    return True

def check_dependencies() -> bool:
    """Check if all required dependencies are installed"""
    required_modules = [
        "transformers", "torch", "sklearn", "pandas", 
        "numpy", "tqdm", "mlflow", "emoji"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.warning(f"Missing modules: {missing_modules}")
        return False
    
    return True

def get_device(config) -> torch.device:
    """Get the appropriate device for training"""
    if config.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("💻 Using CPU")
    
    return device

def log_system_info(config):
    """Log system information to MLflow"""
    import platform
    import psutil
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "gpu_available": torch.cuda.is_available(),
        "colab_environment": is_colab_environment(),
        "google_drive_mounted": os.path.exists(config.drive_mount_point) if config.use_google_drive else False
    }
    
    if torch.cuda.is_available():
        system_info["gpu_name"] = torch.cuda.get_device_name(0)
        system_info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    logger.info("🖥️  System Information:")
    for key, value in system_info.items():
        logger.info(f"   {key}: {value}")
    
    return system_info
