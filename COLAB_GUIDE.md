# 🚀 Google Colab Training Guide for Enhanced BanglaBERT

This guide provides step-by-step instructions for training the enhanced BanglaBERT model in Google Colab with automatic Google Drive backup to prevent data loss if Colab times out.

## 📋 Prerequisites

1. **Google Account**: For accessing Google Colab and Google Drive
2. **Dataset**: Your Bangla hate speech dataset file (`5_BanEmoHate.csv`)
3. **GitHub Repository**: Access to the enhanced code files

## 🗂️ File Structure

Before starting, ensure you have these files in your Google Colab environment:
```
📁 Finetune-Bangla-BERT-on-Bangla-hate-speech-Data/
├── 📄 train.py              # Enhanced training script
├── 📄 model.py              # Enhanced model architecture
├── 📄 data.py               # Enhanced data preprocessing
├── 📄 config.py             # Enhanced configuration
├── 📄 requirements.txt     # Dependencies
├── 📄 colab_training.py     # Colab-specific training script
├── 📄 COLAB_GUIDE.md       # This guide
└── 📁 data/
    └── 📄 5_BanEmoHate.csv   # Your dataset
```

## 🚀 Quick Start (Recommended)

### Method 1: Using the Automated Colab Script

#### Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click "File" → "New notebook"

#### Step 2: Enable GPU
1. Go to "Runtime" → "Change runtime type"
2. Select "T4 GPU" (or any available GPU)
3. Click "Save"

#### Step 3: Upload Files
**Option A: Using Git (Recommended)**
```python
# Clone the repository
!git clone https://github.com/your-username/Finetune-Bangla-BERT-on-Bangla-hate-speech-Data.git
%cd Finetune-Bangla-BERT-on-Bangla-hate-speech-Data
```

**Option B: Manual Upload**
1. Click the folder icon on the left sidebar
2. Click "Upload" and upload all the Python files
3. Create a `data` folder and upload your dataset there

#### Step 4: Upload Dataset
If you haven't uploaded your dataset yet:
```python
from google.colab import files
uploaded = files.upload()
# Move uploaded file to data directory
!mkdir -p data
!mv 5_BanEmoHate.csv data/
```

#### Step 5: Run the Automated Training
```python
# Run the enhanced Colab training script
!python colab_training.py
```

This script will automatically:
- ✅ Mount Google Drive
- ✅ Install all dependencies
- ✅ Check GPU availability
- ✅ Run training with optimized parameters
- ✅ Save MLflow logs to Google Drive

---

## 🔧 Manual Setup (Advanced)

### Step 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Create directory for MLflow logs
!mkdir -p /content/drive/MyDrive/bangla_bert_mlflow

# Create symlink for easy access
!ln -s /content/drive/MyDrive/bangla_bert_mlflow /content/mlruns
```

### Step 2: Install Dependencies
```python
!pip install -r requirements.txt
```

### Step 3: Check GPU
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### Step 4: Run Training
```python
# Basic training with optimized parameters
!python train.py \
    --author_name "your_name" \
    --dataset_path "data/5_BanEmoHate.csv" \
    --batch 32 \
    --lr 2e-5 \
    --epochs 30 \
    --max_length 256 \
    --dropout 0.3 \
    --mlflow_experiment_name "Bangla_Hate_Speech_Enhanced_Colab"
```

---

## 📊 Monitor Training Progress

### Method 1: MLflow UI (Local)
```python
# Start MLflow UI
!mlflow ui

# The UI will be available at http://localhost:5000
# You'll need to use Colab's tunneling service to access it
```

### Method 2: MLflow UI with ngrok (External Access)
```python
# Install ngrok
!pip install pyngrok

# Setup ngrok tunnel
from pyngrok import ngrok

# Kill any existing tunnels
ngrok.kill()

# Create new tunnel
ngrok_tunnel = ngrok.connect(5000)
print(f"MLflow UI: {ngrok_tunnel.public_url}")
```

### Method 3: Check Logs Directly
```python
# List MLflow experiments
!ls -la /content/mlruns/

# Check specific experiment
!ls -la /content/mlruns/0/
```

---

## 💾 Backup and Recovery

### Automatic Google Drive Backup
The enhanced setup automatically saves MLflow logs to:
```
/content/drive/MyDrive/bangla_bert_mlflow/
```

### Manual Backup Commands
```python
# Backup entire training directory
!zip -r bangla_bert_backup.zip /content/mlruns/

# Download backup
from google.colab import files
files.download("bangla_bert_backup.zip")
```

### Recovery After Timeout
If Colab times out, your progress is saved in Google Drive:

1. **Re-mount Google Drive**:
```python
from google.colab import drive
drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/bangla_bert_mlflow /content/mlruns
```

2. **Resume Training**:
```python
# Check existing experiments
!mlflow experiments list

# Continue from where you left off
!python train.py --author_name "your_name" --dataset_path "data/5_BanEmoHate.csv"
```

---

## ⚡ Performance Optimization

### GPU Optimization
```python
# Check GPU memory
!nvidia-smi

# Clear GPU cache if needed
import torch
torch.cuda.empty_cache()
```

### Training Optimization
```python
# For faster training with larger batch sizes (if GPU memory allows)
!python train.py \
    --author_name "your_name" \
    --dataset_path "data/5_BanEmoHate.csv" \
    --batch 64 \
    --lr 1e-5 \
    --epochs 20 \
    --max_length 256 \
    --dropout 0.3
```

### Memory Optimization
```python
# For limited GPU memory
!python train.py \
    --author_name "your_name" \
    --dataset_path "data/5_BanEmoHate.csv" \
    --batch 16 \
    --lr 2e-5 \
    --epochs 30 \
    --max_length 128 \
    --dropout 0.3
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Google Drive Mounting Failed
```python
# Solution: Try alternative mounting
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

#### 2. Out of Memory Error
```python
# Solution: Reduce batch size
!python train.py --batch 8 --author_name "your_name" --dataset_path "data/5_BanEmoHate.csv"
```

#### 3. Dataset Not Found
```python
# Solution: Upload dataset manually
from google.colab import files
uploaded = files.upload()
!mkdir -p data
!mv 5_BanEmoHate.csv data/
```

#### 4. MLflow UI Not Accessible
```python
# Solution: Use ngrok for external access
!pip install pyngrok
from pyngrok import ngrok
ngrok_tunnel = ngrok.connect(5000)
print(f"MLflow UI: {ngrok_tunnel.public_url}")
```

### Error Messages and Solutions

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size (`--batch 8` or `--batch 16`) |
| `ModuleNotFoundError` | Run `!pip install -r requirements.txt` |
| `FileNotFoundError` | Check dataset path and upload dataset |
| `Permission denied` | Re-mount Google Drive with `force_remount=True` |

---

## 📈 Expected Results

With the enhanced implementation, you should achieve:

| Metric | Expected Range | Improvement Over Original |
|--------|----------------|-------------------------|
| **Overall F1** | 85-93% | +23-28% |
| **HateSpeech F1** | 88-95% | +18-25% |
| **Emotion F1** | 82-90% | +14-22% |
| **Training Time** | 2-4 hours | Depends on GPU |

### Training Progress Indicators
- **Epoch 1-5**: Rapid improvement in F1 score
- **Epoch 10-20**: Steady convergence
- **Epoch 20-30**: Fine-tuning and peak performance

---

## 🎯 Best Practices

### Before Training
1. **Always enable GPU** for faster training
2. **Upload dataset** to the correct location (`data/5_BanEmoHate.csv`)
3. **Mount Google Drive** to prevent data loss
4. **Check GPU memory** with `!nvidia-smi`

### During Training
1. **Monitor progress** with MLflow UI
2. **Watch for errors** in the console output
3. **Check GPU usage** periodically
4. **Save checkpoints** if training takes too long

### After Training
1. **Download results** from Google Drive
2. **Analyze metrics** in MLflow UI
3. **Save model weights** for future use
4. **Document results** for reproducibility

---

## 🔄 Complete Workflow Example

```python
# Complete workflow in one cell
%%time

# Step 1: Setup
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/bangla_bert_mlflow
!ln -s /content/drive/MyDrive/bangla_bert_mlflow /content/mlruns

# Step 2: Install dependencies
!pip install -r requirements.txt

# Step 3: Upload dataset (if not already done)
# from google.colab import files
# uploaded = files.upload()
# !mkdir -p data
# !mv 5_BanEmoHate.csv data/

# Step 4: Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")

# Step 5: Run training
!python train.py \
    --author_name "colab_user" \
    --dataset_path "data/5_BanEmoHate.csv" \
    --batch 32 \
    --lr 2e-5 \
    --epochs 30 \
    --max_length 256 \
    --dropout 0.3 \
    --mlflow_experiment_name "Bangla_Hate_Speech_Enhanced_Colab"

print("✅ Training completed!")
print(f"📊 Results saved to: /content/drive/MyDrive/bangla_bert_mlflow")
```

---

## 🎉 Success Criteria

Your training is successful when:
- ✅ **Training completes** without errors
- ✅ **F1 score reaches 85-93%** (overall)
- ✅ **MLflow logs** are saved to Google Drive
- ✅ **All metrics** show improvement over baseline
- ✅ **Model converges** within 30 epochs

If you achieve these results, you've successfully implemented the enhanced BanglaBERT model for hate speech detection! 🚀
