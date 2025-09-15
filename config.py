import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine-tune Bangla BERT for Hate Speech Detection')
    
    # Training parameters
    parser.add_argument('--batch', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--author_name', type=str, default='enhanced_model', help='Author name for experiment tracking')
    
    # Data parameters
    parser.add_argument('--dataset_path', type=str, default='data/5_BanEmoHate.csv', help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='sagorsarker/bangla-bert-base', help='Path to pre-trained model')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    
    # Model parameters
    parser.add_argument('--freeze_base', action='store_true', help='Freeze base BERT layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for classifier')
    
    # MLflow parameters
    parser.add_argument('--mlflow_experiment_name', type=str, default='Bangla_Hate_Speech_Enhanced', help='MLflow experiment name')
    
    return parser.parse_args()
