import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BertMultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.3, multi_task=False, config=None):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.multi_task = multi_task
        self.config = config
        
        # Enhanced classifier with multiple layers and better regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Get logits from classifier
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            # Use different loss functions for different tasks
            if self.multi_task:
                # Binary cross entropy for HateSpeech
                hate_loss_fct = nn.BCEWithLogitsLoss()
                hate_loss = hate_loss_fct(logits[:, :1], labels[:, :1])
                
                # Cross entropy for Emotion (multi-class)
                emotion_loss_fct = nn.CrossEntropyLoss()
                
                # Extract and validate emotion labels
                emotion_labels = labels[:, 1].long()
                
                # Ensure labels are within valid range [0, 2] for 3 emotion classes
                emotion_labels = torch.clamp(emotion_labels, 0, 2)
                
                # Final validation after clamping
                if torch.any(emotion_labels < 0) or torch.any(emotion_labels >= 3):
                    # Emergency fix - set all invalid labels to valid values
                    emotion_labels = torch.where(emotion_labels < 0, torch.zeros_like(emotion_labels), emotion_labels)
                    emotion_labels = torch.where(emotion_labels >= 3, torch.full_like(emotion_labels, 2), emotion_labels)
                
                # Ensure logits and labels have compatible shapes
                emotion_logits = logits[:, 1:]  # Should be [batch_size, 3]
                
                # One final safety check before computing loss
                if torch.any(emotion_labels < 0) or torch.any(emotion_labels >= 3):
                    # Use only hate speech loss if emotion labels are still invalid
                    loss = hate_loss
                else:
                    # Compute emotion loss only if labels are valid
                    emotion_loss = emotion_loss_fct(emotion_logits, emotion_labels)
                    # Combined loss with weights
                    loss = self.config.hate_speech_loss_weight * hate_loss + self.config.emotion_loss_weight * emotion_loss
            else:
                # Fallback to BCE for multi-label
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits} if loss is not None else logits

def freeze_base_layers(model):
    """Freeze BERT base layers with selective unfreezing"""
    # Freeze all BERT layers initially
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # Unfreeze the last 2 layers for better fine-tuning
    for param in model.bert.encoder.layer[-2:].parameters():
        param.requires_grad = True
    
    # Always unfreeze the pooler layer
    for param in model.bert.pooler.parameters():
        param.requires_grad = True

def unfreeze_all_layers(model):
    """Unfreeze all layers for full fine-tuning"""
    for param in model.parameters():
        param.requires_grad = True
