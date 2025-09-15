import torch.nn as nn
from transformers import BertModel, BertConfig

class BertMultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.3):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
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
            if labels.shape[1] == 2:  # HateSpeech + Emotion
                # Binary cross entropy for HateSpeech
                hate_loss_fct = nn.BCEWithLogitsLoss()
                hate_loss = hate_loss_fct(logits[:, :1], labels[:, :1])
                
                # Cross entropy for Emotion (multi-class)
                emotion_loss_fct = nn.CrossEntropyLoss()
                emotion_loss = emotion_loss_fct(logits[:, 1:], labels[:, 1].long())
                
                # Combined loss with weights
                loss = 0.6 * hate_loss + 0.4 * emotion_loss
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
