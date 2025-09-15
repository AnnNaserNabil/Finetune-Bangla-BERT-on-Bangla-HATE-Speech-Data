import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling for better sequence representation"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create query, key, value
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Apply attention mask
        if attention_mask is not None:
            # Convert attention mask to key padding mask format
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
            
        # Multi-head attention
        attn_output, _ = self.attention(query, key, value, key_padding_mask=key_padding_mask)
        
        # Apply output projection and dropout
        pooled = self.output_proj(attn_output.mean(dim=1))  # Mean pooling over sequence
        pooled = self.dropout(pooled)
        
        return pooled

class CrossAttentionModule(nn.Module):
    """Cross-attention module for multi-task learning"""
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, query, key_value, attention_mask=None):
        # Cross-attention
        attn_output, _ = self.attention(query, key_value, key_value, key_padding_mask=attention_mask)
        
        # Residual connection and normalization
        query = self.norm1(query + self.dropout(attn_output))
        
        # Output projection
        output = self.output_proj(query)
        output = self.norm2(output + self.dropout(output))
        
        return output

class GatedResidualConnection(nn.Module):
    """Gated residual connection for better gradient flow"""
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
    def forward(self, x, residual):
        gate = self.gate(x)
        return gate * x + (1 - gate) * residual

class EnhancedClassifier(nn.Module):
    """Enhanced classifier with skip connections and advanced regularization"""
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Build layers with skip connections
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.gates.append(GatedResidualConnection(hidden_size))
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            prev_size = hidden_size
            
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        
        for layer, gate, norm in zip(self.layers, self.gates, self.layer_norms):
            x = layer(x)
            x = F.gelu(x)  # GELU activation for better performance
            x = self.dropout(x)
            x = norm(x)
            
            # Apply gated residual connection if dimensions match
            if x.shape == residual.shape:
                x = gate(x, residual)
            
            residual = x
            
        output = self.output_layer(x)
        return output

class EnhancedBertMultiLabelClassifier(nn.Module):
    """Enhanced BanglaBERT model with advanced architecture improvements"""
    def __init__(self, model_name, num_labels, dropout=0.3, multi_task=False, config=None):
        super(EnhancedBertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.multi_task = multi_task
        self.config = config
        self.hidden_size = self.bert.config.hidden_size
        
        # Multi-head attention pooling instead of [CLS] token
        self.attention_pooling = MultiHeadAttentionPooling(
            self.hidden_size, num_heads=8, dropout=dropout
        )
        
        # Cross-attention modules for multi-task learning
        self.hate_cross_attention = CrossAttentionModule(
            self.hidden_size, num_heads=4, dropout=dropout
        )
        self.emotion_cross_attention = CrossAttentionModule(
            self.hidden_size, num_heads=4, dropout=dropout
        )
        
        # Task-specific enhanced classifiers
        if multi_task:
            # Hate speech classifier (binary)
            self.hate_classifier = EnhancedClassifier(
                self.hidden_size, 
                hidden_sizes=[512, 256, 128], 
                output_size=1,
                dropout=dropout
            )
            
            # Emotion classifier (3-class)
            self.emotion_classifier = EnhancedClassifier(
                self.hidden_size,
                hidden_sizes=[512, 256, 128],
                output_size=3,
                dropout=dropout
            )
        else:
            # Single classifier for all labels
            self.classifier = EnhancedClassifier(
                self.hidden_size,
                hidden_sizes=[512, 256, 128],
                output_size=num_labels,
                dropout=dropout
            )
        
        # Advanced regularization components
        self.layer_dropout = nn.Dropout(dropout)
        self.stochastic_depth = config.stochastic_depth if hasattr(config, 'stochastic_depth') else 0.1
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with better initialization strategies"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def apply_stochastic_depth(self, x, survival_prob):
        """Apply stochastic depth for regularization"""
        if not self.training or survival_prob == 1.0:
            return x
            
        # Generate random mask
        batch_size = x.shape[0]
        random_tensor = torch.rand(batch_size, 1, device=x.device) < survival_prob
        
        # Scale output to maintain expected value
        return x * random_tensor.float() / survival_prob
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get BERT outputs
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        # Apply attention pooling instead of [CLS] token
        pooled_output = self.attention_pooling(sequence_output, attention_mask)
        
        # Apply stochastic depth for regularization
        pooled_output = self.apply_stochastic_depth(pooled_output, 1.0 - self.stochastic_depth)
        
        if self.multi_task:
            # Multi-task learning with cross-attention
            hate_features = self.hate_cross_attention(
                pooled_output, sequence_output, attention_mask
            )
            emotion_features = self.emotion_cross_attention(
                pooled_output, sequence_output, attention_mask
            )
            
            # Task-specific classification
            hate_logits = self.hate_classifier(hate_features)
            emotion_logits = self.emotion_classifier(emotion_features)
            
            # Combine logits
            logits = torch.cat([hate_logits, emotion_logits], dim=1)
            
            # Compute loss
            loss = None
            if labels is not None:
                # Binary cross entropy for HateSpeech
                hate_loss_fct = nn.BCEWithLogitsLoss()
                hate_loss = hate_loss_fct(hate_logits, labels[:, :1])
                
                # Cross entropy for Emotion (multi-class)
                emotion_loss_fct = nn.CrossEntropyLoss()
                
                # Extract and validate emotion labels
                emotion_labels = labels[:, 1].long()
                emotion_labels = torch.clamp(emotion_labels, 0, 2)
                
                # Ensure labels are within valid range
                if torch.any(emotion_labels < 0) or torch.any(emotion_labels >= 3):
                    emotion_labels = torch.where(emotion_labels < 0, torch.zeros_like(emotion_labels), emotion_labels)
                    emotion_labels = torch.where(emotion_labels >= 3, torch.full_like(emotion_labels, 2), emotion_labels)
                
                # Compute emotion loss only if labels are valid
                if torch.any(emotion_labels < 0) or torch.any(emotion_labels >= 3):
                    loss = hate_loss
                else:
                    emotion_loss = emotion_loss_fct(emotion_logits, emotion_labels)
                    # Combined loss with configurable weights
                    hate_weight = getattr(self.config, 'hate_speech_loss_weight', 0.6)
                    emotion_weight = getattr(self.config, 'emotion_loss_weight', 0.4)
                    loss = hate_weight * hate_loss + emotion_weight * emotion_loss
                    
        else:
            # Single task classification
            logits = self.classifier(pooled_output)
            
            # Compute loss
            loss = None
            if labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits} if loss is not None else logits

class BertMultiLabelClassifier(nn.Module):
    """Original model for backward compatibility"""
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

def freeze_base_layers_enhanced(model):
    """Enhanced freezing strategy for the new model architecture"""
    # Freeze BERT base layers
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # Unfreeze the last 3 layers for better fine-tuning
    for param in model.bert.encoder.layer[-3:].parameters():
        param.requires_grad = True
    
    # Unfreeze pooler layer
    for param in model.bert.pooler.parameters():
        param.requires_grad = True
    
    # Always keep attention and classifier layers trainable
    for param in model.attention_pooling.parameters():
        param.requires_grad = True
    
    if hasattr(model, 'hate_cross_attention'):
        for param in model.hate_cross_attention.parameters():
            param.requires_grad = True
    
    if hasattr(model, 'emotion_cross_attention'):
        for param in model.emotion_cross_attention.parameters():
            param.requires_grad = True
    
    if hasattr(model, 'hate_classifier'):
        for param in model.hate_classifier.parameters():
            param.requires_grad = True
    
    if hasattr(model, 'emotion_classifier'):
        for param in model.emotion_classifier.parameters():
            param.requires_grad = True
    
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
