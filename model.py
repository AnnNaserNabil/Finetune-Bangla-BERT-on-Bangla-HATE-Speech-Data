import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

# -----------------------------
# Mask helper (HF -> MHA)
# -----------------------------
def to_key_padding_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """
    Convert HuggingFace attention_mask (1=keep, 0=pad; any dtype)
    to PyTorch MultiheadAttention key_padding_mask (True = PAD/IGNORE).
    """
    if attention_mask is None:
        return None
    if attention_mask.dtype == torch.bool:
        # If someone already casted to bool keep-mask (True=keep), invert:
        # we need True=PAD for key_padding_mask.
        return ~attention_mask
    # Works for int/long/float: True where PAD
    return (attention_mask == 0)


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling for better sequence representation"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: (N, S, E)
        attention_mask: (N, S) with HF semantics (1=keep, 0=pad)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create query, key, value
        query = self.query(hidden_states)   # (N, S, E)
        key   = self.key(hidden_states)     # (N, S, E)
        value = self.value(hidden_states)   # (N, S, E)
        
        # Convert to key_padding_mask (True=PAD)
        key_padding_mask = to_key_padding_mask(attention_mask)  # (N, S) bool or None
            
        # Multi-head self-attention pooling over the sequence
        attn_output, _ = self.attention(
            query, key, value, key_padding_mask=key_padding_mask
        )  # (N, S, E)
        
        # Mean pool across sequence (on attn_output)
        pooled = self.output_proj(attn_output.mean(dim=1))  # (N, E)
        pooled = self.dropout(pooled)
        return pooled


class CrossAttentionModule(nn.Module):
    """Cross-attention module for multi-task learning"""
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, query, key_value, attention_mask=None):
        """
        query: (N, E) or (N, 1, E)
        key_value: (N, S, E)
        attention_mask: (N, S) with HF semantics (1=keep, 0=pad)
        returns: (N, 1, E)  (caller may squeeze to (N, E))
        """
        # Ensure query is 3D (N, 1, E)
        if query.dim() == 2:
            query = query.unsqueeze(1)
        elif query.dim() == 3:
            # keep as is (N, 1, E) or (N, Q, E); key_padding_mask masks keys, not queries
            pass
        else:
            raise ValueError(f"CrossAttentionModule query must be (N,E) or (N,1,E); got {tuple(query.shape)}")
        
        # Convert HF mask -> key_padding_mask
        key_padding_mask = to_key_padding_mask(attention_mask)  # (N, S) bool
        
        # Cross-attention (query attends over key/value sequence)
        attn_output, _ = self.attention(
            query, key_value, key_value, key_padding_mask=key_padding_mask
        )  # (N, 1, E)
        
        # Residual connection and normalization on the query path
        x = self.norm1(query + self.dropout(attn_output))  # (N, 1, E)
        
        # MLP/projection + second residual
        y = self.output_proj(x)                             # (N, 1, E)
        y = self.norm2(y + self.dropout(y))                # (N, 1, E)
        return y


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
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.gates.append(GatedResidualConnection(hidden_size))
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            prev_size = hidden_size
            
        self.output_layer = nn.Linear(prev_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        for layer, gate, norm in zip(self.layers, self.gates, self.layer_norms):
            x = layer(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = norm(x)
            # Only apply gated residual when dims match
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
        self.stochastic_depth = getattr(config, 'stochastic_depth', 0.1) if config is not None else 0.1
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with better initialization strategies"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def apply_stochastic_depth(self, x, survival_prob):
        """Apply stochastic depth for regularization (per-example)"""
        if (not self.training) or survival_prob >= 1.0:
            return x
        # Generate per-sample mask (broadcast along features)
        keep = (torch.rand(x.shape[0], 1, device=x.device) < survival_prob).float()
        return x * keep / survival_prob
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # BERT encoder
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = bert_outputs.last_hidden_state  # (N, S, E)
        
        # Attention pooling (uses proper key_padding_mask internally)
        pooled_output = self.attention_pooling(sequence_output, attention_mask)  # (N, E)
        
        # Stochastic depth
        pooled_output = self.apply_stochastic_depth(pooled_output, 1.0 - self.stochastic_depth)
        
        if self.multi_task:
            # Cross-attend from a single-token query derived from pooled_output
            hate_ctx   = self.hate_cross_attention(pooled_output, sequence_output, attention_mask)   # (N, 1, E)
            emotion_ctx= self.emotion_cross_attention(pooled_output, sequence_output, attention_mask) # (N, 1, E)
            
            # Squeeze back to (N, E)
            hate_ctx    = hate_ctx.squeeze(1)
            emotion_ctx = emotion_ctx.squeeze(1)
            
            # Task-specific classification
            hate_logits    = self.hate_classifier(hate_ctx)       # (N, 1)
            emotion_logits = self.emotion_classifier(emotion_ctx) # (N, 3)
            
            # Combine logits
            logits = torch.cat([hate_logits, emotion_logits], dim=1)  # (N, 4)
            
            loss = None
            if labels is not None:
                # Binary cross entropy for HateSpeech
                hate_loss_fct = nn.BCEWithLogitsLoss()
                hate_targets = labels[:, :1].float()
                hate_loss = hate_loss_fct(hate_logits, hate_targets)
                
                # Cross entropy for Emotion (multi-class)
                emotion_loss_fct = nn.CrossEntropyLoss()
                emotion_labels = labels[:, 1].long()
                emotion_labels = torch.clamp(emotion_labels, 0, 2)  # ensure in range [0,2]
                emotion_loss = emotion_loss_fct(emotion_logits, emotion_labels)
                
                hate_weight = float(getattr(self.config, 'hate_speech_loss_weight', 0.6)) if self.config is not None else 0.6
                emotion_weight = float(getattr(self.config, 'emotion_loss_weight', 0.4)) if self.config is not None else 0.4
                loss = hate_weight * hate_loss + emotion_weight * emotion_loss
                    
        else:
            # Single task classification
            logits = self.classifier(pooled_output)  # (N, num_labels)
            loss = None
            if labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
        
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (N, E)
        logits = self.classifier(cls_output)             # (N, num_labels)
        
        loss = None
        if labels is not None:
            if self.multi_task:
                # Binary cross entropy for HateSpeech
                hate_loss_fct = nn.BCEWithLogitsLoss()
                hate_targets = labels[:, :1].float()
                hate_loss = hate_loss_fct(logits[:, :1], hate_targets)
                
                # Cross entropy for Emotion (multi-class)
                emotion_loss_fct = nn.CrossEntropyLoss()
                emotion_labels = labels[:, 1].long()
                emotion_labels = torch.clamp(emotion_labels, 0, 2)
                emotion_logits = logits[:, 1:]  # (N, 3)
                emotion_loss = emotion_loss_fct(emotion_logits, emotion_labels)
                
                hate_w = float(getattr(self.config, 'hate_speech_loss_weight', 0.6)) if self.config is not None else 0.6
                emo_w  = float(getattr(self.config, 'emotion_loss_weight', 0.4)) if self.config is not None else 0.4
                loss = hate_w * hate_loss + emo_w * emotion_loss
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
        
        return {'loss': loss, 'logits': logits} if loss is not None else logits


# -----------------------------
# Freezing utilities
# -----------------------------
def freeze_base_layers(model):
    """Freeze BERT base layers with selective unfreezing"""
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
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.layer[-3:].parameters():
        param.requires_grad = True
    for param in model.bert.pooler.parameters():
        param.requires_grad = True
    
    # Keep task heads & attention trainable
    for m in [
        getattr(model, 'attention_pooling', None),
        getattr(model, 'hate_cross_attention', None),
        getattr(model, 'emotion_cross_attention', None),
        getattr(model, 'hate_classifier', None),
        getattr(model, 'emotion_classifier', None),
        getattr(model, 'classifier', None),
    ]:
        if m is not None:
            for p in m.parameters():
                p.requires_grad = True
