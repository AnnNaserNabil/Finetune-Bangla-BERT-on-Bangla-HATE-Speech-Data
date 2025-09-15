# =========================
# model.py  (fixed masks for MultiheadAttention)
# =========================
import torch
import torch.nn as nn
from transformers import BertModel


# -----------------------------
# Utility: freezing helpers
# -----------------------------
def freeze_base_layers(model: nn.Module):
    """Freeze encoder params for the simple model."""
    if hasattr(model, "bert"):
        for p in model.bert.parameters():
            p.requires_grad = False


def unfreeze_all_layers(model: nn.Module):
    """Unfreeze everything."""
    for p in model.parameters():
        p.requires_grad = True


def freeze_base_layers_enhanced(model: nn.Module):
    """Freeze encoder params for the enhanced model."""
    if hasattr(model, "bert"):
        for p in model.bert.parameters():
            p.requires_grad = False


# -----------------------------
# Key padding mask normalization
# -----------------------------
def to_key_padding_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """
    Convert a HuggingFace attention_mask (1=keep, 0=pad; dtype long/int/bool/float)
    into PyTorch MultiheadAttention key_padding_mask (True=PAD/IGNORE).
    Returns None if input is None.
    """
    if attention_mask is None:
        return None

    # Normalize to bool with True=PAD
    if attention_mask.dtype == torch.bool:
        # Assume HF semantics if bool: True=keep -> invert
        return ~attention_mask
    else:
        # Works for float/long/int: True where PAD (==0)
        return (attention_mask == 0)


# -----------------------------
# Simple classifier
# -----------------------------
class BertMultiLabelClassifier(nn.Module):
    """
    Output logits shape: (N, 4) where:
      logits[:, :1] -> Hate (binary, use BCEWithLogits)
      logits[:, 1:] -> Emotion classes (3 classes, use CrossEntropy)
    """
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 4,
        dropout: float = 0.1,
        multi_task: bool = True,
        config=None
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.multi_task = multi_task
        self.config = config

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # Use pooled output (CLS) for simplicity
        pooled = outputs.pooler_output  # (N, H)
        x = self.dropout(pooled)
        logits = self.classifier(x)  # (N, 4)

        out = {"logits": logits}

        if labels is not None:
            # labels: (N, 2) -> [:,0]=hate(0/1), [:,1]=emotion class id in {0,1,2}
            hate_targets = labels[:, 0].float()
            emo_targets = labels[:, 1].long()

            hate_logits = logits[:, 0]            # (N,)
            emo_logits = logits[:, 1:]            # (N, 3)

            hate_loss = nn.functional.binary_cross_entropy_with_logits(hate_logits, hate_targets)
            emo_loss = nn.functional.cross_entropy(emo_logits, emo_targets)

            loss = hate_loss + emo_loss
            out["loss"] = loss

        return out


# -----------------------------
# Enhanced cross-attention block
# -----------------------------
class CrossAttentionBlock(nn.Module):
    """
    One-head block that queries the sequence using a learned projection of CLS (or any query).
    Uses nn.MultiheadAttention with batch_first=True and a correct key_padding_mask.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, attn_dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,      # IMPORTANT: (N, S, E)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.ff_norm = nn.LayerNorm(hidden_size)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, attention_mask: torch.Tensor | None):
        """
        query:      (N, 1, E)
        key_value:  (N, S, E)
        attention_mask (HF style): (N, S) where 1=keep, 0=pad
        """
        key_padding_mask = to_key_padding_mask(attention_mask)  # (N, S), True=PAD

        attn_output, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask
        )  # (N, 1, E)

        # Residual + norm on the query path
        x = self.norm(query + attn_output)
        y = self.ff_norm(x + self.ff(x))
        return y  # (N, 1, E)


# -----------------------------
# Enhanced classifier with cross-attention heads
# -----------------------------
class EnhancedBertMultiLabelClassifier(nn.Module):
    """
    Uses BERT to encode tokens, then runs two small cross-attention heads that
    each attend from a CLS-query to the full sequence, and fuses them.
    """
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 4,
        dropout: float = 0.1,
        multi_task: bool = True,
        config=None
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        hidden = self.bert.config.hidden_size
        self.multi_task = multi_task
        self.config = config

        num_heads = getattr(config, "num_attention_heads", 8)
        attn_dropout = getattr(config, "attn_dropout", dropout)

        print(f"Using enhanced model architecture with {num_heads} attention heads")

        # Two parallel cross-attention blocks (can be extended)
        self.hate_cross_attention = CrossAttentionBlock(hidden, num_heads=num_heads, attn_dropout=attn_dropout)
        self.emotion_cross_attention = CrossAttentionBlock(hidden, num_heads=num_heads, attn_dropout=attn_dropout)

        # Projection from CLS to queries
        self.query_proj_hate = nn.Linear(hidden, hidden)
        self.query_proj_emotion = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(dropout)

        # Heads
        self.hate_head = nn.Linear(hidden, 1)   # binary logit
        self.emo_head = nn.Linear(hidden, 3)    # 3-way emotion

    def forward(self, input_ids, attention_mask=None, labels=None):
        enc = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq = enc.last_hidden_state            # (N, S, H)
        cls = enc.pooler_output                # (N, H)  (tanh(W h_cls))
        # Use CLS as the seed query, project, and add a sequence length dim=1
        q_hate = self.query_proj_hate(cls).unsqueeze(1)     # (N, 1, H)
        q_emo  = self.query_proj_emotion(cls).unsqueeze(1)  # (N, 1, H)

        # Cross-attend over the token sequence
        hate_ctx = self.hate_cross_attention(q_hate, seq, attention_mask).squeeze(1)  # (N, H)
        emo_ctx  = self.emotion_cross_attention(q_emo, seq, attention_mask).squeeze(1)  # (N, H)

        hate_ctx = self.dropout(hate_ctx)
        emo_ctx  = self.dropout(emo_ctx)

        hate_logit = self.hate_head(hate_ctx)     # (N, 1)
        emo_logits = self.emo_head(emo_ctx)       # (N, 3)

        logits = torch.cat([hate_logit, emo_logits], dim=1)  # (N, 4)
        out = {"logits": logits}

        if labels is not None:
            # labels: (N, 2) -> [:,0]=hate(0/1), [:,1]=emotion class id in {0,1,2}
            hate_targets = labels[:, 0].float()
            emo_targets = labels[:, 1].long()

            hate_loss = nn.functional.binary_cross_entropy_with_logits(hate_logit.squeeze(1), hate_targets)
            emo_loss = nn.functional.cross_entropy(emo_logits, emo_targets)

            # Optional weighting knobs (not required; defaults are 1.0)
            hate_w = float(getattr(self.config, "loss_weight_hate", 1.0)) if self.config is not None else 1.0
            emo_w  = float(getattr(self.config, "loss_weight_emotion", 1.0)) if self.config is not None else 1.0

            loss = hate_w * hate_loss + emo_w * emo_loss
            out["loss"] = loss

        return out
