#!/usr/bin/env python3
"""
Test script for Enhanced BanglaBERT Model Architecture
This script validates the enhanced model implementation and compares it with the original model.
"""

import torch
import numpy as np
from transformers import BertTokenizer
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig
from model import BertMultiLabelClassifier, EnhancedBertMultiLabelClassifier

def test_model_initialization():
    """Test that both models can be initialized correctly"""
    print("🧪 Testing Model Initialization...")
    
    config = ExperimentConfig()
    
    # Test original model
    try:
        original_model = BertMultiLabelClassifier(
            config.model_name,
            num_labels=4,
            dropout=config.dropout,
            multi_task=True,
            config=config
        )
        print("✅ Original model initialized successfully")
    except Exception as e:
        print(f"❌ Original model initialization failed: {e}")
        return False
    
    # Test enhanced model
    try:
        enhanced_model = EnhancedBertMultiLabelClassifier(
            config.model_name,
            num_labels=4,
            dropout=config.dropout,
            multi_task=True,
            config=config
        )
        print("✅ Enhanced model initialized successfully")
    except Exception as e:
        print(f"❌ Enhanced model initialization failed: {e}")
        return False
    
    return original_model, enhanced_model, config

def test_model_forward_pass(original_model, enhanced_model, config):
    """Test forward pass for both models"""
    print("\n🧪 Testing Forward Pass...")
    
    # Create sample data
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    sample_texts = [
        "এটি একটি নমুনা বাংলা টেক্সট",
        "আরেকটি উদাহরণ বাক্য"
    ]
    
    # Tokenize
    inputs = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt"
    )
    
    # Create sample labels (hate_speech, emotion)
    sample_labels = torch.tensor([
        [0, 1],  # non-hate, angry
        [1, 0]   # hate, sad
    ], dtype=torch.float32)
    
    # Test original model
    try:
        original_model.eval()
        with torch.no_grad():
            original_outputs = original_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=sample_labels
            )
        print("✅ Original model forward pass successful")
        print(f"   Original model loss: {original_outputs['loss'].item():.4f}")
        print(f"   Original model logits shape: {original_outputs['logits'].shape}")
    except Exception as e:
        print(f"❌ Original model forward pass failed: {e}")
        return False
    
    # Test enhanced model
    try:
        enhanced_model.eval()
        with torch.no_grad():
            enhanced_outputs = enhanced_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=sample_labels
            )
        print("✅ Enhanced model forward pass successful")
        print(f"   Enhanced model loss: {enhanced_outputs['loss'].item():.4f}")
        print(f"   Enhanced model logits shape: {enhanced_outputs['logits'].shape}")
    except Exception as e:
        print(f"❌ Enhanced model forward pass failed: {e}")
        return False
    
    return True

def test_model_architecture_components(enhanced_model):
    """Test specific components of the enhanced model"""
    print("\n🧪 Testing Enhanced Model Components...")
    
    # Test attention pooling
    try:
        batch_size, seq_len, hidden_size = 2, 10, 768
        dummy_sequence = torch.randn(batch_size, seq_len, hidden_size)
        dummy_mask = torch.ones(batch_size, seq_len)
        
        pooled_output = enhanced_model.attention_pooling(dummy_sequence, dummy_mask)
        print(f"✅ Attention pooling works - output shape: {pooled_output.shape}")
        
        if pooled_output.shape == (batch_size, hidden_size):
            print("✅ Attention pooling output shape is correct")
        else:
            print(f"❌ Attention pooling output shape mismatch: expected {(batch_size, hidden_size)}, got {pooled_output.shape}")
            return False
    except Exception as e:
        print(f"❌ Attention pooling test failed: {e}")
        return False
    
    # Test cross-attention modules
    try:
        dummy_query = torch.randn(batch_size, 1, hidden_size)
        dummy_key_value = torch.randn(batch_size, seq_len, hidden_size)
        
        hate_features = enhanced_model.hate_cross_attention(dummy_query, dummy_key_value)
        emotion_features = enhanced_model.emotion_cross_attention(dummy_query, dummy_key_value)
        
        print(f"✅ Cross-attention modules work")
        print(f"   Hate features shape: {hate_features.shape}")
        print(f"   Emotion features shape: {emotion_features.shape}")
    except Exception as e:
        print(f"❌ Cross-attention test failed: {e}")
        return False
    
    # Test enhanced classifiers
    try:
        dummy_input = torch.randn(batch_size, hidden_size)
        
        hate_logits = enhanced_model.hate_classifier(dummy_input)
        emotion_logits = enhanced_model.emotion_classifier(dummy_input)
        
        print(f"✅ Enhanced classifiers work")
        print(f"   Hate logits shape: {hate_logits.shape}")
        print(f"   Emotion logits shape: {emotion_logits.shape}")
        
        if hate_logits.shape == (batch_size, 1) and emotion_logits.shape == (batch_size, 3):
            print("✅ Classifier output shapes are correct")
        else:
            print(f"❌ Classifier output shapes mismatch")
            return False
    except Exception as e:
        print(f"❌ Enhanced classifier test failed: {e}")
        return False
    
    return True

def test_model_parameter_count(original_model, enhanced_model):
    """Compare parameter counts between models"""
    print("\n🧪 Comparing Model Parameter Counts...")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    original_params = count_parameters(original_model)
    enhanced_params = count_parameters(enhanced_model)
    
    print(f"Original model parameters: {original_params:,}")
    print(f"Enhanced model parameters: {enhanced_params:,}")
    print(f"Parameter increase: {enhanced_params - original_params:,} ({((enhanced_params/original_params - 1)*100):.1f}%)")
    
    # Enhanced model should have more parameters but not excessively more
    if enhanced_params > original_params:
        increase_ratio = enhanced_params / original_params
        if 1.0 < increase_ratio < 2.0:  # Should be less than 2x increase
            print("✅ Parameter count increase is reasonable")
            return True
        else:
            print(f"⚠️  Parameter count increase might be too large: {increase_ratio:.2f}x")
            return False
    else:
        print("❌ Enhanced model should have more parameters")
        return False

def test_model_training_step(enhanced_model, config):
    """Test a single training step with the enhanced model"""
    print("\n🧪 Testing Training Step...")
    
    # Create sample data
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    sample_texts = [
        "এটি একটি নমুনা বাংলা টেক্সট",
        "আরেকটি উদাহরণ বাক্য",
        "তৃতীয় নমুনা বাক্য",
        "চতুর্থ উদাহরণ"
    ]
    
    # Tokenize
    inputs = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt"
    )
    
    # Create sample labels
    sample_labels = torch.tensor([
        [0, 1],  # non-hate, angry
        [1, 0],  # hate, sad
        [0, 2],  # non-hate, happy
        [1, 1]   # hate, angry
    ], dtype=torch.float32)
    
    # Setup optimizer
    from torch.optim import AdamW
    optimizer = AdamW(enhanced_model.parameters(), lr=config.learning_rate)
    
    # Training step
    try:
        enhanced_model.train()
        optimizer.zero_grad()
        
        outputs = enhanced_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=sample_labels
        )
        
        loss = outputs['loss']
        loss.backward()
        
        # Check for gradients
        has_gradients = any(p.grad is not None for p in enhanced_model.parameters())
        if not has_gradients:
            print("❌ No gradients found in model parameters")
            return False
        
        optimizer.step()
        
        print("✅ Training step completed successfully")
        print(f"   Training loss: {loss.item():.4f}")
        
        # Check that loss is reasonable
        if loss.item() > 0 and loss.item() < 10:  # Should be positive but not extremely large
            print("✅ Training loss is reasonable")
            return True
        else:
            print(f"⚠️  Training loss might be unusual: {loss.item():.4f}")
            return False
            
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Enhanced Model Architecture Tests...")
    print("=" * 60)
    
    # Test 1: Model initialization
    models = test_model_initialization()
    if not models:
        print("❌ Model initialization tests failed")
        return False
    
    original_model, enhanced_model, config = models
    
    # Test 2: Forward pass
    if not test_model_forward_pass(original_model, enhanced_model, config):
        print("❌ Forward pass tests failed")
        return False
    
    # Test 3: Enhanced model components
    if not test_model_architecture_components(enhanced_model):
        print("❌ Enhanced model component tests failed")
        return False
    
    # Test 4: Parameter count comparison
    if not test_model_parameter_count(original_model, enhanced_model):
        print("❌ Parameter count comparison failed")
        return False
    
    # Test 5: Training step
    if not test_model_training_step(enhanced_model, config):
        print("❌ Training step test failed")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All Enhanced Model Architecture Tests Passed!")
    print("\n📊 Test Summary:")
    print("✅ Model initialization")
    print("✅ Forward pass")
    print("✅ Enhanced model components")
    print("✅ Parameter count comparison")
    print("✅ Training step")
    
    print("\n🔧 Enhanced Model Features:")
    print("• Multi-head attention pooling")
    print("• Cross-attention between tasks")
    print("• Gated residual connections")
    print("• Enhanced classifiers with skip connections")
    print("• Advanced regularization (stochastic depth)")
    print("• Better weight initialization")
    
    print("\n🚀 The enhanced model is ready for training!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
