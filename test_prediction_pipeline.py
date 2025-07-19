#!/usr/bin/env python3
"""
🔮 COMPLETE TEST PREDICTION PIPELINE
✅ Same preprocessing as train/val data
✅ Load best trained model
✅ Predict on test data with no labels
✅ Save predictions to CSV
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import json
import re
from transformers import (
    VisualBertModel, VisualBertConfig, VisualBertTokenizer,
    ViTImageProcessor, ViTModel
)
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# ========================================
# 📁 CONFIGURATION CLASS (SAME AS TRAINING)
# ========================================
class Config:
    # Dataset
    DATASET_NAME = "limjiayi/memotion_dataset_3"
    MAX_LENGTH = 128
    IMAGE_SIZE = 224
    
    # Paths
    OUTPUT_DIR = "./memotion_results"
    CACHE_DIR = "./cache"
    VIT_CACHE_FILE = "./vit_features_cache.pt"
    MODEL_PATH = "./memotion_results/best_model"
    PREDICTIONS_CSV = "./test_predictions.csv"
    
    # Model
    VISUAL_BERT_MODEL = "uclanlp/visualbert-vqa-coco-pre"
    VIT_MODEL = "google/vit-base-patch16-224-in21k"

config = Config()

# ========================================
# 🧹 ENHANCED OCR TEXT CLEANING (SAME AS TRAINING)
# ========================================
def enhanced_ocr_cleaning(text):
    """Advanced OCR text cleaning with regex patterns"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove URLs, handles, hashtags, and special characters
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip().lower()

# ========================================
# 🚀 FEATURE CACHING FOR TEST DATA (SAME PREPROCESSING)
# ========================================
def cache_test_vit_features(dataset, processor, model, cache_file, device):
    """Pre-compute and cache ViT features for test data"""
    print("🔄 Caching ViT features for test data...")
    
    if os.path.exists(cache_file):
        print("✅ Loading cached test ViT features...")
        return torch.load(cache_file)
    
    features_cache = {}
    model.eval()
    
    for idx, sample in enumerate(dataset):
        if idx % 1000 == 0:
            print(f"Cached {idx}/{len(dataset)} test features...")
        
        try:
            # Load and process image (SAME AS TRAINING)
            image_path = sample.get('image_path', sample.get('image', ''))
            if not image_path or not os.path.exists(image_path):
                features_cache[idx] = torch.zeros(768)  # Default ViT feature size
                continue
                
            image = Image.open(image_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token embedding
                features_cache[idx] = outputs.last_hidden_state[:, 0, :].cpu().squeeze()
                
        except Exception as e:
            print(f"⚠️ Error processing test image {idx}: {e}")
            features_cache[idx] = torch.zeros(768)
    
    # Save cache
    torch.save(features_cache, cache_file)
    print(f"💾 Cached {len(features_cache)} test ViT features to {cache_file}")
    
    return features_cache

# ========================================
# 📊 TEST DATASET (SAME PREPROCESSING, NO LABELS)
# ========================================
class TestMemotionDataset(Dataset):
    def __init__(self, dataset, tokenizer, vit_features_cache, max_length=128, device='cuda'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.vit_features = vit_features_cache
        self.max_length = max_length
        self.device = device
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Process text (SAME AS TRAINING)
        text = enhanced_ocr_cleaning(str(sample.get('text', '')))
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get cached ViT features (SAME AS TRAINING)
        image_features = self.vit_features.get(idx, torch.zeros(768))
        if not isinstance(image_features, torch.Tensor):
            image_features = torch.tensor(image_features)
        image_features = image_features.to(self.device)
        
        return {
            'input_ids': encoding['input_ids'].squeeze().to(self.device),
            'attention_mask': encoding['attention_mask'].squeeze().to(self.device),
            'visual_embeds': image_features.unsqueeze(0),
            'visual_attention_mask': torch.ones(1, dtype=torch.long).to(self.device),
            'sample_id': idx,  # Keep track of sample index
            'original_text': str(sample.get('text', '')),  # Original text for CSV
            'image_path': str(sample.get('image_path', sample.get('image', '')))  # Image path for CSV
        }

# ========================================
# 🧠 ENHANCED VISUALBERT CLASSIFIER (SAME AS TRAINING)
# ========================================
class EnhancedVisualBertClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        
        # Load VisualBERT
        self.visual_bert = VisualBertModel.from_pretrained(config.VISUAL_BERT_MODEL)
        
        # Enhanced classifier head (SAME AS TRAINING)
        hidden_size = self.visual_bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, visual_embeds, visual_attention_mask, labels=None):
        outputs = self.visual_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        return {'logits': logits}

# ========================================
# 🔄 LOAD TRAINED MODEL
# ========================================
def load_trained_model(model_path, device):
    """Load the saved trained model"""
    print(f"🔄 Loading trained model from {model_path}")
    
    # Initialize model with same architecture
    model = EnhancedVisualBertClassifier(num_labels=2)
    
    # Load saved weights
    model_weights_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print("✅ Model weights loaded successfully!")
    else:
        print(f"❌ Model weights not found at {model_weights_path}")
        print("Available files:", os.listdir(model_path) if os.path.exists(model_path) else "Directory not found")
        return None
    
    model.to(device)
    model.eval()
    
    return model

# ========================================
# 🔮 COMPLETE TEST PREDICTION PIPELINE
# ========================================
def predict_test_data():
    """Complete pipeline to predict on test data and save to CSV"""
    print("🔮 Starting Test Data Prediction Pipeline")
    print("✅ Using SAME preprocessing as training data")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # 1. Load test dataset
    print("📊 Loading test dataset...")
    try:
        dataset = load_dataset(config.DATASET_NAME)
        test_dataset = dataset['test'] if 'test' in dataset else dataset['validation']
        print(f"📊 Test dataset size: {len(test_dataset)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    # 2. Initialize ViT for feature caching (SAME AS TRAINING)
    print("🔧 Initializing ViT for feature caching...")
    vit_processor = ViTImageProcessor.from_pretrained(config.VIT_MODEL)
    vit_model = ViTModel.from_pretrained(config.VIT_MODEL).to(device)
    
    # 3. Cache ViT features for test data (SAME PREPROCESSING)
    test_cache_file = f"{config.VIT_CACHE_FILE}_test"
    test_cache = cache_test_vit_features(test_dataset, vit_processor, vit_model, test_cache_file, device)
    
    # 4. Initialize tokenizer (SAME AS TRAINING)
    print("🔧 Loading VisualBERT tokenizer...")
    tokenizer = VisualBertTokenizer.from_pretrained(config.VISUAL_BERT_MODEL)
    
    # 5. Create test dataset (SAME PREPROCESSING)
    print("📚 Creating test dataset with same preprocessing...")
    test_torch_dataset = TestMemotionDataset(test_dataset, tokenizer, test_cache, device=device)
    
    # 6. Load trained model
    print("🧠 Loading trained model...")
    model = load_trained_model(config.MODEL_PATH, device)
    if model is None:
        return None
    
    # 7. Create DataLoader for batch processing
    test_dataloader = DataLoader(test_torch_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # 8. Make predictions
    print("🔮 Making predictions on test data...")
    predictions = []
    sample_info = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx * 16}/{len(test_dataset)} samples...")
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                visual_embeds=batch['visual_embeds'],
                visual_attention_mask=batch['visual_attention_mask']
            )
            
            # Get predictions
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(probabilities, dim=-1)
            confidence_scores = torch.max(probabilities, dim=-1)[0]
            
            # Store results
            for i in range(len(predicted_labels)):
                sample_idx = batch['sample_id'][i].item()
                
                predictions.append({
                    'sample_id': sample_idx,
                    'original_text': batch['original_text'][i],
                    'cleaned_text': enhanced_ocr_cleaning(batch['original_text'][i]),
                    'image_path': batch['image_path'][i],
                    'predicted_label': predicted_labels[i].item(),
                    'prediction': 'Hate Speech' if predicted_labels[i].item() == 1 else 'Not Hate Speech',
                    'confidence': confidence_scores[i].item(),
                    'prob_not_hate': probabilities[i][0].item(),
                    'prob_hate': probabilities[i][1].item()
                })
    
    # 9. Create DataFrame and save to CSV
    print("💾 Saving predictions to CSV...")
    df_predictions = pd.DataFrame(predictions)
    
    # Add summary statistics
    hate_count = (df_predictions['predicted_label'] == 1).sum()
    not_hate_count = (df_predictions['predicted_label'] == 0).sum()
    avg_confidence = df_predictions['confidence'].mean()
    
    print(f"\n📊 PREDICTION SUMMARY:")
    print(f"   Total samples: {len(df_predictions)}")
    print(f"   Predicted Hate Speech: {hate_count} ({hate_count/len(df_predictions)*100:.2f}%)")
    print(f"   Predicted Not Hate Speech: {not_hate_count} ({not_hate_count/len(df_predictions)*100:.2f}%)")
    print(f"   Average Confidence: {avg_confidence:.4f}")
    
    # Save to CSV
    df_predictions.to_csv(config.PREDICTIONS_CSV, index=False)
    print(f"✅ Predictions saved to: {config.PREDICTIONS_CSV}")
    
    # Save summary file
    summary_file = config.PREDICTIONS_CSV.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("🔮 TEST PREDICTIONS SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total samples: {len(df_predictions)}\n")
        f.write(f"Predicted Hate Speech: {hate_count} ({hate_count/len(df_predictions)*100:.2f}%)\n")
        f.write(f"Predicted Not Hate Speech: {not_hate_count} ({not_hate_count/len(df_predictions)*100:.2f}%)\n")
        f.write(f"Average Confidence: {avg_confidence:.4f}\n")
        f.write(f"High Confidence (>0.8): {(df_predictions['confidence'] > 0.8).sum()}\n")
        f.write(f"Low Confidence (<0.6): {(df_predictions['confidence'] < 0.6).sum()}\n")
    
    print(f"📊 Summary saved to: {summary_file}")
    
    # Show sample predictions
    print("\n🔍 SAMPLE PREDICTIONS:")
    print(df_predictions[['original_text', 'prediction', 'confidence']].head(10).to_string())
    
    return df_predictions

# ========================================
# 🎬 MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    print("🔮 MEMOTION 3.0 TEST PREDICTION PIPELINE")
    print("✅ Same preprocessing as training data")
    print("✅ Loading best trained model")
    print("✅ Predicting on test data (no labels)")
    print("✅ Saving predictions to CSV")
    print("-" * 60)
    
    # Run prediction pipeline
    results = predict_test_data()
    
    if results is not None:
        print("\n🎯 TEST PREDICTION COMPLETED!")
        print("✅ All preprocessing steps applied")
        print("✅ Predictions saved to CSV file")
        print("🚀 Ready for submission or analysis!")
    else:
        print("\n❌ Prediction failed. Check model path and files.")