#!/usr/bin/env python3
"""
🔮 COMPLETE TEST PREDICTION PIPELINE WITH GOOGLE DRIVE BACKUP
✅ Same preprocessing as train/val data
✅ Load best trained model
✅ Predict on test data with no labels
✅ Save predictions to CSV
✅ Backup model and results to Google Drive
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import json
import re
import shutil
import zipfile
from datetime import datetime
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
    GDRIVE_BACKUP_DIR = "/content/drive/MyDrive/Memotion_Backup"
    
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
# 💾 GOOGLE DRIVE BACKUP FUNCTIONS
# ========================================
def mount_google_drive():
    """Mount Google Drive in Colab"""
    try:
        from google.colab import drive
        print("🔄 Mounting Google Drive...")
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully!")
        return True
    except ImportError:
        print("⚠️ Not running in Google Colab - skipping Drive mount")
        return False
    except Exception as e:
        print(f"❌ Error mounting Google Drive: {e}")
        return False

def create_model_backup_zip(model_path, backup_dir):
    """Create a zip file of the trained model"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"memotion_model_backup_{timestamp}.zip"
    zip_path = os.path.join(backup_dir, zip_filename)
    
    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"📦 Creating model backup zip...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all model files
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, model_path)
                zipf.write(file_path, arcname)
                print(f"   Added: {arcname}")
    
    print(f"✅ Model backup created: {zip_path}")
    return zip_path

def backup_to_google_drive(model_path, predictions_csv, summary_file):
    """Backup model and predictions to Google Drive"""
    print("\n💾 BACKING UP TO GOOGLE DRIVE...")
    
    # Mount Google Drive
    if not mount_google_drive():
        print("❌ Cannot backup to Google Drive - not in Colab environment")
        return False
    
    try:
        # Create backup directory
        backup_dir = config.GDRIVE_BACKUP_DIR
        os.makedirs(backup_dir, exist_ok=True)
        print(f"📁 Backup directory: {backup_dir}")
        
        # Create timestamp for this backup session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(backup_dir, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        
        # 1. Backup model as zip
        if os.path.exists(model_path):
            model_zip = create_model_backup_zip(model_path, session_dir)
            print(f"✅ Model backed up to: {model_zip}")
        else:
            print("⚠️ Model path not found - skipping model backup")
        
        # 2. Copy predictions CSV
        if os.path.exists(predictions_csv):
            csv_backup = os.path.join(session_dir, f"test_predictions_{timestamp}.csv")
            shutil.copy2(predictions_csv, csv_backup)
            print(f"✅ Predictions CSV backed up to: {csv_backup}")
        else:
            print("⚠️ Predictions CSV not found - skipping CSV backup")
        
        # 3. Copy summary file
        if os.path.exists(summary_file):
            summary_backup = os.path.join(session_dir, f"predictions_summary_{timestamp}.txt")
            shutil.copy2(summary_file, summary_backup)
            print(f"✅ Summary backed up to: {summary_backup}")
        else:
            print("⚠️ Summary file not found - skipping summary backup")
        
        # 4. Create backup info file
        info_file = os.path.join(session_dir, "backup_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"🔮 MEMOTION 3.0 BACKUP SESSION\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Backup Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Predictions CSV: {predictions_csv}\n")
            f.write(f"Summary File: {summary_file}\n")
            f.write(f"Backup Location: {session_dir}\n")
            f.write(f"\nFiles in this backup:\n")
            for file in os.listdir(session_dir):
                f.write(f"  - {file}\n")
        
        print(f"📋 Backup info saved to: {info_file}")
        print(f"\n🎯 BACKUP COMPLETED SUCCESSFULLY!")
        print(f"📁 All files saved to: {session_dir}")
        print(f"🔗 Access via: Google Drive > MyDrive > Memotion_Backup > session_{timestamp}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during backup: {e}")
        return False

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
        f.write(f"\nTop 10 Most Confident Hate Speech Predictions:\n")
        hate_df = df_predictions[df_predictions['predicted_label'] == 1].nlargest(10, 'confidence')
        for idx, row in hate_df.iterrows():
            f.write(f"  Confidence: {row['confidence']:.4f} - Text: {row['original_text'][:100]}...\n")
        f.write(f"\nTop 10 Most Confident Not Hate Speech Predictions:\n")
        not_hate_df = df_predictions[df_predictions['predicted_label'] == 0].nlargest(10, 'confidence')
        for idx, row in not_hate_df.iterrows():
            f.write(f"  Confidence: {row['confidence']:.4f} - Text: {row['original_text'][:100]}...\n")
    
    print(f"📊 Summary saved to: {summary_file}")
    
    # Show sample predictions
    print("\n🔍 SAMPLE PREDICTIONS:")
    sample_df = df_predictions[['original_text', 'prediction', 'confidence']].head(10)
    for idx, row in sample_df.iterrows():
        print(f"Text: {row['original_text'][:60]}... -> {row['prediction']} ({row['confidence']:.3f})")
    
    # 10. Backup to Google Drive
    backup_success = backup_to_google_drive(config.MODEL_PATH, config.PREDICTIONS_CSV, summary_file)
    
    return df_predictions

# ========================================
# 🔮 PREDICT ON CUSTOM SAMPLES
# ========================================
def predict_custom_samples(text_image_pairs, model_path=None):
    """Predict on custom text-image pairs"""
    if model_path is None:
        model_path = config.MODEL_PATH
    
    print("🔮 Predicting on custom samples...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and processors
    model = load_trained_model(model_path, device)
    if model is None:
        return None
    
    tokenizer = VisualBertTokenizer.from_pretrained(config.VISUAL_BERT_MODEL)
    vit_processor = ViTImageProcessor.from_pretrained(config.VIT_MODEL)
    vit_model = ViTModel.from_pretrained(config.VIT_MODEL).to(device)
    vit_model.eval()
    
    results = []
    for i, (text, image_path) in enumerate(text_image_pairs):
        print(f"Processing sample {i+1}/{len(text_image_pairs)}...")
        
        # Process text
        clean_text = enhanced_ocr_cleaning(text)
        text_inputs = tokenizer(
            clean_text, 
            truncation=True, 
            padding='max_length', 
            max_length=config.MAX_LENGTH, 
            return_tensors='pt'
        ).to(device)
        
        # Process image
        try:
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image_inputs = vit_processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    vit_outputs = vit_model(**image_inputs)
                    image_features = vit_outputs.last_hidden_state[:, 0, :].unsqueeze(1)
            else:
                image_features = torch.zeros(1, 1, 768).to(device)
        except Exception as e:
            print(f"⚠️ Error processing image {image_path}: {e}")
            image_features = torch.zeros(1, 1, 768).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                visual_embeds=image_features,
                visual_attention_mask=torch.ones(1, 1).to(device)
            )
            
            probabilities = torch.softmax(outputs['logits'], dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
        
        result = {
            'text': text,
            'image_path': image_path,
            'cleaned_text': clean_text,
            'prediction': 'Hate Speech' if prediction.item() == 1 else 'Not Hate Speech',
            'confidence': confidence.item(),
            'prob_not_hate': probabilities[0][0].item(),
            'prob_hate': probabilities[0][1].item()
        }
        results.append(result)
        
        print(f"   Text: {text[:50]}...")
        print(f"   Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
    
    return results

# ========================================
# 🎬 MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    print("🔮 MEMOTION 3.0 TEST PREDICTION PIPELINE WITH GOOGLE DRIVE BACKUP")
    print("✅ Same preprocessing as training data")
    print("✅ Loading best trained model")
    print("✅ Predicting on test data (no labels)")
    print("✅ Saving predictions to CSV")
    print("✅ Backing up to Google Drive")
    print("-" * 80)
    
    # Run prediction pipeline
    results = predict_test_data()
    
    if results is not None:
        print("\n🎯 TEST PREDICTION COMPLETED!")
        print("✅ All preprocessing steps applied")
        print("✅ Predictions saved to CSV file")
        print("✅ Model and results backed up to Google Drive")
        print("🚀 Ready for submission or analysis!")
        
        # Example of custom prediction
        print("\n" + "="*60)
        print("🔮 EXAMPLE: CUSTOM PREDICTION")
        print("="*60)
        
        # Example custom samples (replace with your own)
        custom_samples = [
            ("This is a test message", ""),  # Text only, no image
            ("Another test with offensive content", ""),  # Another text sample
        ]
        
        print("Testing custom samples...")
        custom_results = predict_custom_samples(custom_samples)
        
        if custom_results:
            print("\n📋 Custom Prediction Results:")
            for i, result in enumerate(custom_results):
                print(f"{i+1}. '{result['text'][:60]}...' -> {result['prediction']} ({result['confidence']:.3f})")
        
    else:
        print("\n❌ Prediction failed. Check model path and files.")

# ========================================
# 📝 INSTRUCTIONS FOR COLAB
# ========================================
"""
🚀 HOW TO USE IN GOOGLE COLAB:

1. First, run your training pipeline to get the trained model
2. Then run this script to:
   - Load the trained model
   - Predict on test data
   - Save predictions to CSV
   - Backup everything to Google Drive

3. Convert this to .ipynb format and add these cells:

CELL 1 (Install packages):
```python
!pip install torch torchvision transformers datasets pillow scikit-learn pandas numpy tqdm matplotlib seaborn accelerate
```

CELL 2 (Run predictions):
```python
# Run the complete test prediction pipeline
results = predict_test_data()
```

CELL 3 (Custom predictions):
```python
# Test on your own samples
custom_samples = [
    ("Your custom text here", "path/to/image.jpg"),
    ("Another text sample", ""),  # No image
]
custom_results = predict_custom_samples(custom_samples)
```

4. Your files will be saved to:
   - Local: ./test_predictions.csv
   - Google Drive: /content/drive/MyDrive/Memotion_Backup/session_TIMESTAMP/

5. Download the CSV or access it from Google Drive!
"""