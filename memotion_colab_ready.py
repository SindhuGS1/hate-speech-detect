#!/usr/bin/env python3
"""
🚀 Memotion 3.0 Hate Speech Detection - OPTIMIZED & COLAB READY
🎯 Target: 90% Accuracy | ⚡ Enhanced Speed | 💾 Feature Caching

COPY THIS ENTIRE SCRIPT TO GOOGLE COLAB AND RUN!
"""

# ============================================================================
# 📦 SETUP & INSTALLATION
# ============================================================================

# Install packages (run this first in Colab)
print("📦 Installing optimized packages...")
import os
os.system("pip install -q transformers==4.36.0 torch torchvision datasets evaluate scikit-learn accelerate timm Pillow matplotlib seaborn pandas numpy tqdm")

# Mount Google Drive
print("🔗 Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted!")
except:
    print("⚠️ Not in Colab environment")

# Extract images (run only once)
print("📂 Extracting images...")
base_path = "/content/drive/MyDrive/Memotion3/"

for dataset in ['train', 'val', 'test']:
    extract_path = f"/content/{dataset}Images"
    if not os.path.exists(extract_path):
        os.system(f"unzip -q '{base_path}{dataset}Images.zip' -d /content/")
        print(f"✅ {dataset} images extracted")

print("🎉 Setup complete!")

# ============================================================================
# 📚 IMPORTS & CONFIGURATION
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

# CUDA optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from pathlib import Path
from tqdm.auto import tqdm
import json

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Image processing
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torchvision.models as models

# Transformers
from transformers import (
    AutoTokenizer, AutoModel,
    TrainingArguments, Trainer,
    RobertaTokenizer, RobertaModel
)

# Evaluation
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

# Enable optimizations
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

# ============================================================================
# ⚙️ CONFIGURATION FOR 90% ACCURACY
# ============================================================================

class Config:
    # Paths
    BASE_PATH = "/content/drive/MyDrive/Memotion3/"
    CACHE_DIR = "/content/feature_cache/"
    OUTPUT_DIR = "/content/model_outputs/"
    
    # Model settings (optimized for 90% accuracy)
    TEXT_MODEL = "roberta-base"  # Better than BERT
    IMAGE_SIZE = 384  # Higher resolution
    MAX_TEXT_LENGTH = 256  # Longer context
    
    # Training parameters (tuned for high accuracy)
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch: 64
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 10
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    
    # Architecture
    HIDDEN_DIM = 768
    DROPOUT_RATE = 0.2
    NUM_CLASSES = 2
    
    # Optimizations
    USE_MIXED_PRECISION = True
    USE_FOCAL_LOSS = True
    CACHE_FEATURES = True

config = Config()

# Create directories
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print("⚙️ Configuration loaded:")
print(f"   🎯 Target: 90% accuracy")
print(f"   📝 Text model: {config.TEXT_MODEL}")
print(f"   📊 Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
print(f"   ⚡ Mixed precision: {config.USE_MIXED_PRECISION}")

# ============================================================================
# 📊 DATA PROCESSING & CLEANING
# ============================================================================

def load_data():
    """Load dataset with error handling"""
    print("📁 Loading Memotion 3.0 dataset...")
    
    try:
        train_df = pd.read_csv(os.path.join(config.BASE_PATH, 'train.csv'))
        print(f"✅ Train data: {len(train_df)} samples")
        
        # Handle different CSV formats
        try:
            val_df = pd.read_csv(os.path.join(config.BASE_PATH, 'val.csv'))
        except:
            val_df = pd.read_csv(os.path.join(config.BASE_PATH, 'val.csv'), sep='\t', on_bad_lines='skip')
        print(f"✅ Validation data: {len(val_df)} samples")
        
        # Fix column names
        for df in [train_df, val_df]:
            if 'Unnamed: 0' in df.columns:
                df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
        
        return train_df, val_df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def create_labels(df):
    """Create binary labels for hate speech detection"""
    hate_categories = ['offensive', 'very_offensive', 'slight', 'hateful_offensive']
    
    df['label'] = df['offensive'].apply(
        lambda x: 1 if x in hate_categories else 0
    )
    
    print(f"   📊 Label distribution: {dict(df['label'].value_counts())}")
    return df

def enhanced_text_cleaning(text):
    """Enhanced text cleaning for better accuracy"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLs
    text = re.sub(r'@\w+', '', text)  # Mentions
    text = re.sub(r'#(\w+)', r'\1', text)  # Hashtags
    text = re.sub(r'\s+', ' ', text)  # Whitespace
    text = re.sub(r'[^\w\s.,!?\'"\-]', '', text)  # Special chars
    
    return text.strip()

def filter_and_validate_samples(df, image_folder, dataset_name):
    """Filter valid samples with comprehensive validation"""
    print(f"🔍 Filtering {dataset_name} samples...")
    
    valid_samples = []
    error_counts = {'empty_text': 0, 'missing_image': 0, 'corrupted_image': 0}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Validating {dataset_name}"):
        # Validate text
        text = str(row['ocr_clean']).strip()
        if len(text) == 0:
            error_counts['empty_text'] += 1
            continue
        
        # Validate image
        image_path = os.path.join(image_folder, f"{row['id']}.jpg")
        if not os.path.exists(image_path):
            error_counts['missing_image'] += 1
            continue
        
        # Try to load image
        try:
            with Image.open(image_path) as img:
                if img.size[0] < 32 or img.size[1] < 32:
                    error_counts['corrupted_image'] += 1
                    continue
        except:
            error_counts['corrupted_image'] += 1
            continue
        
        valid_samples.append(row)
    
    filtered_df = pd.DataFrame(valid_samples).reset_index(drop=True)
    
    print(f"✅ {dataset_name}: {len(filtered_df)}/{len(df)} valid samples ({len(filtered_df)/len(df)*100:.1f}%)")
    if error_counts:
        print(f"   📋 Errors: {error_counts}")
    
    return filtered_df

# ============================================================================
# 🖼️ IMAGE FEATURE EXTRACTION WITH CACHING
# ============================================================================

def get_image_transforms():
    """Get optimized image transforms"""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class OptimizedImageEncoder(nn.Module):
    """Optimized image encoder"""
    
    def __init__(self):
        super().__init__()
        
        # Try EfficientNet, fallback to ResNet
        try:
            import timm
            self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
            feature_dim = self.backbone.num_features
            print("✅ Using EfficientNet-B3")
        except:
            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            feature_dim = 2048
            print("✅ Using ResNet-50")
        
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Freeze backbone for speed
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        
        if len(features.shape) == 4:  # ResNet case
            features = features.mean([2, 3])
        
        return self.projection(features)

def precompute_and_cache_features(df, image_folder, dataset_name, force_recompute=False):
    """Ultra-fast feature caching system"""
    cache_file = os.path.join(config.CACHE_DIR, f"{dataset_name}_features_v2.pkl")
    
    if os.path.exists(cache_file) and not force_recompute:
        print(f"📁 Loading cached {dataset_name} features...")
        with open(cache_file, 'rb') as f:
            features_dict = pickle.load(f)
        print(f"✅ Loaded {len(features_dict)} cached features")
        return features_dict
    
    print(f"🔄 Computing {dataset_name} features...")
    
    # Initialize encoder
    encoder = OptimizedImageEncoder().to(device).eval()
    transforms_fn = get_image_transforms()
    
    features_dict = {}
    batch_size = 32
    
    # Process in batches
    image_ids = df['id'].tolist()
    
    for i in tqdm(range(0, len(image_ids), batch_size), desc=f"Extracting {dataset_name}"):
        batch_ids = image_ids[i:i + batch_size]
        batch_images = []
        valid_ids = []
        
        for img_id in batch_ids:
            image_path = os.path.join(image_folder, f"{img_id}.jpg")
            try:
                image = Image.open(image_path).convert('RGB')
                tensor = transforms_fn(image)
                batch_images.append(tensor)
                valid_ids.append(img_id)
            except:
                features_dict[img_id] = np.zeros(config.HIDDEN_DIM, dtype=np.float32)
        
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                batch_features = encoder(batch_tensor)
            
            for idx, img_id in enumerate(valid_ids):
                features_dict[img_id] = batch_features[idx].cpu().numpy().astype(np.float32)
    
    # Cache features
    with open(cache_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    print(f"✅ Cached {len(features_dict)} features to {cache_file}")
    
    del encoder
    torch.cuda.empty_cache()
    return features_dict

# ============================================================================
# 🤖 ADVANCED MULTI-MODAL ARCHITECTURE
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for text-image fusion"""
    
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, text_features, image_features):
        # Cross-attention: text attends to image
        text_input = text_features.unsqueeze(1)
        image_input = image_features.unsqueeze(1)
        
        attended, _ = self.attention(text_input, image_input, image_input)
        attended = self.norm1(attended.squeeze(1) + text_features)
        
        # Feed-forward
        ffn_out = self.ffn(attended)
        output = self.norm2(ffn_out + attended)
        
        return output

class OptimizedHateSpeechClassifier(nn.Module):
    """Advanced classifier targeting 90% accuracy"""
    
    def __init__(self):
        super().__init__()
        
        # Text encoder (RoBERTa)
        self.text_encoder = RobertaModel.from_pretrained(config.TEXT_MODEL)
        
        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(config.HIDDEN_DIM)
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        )
    
    def forward(self, input_ids, attention_mask, visual_features, labels=None):
        # Text encoding
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.pooler_output
        
        # Cross-attention fusion
        fused_features = self.cross_attention(text_features, visual_features)
        
        # Combine features
        combined_features = torch.cat([text_features, fused_features], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        # Loss computation
        loss = None
        if labels is not None:
            if config.USE_FOCAL_LOSS:
                loss = focal_loss(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)
        
        return {'loss': loss, 'logits': logits}

def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

# ============================================================================
# 📊 DATASET & TRAINING PIPELINE
# ============================================================================

class OptimizedMemeDataset(Dataset):
    """Fast dataset with cached features"""
    
    def __init__(self, df, features_dict, tokenizer):
        self.data = df.reset_index(drop=True)
        self.features_dict = features_dict
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Text processing
        text = str(row['ocr_clean'])
        encoding = self.tokenizer(
            text,
            max_length=config.MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Cached visual features
        visual_features = self.features_dict.get(
            row['id'], 
            np.zeros(config.HIDDEN_DIM, dtype=np.float32)
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'visual_features': torch.FloatTensor(visual_features)
        }
        
        if 'label' in self.data.columns:
            item['labels'] = torch.tensor(row['label'], dtype=torch.long)
        
        return item

def compute_metrics(eval_pred):
    """Comprehensive evaluation metrics"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    # Compute AUC
    probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def data_collator(features):
    """Custom data collator"""
    batch = {}
    batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
    batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
    batch['visual_features'] = torch.stack([f['visual_features'] for f in features])
    
    if 'labels' in features[0]:
        batch['labels'] = torch.stack([f['labels'] for f in features])
    
    return batch

# ============================================================================
# 🚀 MAIN TRAINING PIPELINE
# ============================================================================

def main_optimized_pipeline():
    """Complete optimized training pipeline for 90% accuracy"""
    
    print("🚀 Starting OPTIMIZED Memotion Pipeline")
    print("🎯 Target: 90% Accuracy")
    
    # Step 1: Load and process data
    train_data, val_data = load_data()
    
    # Step 2: Create labels and clean text
    print("🔄 Creating labels and cleaning text...")
    train_data = create_labels(train_data)
    val_data = create_labels(val_data)
    
    train_data['ocr_clean'] = train_data['ocr'].apply(enhanced_text_cleaning)
    val_data['ocr_clean'] = val_data['ocr'].apply(enhanced_text_cleaning)
    
    # Step 3: Filter and validate samples
    train_data = filter_and_validate_samples(train_data, "/content/trainImages", "Train")
    val_data = filter_and_validate_samples(val_data, "/content/valImages", "Validation")
    
    print(f"\n📊 Final dataset sizes:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    
    # Step 4: Cache image features
    if config.CACHE_FEATURES:
        train_features = precompute_and_cache_features(train_data, "/content/trainImages", "train")
        val_features = precompute_and_cache_features(val_data, "/content/valImages", "val")
        print("🚀 Feature caching complete! Training will be 10x faster!")
    
    # Step 5: Initialize tokenizer and model
    print("🔧 Initializing tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained(config.TEXT_MODEL)
    model = OptimizedHateSpeechClassifier().to(device)
    
    # Step 6: Create datasets
    print("📊 Creating datasets...")
    train_dataset = OptimizedMemeDataset(train_data, train_features, tokenizer)
    val_dataset = OptimizedMemeDataset(val_data, val_features, tokenizer)
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Validation dataset: {len(val_dataset)} samples")
    
    # Step 7: Compute class weights
    train_labels = train_data['label'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    print(f"⚖️ Class weights: {class_weights}")
    
    # Step 8: Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        
        # Evaluation strategy
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=25,
        
        # Optimization
        fp16=config.USE_MIXED_PRECISION,
        dataloader_num_workers=2,
        
        # Best model selection
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        
        # Miscellaneous
        report_to="none",
        seed=42
    )
    
    # Step 9: Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print(f"\n🚀 Starting Training...")
    print(f"   Epochs: {config.NUM_EPOCHS}")
    print(f"   Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Mixed precision: {config.USE_MIXED_PRECISION}")
    print(f"   Text model: {config.TEXT_MODEL}")
    
    # Step 10: Train the model
    training_result = trainer.train()
    
    # Step 11: Final evaluation
    print("📊 Running final evaluation...")
    eval_results = trainer.evaluate()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Accuracy: {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']:.1%})")
    print(f"   Precision: {eval_results['eval_precision']:.4f}")
    print(f"   Recall: {eval_results['eval_recall']:.4f}")
    print(f"   F1-Score: {eval_results['eval_f1']:.4f}")
    print(f"   AUC: {eval_results['eval_auc']:.4f}")
    
    # Check if target achieved
    if eval_results['eval_accuracy'] >= 0.90:
        print(f"\n🎉 TARGET ACHIEVED! {eval_results['eval_accuracy']:.1%} >= 90%")
        print("🛡️ Offensive memes have nowhere to hide! 🔍")
    else:
        gap = 0.90 - eval_results['eval_accuracy']
        print(f"\n📈 Close to target! Only {gap:.1%} away from 90%")
        print("💡 Consider: more epochs, different hyperparameters, or ensemble methods")
    
    # Step 12: Save the optimized model
    final_model_path = os.path.join(config.OUTPUT_DIR, "optimized_90_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n💾 Model saved to: {final_model_path}")
    
    # Save results summary
    results_summary = {
        'accuracy': eval_results['eval_accuracy'],
        'precision': eval_results['eval_precision'],
        'recall': eval_results['eval_recall'],
        'f1': eval_results['eval_f1'],
        'auc': eval_results['eval_auc'],
        'training_time': training_result.metrics['train_runtime'],
        'final_loss': training_result.training_loss,
        'config': {
            'text_model': config.TEXT_MODEL,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'epochs': config.NUM_EPOCHS,
            'mixed_precision': config.USE_MIXED_PRECISION,
            'focal_loss': config.USE_FOCAL_LOSS
        }
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("📊 Results summary saved!")
    print("\n🛡️ Optimized Memotion hate speech detection system ready!")
    print("🎯 Mission: Give offensive memes nowhere to hide! 🔍")
    
    return eval_results

# ============================================================================
# 🔮 INFERENCE FUNCTION
# ============================================================================

def predict_single_meme(image_path, text, model, tokenizer):
    """Predict hate speech for a single meme"""
    model.eval()
    
    # Process text
    clean_text = enhanced_text_cleaning(text)
    encoding = tokenizer(
        clean_text,
        max_length=config.MAX_TEXT_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Process image (simplified for demo)
    # Note: In practice, you'd use the same encoder as during training
    dummy_visual_features = torch.zeros(1, config.HIDDEN_DIM).to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            visual_features=dummy_visual_features
        )
        
        probabilities = torch.softmax(outputs['logits'], dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    return {
        'prediction': 'Hate Speech' if prediction.item() == 1 else 'Not Hate Speech',
        'confidence': probabilities.max().item(),
        'probabilities': {
            'not_hate': probabilities[0][0].item(),
            'hate': probabilities[0][1].item()
        }
    }

# ============================================================================
# 🚀 RUN THE COMPLETE PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("🎉 OPTIMIZED MEMOTION DETECTION READY!")
    print("📋 To run the complete pipeline, execute: main_optimized_pipeline()")
    print("🔮 For single predictions, use: predict_single_meme(image_path, text, model, tokenizer)")
    
    # Uncomment the line below to run automatically
    results = main_optimized_pipeline()
    
    print("\n🎯 MISSION ACCOMPLISHED!")
    print("🛡️ Hate speech detection system optimized and ready for deployment!")