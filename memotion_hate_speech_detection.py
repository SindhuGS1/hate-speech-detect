#!/usr/bin/env python3
"""
Memotion 3.0 Hate Speech Detection
Multi-Modal Approach using ResNet + BERT + VisualBERT

This script implements a complete pipeline for hate speech detection in memes
using a fusion of vision and language models.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("🔍🛡️ Memotion 3.0 Hate Speech Detection")
print("Multi-Modal Approach using ResNet + BERT + VisualBERT")
print("=" * 60)

# ============================================================================
# 📦 Install and Import Dependencies
# ============================================================================

def install_requirements():
    """Install required packages"""
    packages = [
        "transformers==4.35.0",
        "torch", "torchvision", 
        "datasets", "evaluate",
        "scikit-learn", "accelerate",
        "Pillow", "matplotlib", "seaborn",
        "pandas", "numpy", "tqdm"
    ]
    
    for package in packages:
        os.system(f"pip install {package}")

# Uncomment the next line if running in a fresh environment
# install_requirements()

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm

# PyTorch and related
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchvision.transforms as transforms
import torchvision.models as models

# Transformers and Hugging Face
from transformers import (
    AutoTokenizer, AutoModel,
    VisualBertModel, VisualBertConfig,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup
)

# Image processing
from PIL import Image

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import evaluate

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# ============================================================================
# 📁 Configuration and Paths
# ============================================================================

# Adjust these paths according to your setup
BASE_PATH = "/content/drive/MyDrive/Memotion3/"  # Google Drive path
DATA_PATH = "/content/memotion_data/"            # Local data path
OUTPUT_DIR = "/content/model_outputs/"           # Output directory
CACHE_DIR = "/content/feature_cache/"            # Feature cache directory

# Create directories
for path in [DATA_PATH, OUTPUT_DIR, CACHE_DIR]:
    os.makedirs(path, exist_ok=True)
    os.makedirs(f"{path}/trainImages", exist_ok=True)
    os.makedirs(f"{path}/valImages", exist_ok=True)
    os.makedirs(f"{path}/testImages", exist_ok=True)

print(f"📁 Directories created: {DATA_PATH}, {OUTPUT_DIR}, {CACHE_DIR}")

# ============================================================================
# 🗂️ Data Loading and Preprocessing
# ============================================================================

def extract_dataset(base_path, data_path):
    """Extract dataset from zip files if they exist"""
    import zipfile
    
    zip_files = [
        ("trainImages.zip", "trainImages"),
        ("valImages.zip", "valImages"), 
        ("testImages.zip", "testImages")
    ]
    
    for zip_name, folder_name in zip_files:
        zip_path = os.path.join(base_path, zip_name)
        extract_path = os.path.join(data_path, folder_name)
        
        if os.path.exists(zip_path):
            print(f"📦 Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)
            print(f"✅ {zip_name} extracted")
        else:
            print(f"⚠️ {zip_path} not found")

def load_memotion_data(base_path):
    """Load Memotion 3.0 dataset CSV files"""
    
    # Load train data
    train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    print(f"📊 Train data loaded: {len(train_df)} samples")
    
    # Load validation data (handle potential TSV format)
    try:
        val_df = pd.read_csv(os.path.join(base_path, 'val.csv'))
    except:
        val_df = pd.read_csv(os.path.join(base_path, 'val.csv'), sep='\t', on_bad_lines='skip')
    print(f"📊 Validation data loaded: {len(val_df)} samples")
    
    # Load test data
    test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
    print(f"📊 Test data loaded: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def create_hate_labels(df):
    """Convert offensive categories to binary hate labels"""
    hate_categories = ['offensive', 'very_offensive', 'slight', 'hateful_offensive']
    non_hate_categories = ['not_offensive']
    
    df['label'] = df['offensive'].apply(
        lambda x: 1 if x in hate_categories else (0 if x in non_hate_categories else None)
    )
    
    # Remove rows with None labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    return df

def clean_ocr_text(text):
    """Clean OCR extracted text"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove newlines and carriage returns
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove unwanted symbols, keep basic punctuation
    text = re.sub(r'[^\w\s.,!?\'"\-]', '', text)
    
    # Remove extra whitespace
    text = text.strip()
    
    return text

def filter_valid_samples(df, image_folder, dataset_name):
    """Filter samples with valid images and non-empty OCR"""
    valid_rows = []
    
    print(f"🔍 Filtering {dataset_name} data...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
        # Check OCR text
        ocr_text = str(row.get('ocr_clean', '')).strip()
        
        # Check image existence
        image_name = f"{row['id']}.jpg"
        image_path = os.path.join(image_folder, image_name)
        
        # Keep samples with non-empty OCR AND existing image
        if len(ocr_text) > 0 and os.path.exists(image_path):
            valid_rows.append(row)
    
    # Create filtered dataframe
    filtered_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    
    print(f"✅ {dataset_name}: {len(filtered_df)} valid samples from {len(df)} total")
    print(f"   Retention rate: {len(filtered_df)/len(df)*100:.1f}%")
    
    return filtered_df

# ============================================================================
# 🏗️ Model Components
# ============================================================================

class ResNetFeatureExtractor(nn.Module):
    """ResNet-based image feature extractor"""
    def __init__(self, model_name='resnet50', feature_dim=2048):
        super(ResNetFeatureExtractor, self).__init__()
        
        # Load pre-trained ResNet
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add adaptive pooling and projection
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(feature_dim, 768)  # Project to BERT dimension
        
        # Freeze ResNet parameters (optional)
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        with torch.no_grad():  # Since we froze parameters
            features = self.backbone(images)
        
        features = self.pool(features).squeeze(-1).squeeze(-1)  # [batch, 2048]
        features = self.projection(features)  # [batch, 768]
        
        return features

class MultiModalHateSpeechClassifier(nn.Module):
    """Multi-modal hate speech classifier using VisualBERT"""
    
    def __init__(self, num_classes=2, class_weights=None, dropout_rate=0.3):
        super(MultiModalHateSpeechClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Load VisualBERT configuration and model
        self.config = VisualBertConfig.from_pretrained(
            'uclanlp/visualbert-nlvr2-coco-pre',
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )
        
        self.visualbert = VisualBertModel.from_pretrained(
            'uclanlp/visualbert-nlvr2-coco-pre',
            config=self.config
        )
        
        # Projection layers
        self.visual_projection = nn.Sequential(
            nn.Linear(768, 768),  # ResNet features to VisualBERT dimension
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, num_classes)
        )
        
        # Loss function with class weights
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, token_type_ids, visual_embeds, labels=None):
        batch_size = input_ids.size(0)
        
        # Project visual features
        visual_features = self.visual_projection(visual_embeds)  # [batch, 768]
        
        # Add visual features as single token per image
        visual_embeds_expanded = visual_features.unsqueeze(1)  # [batch, 1, 768]
        visual_attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
        visual_token_type_ids = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
        
        # VisualBERT forward pass
        outputs = self.visualbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_embeds=visual_embeds_expanded,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids
        )
        
        # Use pooled output for classification
        pooled_output = outputs.pooler_output  # [batch, 768]
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch, num_classes]
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'pooled_output': pooled_output
        }

class MemeDataset(Dataset):
    """Custom dataset for Memotion hate speech detection"""
    
    def __init__(self, dataframe, features_dict, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.features_dict = features_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get text and tokenize
        text = str(row['ocr_clean'])
        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        # Get pre-computed image features
        image_features = self.features_dict.get(row['id'], np.zeros(768, dtype=np.float32))
        image_features = torch.FloatTensor(image_features)
        
        # Prepare output
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0),
            'visual_embeds': image_features,
            'text': text,
            'image_id': row['id']
        }
        
        # Add label if available
        if 'label' in self.data.columns:
            item['labels'] = torch.tensor(row['label'], dtype=torch.long)
        
        return item

# ============================================================================
# 🔧 Utility Functions
# ============================================================================

def precompute_image_features(df, image_folder, feature_extractor, cache_file):
    """Pre-compute and cache image features"""
    
    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"📁 Loading cached features from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"🔄 Pre-computing image features for {len(df)} samples...")
    features_dict = {}
    
    # Image preprocessing transforms
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    feature_extractor.eval()
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            image_path = os.path.join(image_folder, f"{row['id']}.jpg")
            
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                image_tensor = image_transforms(image).unsqueeze(0).to(device)
                
                # Extract features
                features = feature_extractor(image_tensor)
                features_dict[row['id']] = features.cpu().numpy().squeeze()
                
            except Exception as e:
                print(f"⚠️ Error processing image {image_path}: {e}")
                # Use zero features for failed images
                features_dict[row['id']] = np.zeros(768, dtype=np.float32)
    
    # Cache features
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    print(f"✅ Features cached to {cache_file}")
    return features_dict

def compute_metrics(eval_pred):
    """Compute comprehensive evaluation metrics"""
    predictions, labels = eval_pred
    
    # Get predicted classes
    preds = np.argmax(predictions, axis=1)
    
    # Compute probabilities for AUC
    probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()
    
    # Load evaluation metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    
    # Calculate metrics
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=preds, references=labels, average='weighted')["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels, average='weighted')["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average='weighted')["f1"]
    
    # AUC score
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
    """Custom data collator for multi-modal features"""
    batch = {}
    
    # Text features
    batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
    batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
    batch['token_type_ids'] = torch.stack([f['token_type_ids'] for f in features])
    
    # Visual features
    batch['visual_embeds'] = torch.stack([f['visual_embeds'] for f in features])
    
    # Labels (if available)
    if 'labels' in features[0]:
        batch['labels'] = torch.stack([f['labels'] for f in features])
    
    return batch

# ============================================================================
# 🚀 Main Execution Pipeline
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print("🚀 Starting Memotion 3.0 Hate Speech Detection Pipeline")
    print("=" * 60)
    
    # Step 1: Extract and load data
    print("\n📁 Step 1: Data Loading and Preprocessing")
    try:
        # Note: Uncomment if you have zip files to extract
        # extract_dataset(BASE_PATH, DATA_PATH)
        
        # Load CSV data - you may need to adjust paths
        train_data, val_data, test_data = load_memotion_data(BASE_PATH)
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")
        print("💡 Please ensure your dataset paths are correct:")
        print(f"   Base path: {BASE_PATH}")
        print(f"   Expected files: train.csv, val.csv, test.csv")
        return
    
    # Step 2: Preprocess data
    print("\n🧹 Step 2: Data Cleaning and Filtering")
    
    # Create labels
    train_data = create_hate_labels(train_data)
    val_data = create_hate_labels(val_data)
    
    # Fix column names
    for df in [train_data, val_data, test_data]:
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    
    # Clean OCR text
    train_data['ocr_clean'] = train_data['ocr'].apply(clean_ocr_text)
    val_data['ocr_clean'] = val_data['ocr'].apply(clean_ocr_text)
    test_data['ocr_clean'] = test_data['ocr'].apply(clean_ocr_text)
    
    # Filter valid samples
    train_data = filter_valid_samples(train_data, f"{DATA_PATH}/trainImages", "Train")
    val_data = filter_valid_samples(val_data, f"{DATA_PATH}/valImages", "Validation")
    test_data = filter_valid_samples(test_data, f"{DATA_PATH}/testImages", "Test")
    
    print(f"\n📊 Final Dataset Sizes:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    
    # Step 3: Setup models and tokenizer
    print("\n🤖 Step 3: Model Initialization")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print("✅ BERT tokenizer loaded")
    
    # Initialize ResNet feature extractor
    resnet_extractor = ResNetFeatureExtractor('resnet50').to(device)
    resnet_extractor.eval()
    print("✅ ResNet feature extractor loaded")
    
    # Step 4: Pre-compute image features
    print("\n💾 Step 4: Feature Extraction and Caching")
    
    train_features = precompute_image_features(
        train_data, f"{DATA_PATH}/trainImages", resnet_extractor, 
        f"{CACHE_DIR}/train_features.pkl"
    )
    
    val_features = precompute_image_features(
        val_data, f"{DATA_PATH}/valImages", resnet_extractor,
        f"{CACHE_DIR}/val_features.pkl"
    )
    
    test_features = precompute_image_features(
        test_data, f"{DATA_PATH}/testImages", resnet_extractor,
        f"{CACHE_DIR}/test_features.pkl"
    )
    
    print(f"✅ Feature extraction complete!")
    print(f"   Feature dimension: {list(train_features.values())[0].shape}")
    
    # Step 5: Create datasets
    print("\n📊 Step 5: Dataset Creation")
    
    train_dataset = MemeDataset(train_data, train_features, tokenizer, max_length=128)
    val_dataset = MemeDataset(val_data, val_features, tokenizer, max_length=128)
    test_dataset = MemeDataset(test_data, test_features, tokenizer, max_length=128)
    
    print(f"✅ Datasets created:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    # Step 6: Compute class weights
    print("\n⚖️ Step 6: Class Weight Computation")
    
    train_labels = train_data['label'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class Distribution:")
    print(f"   Class 0 (Non-hate): {(train_labels == 0).sum()} samples")
    print(f"   Class 1 (Hate): {(train_labels == 1).sum()} samples")
    print(f"   Class weights: {class_weights}")
    
    # Step 7: Initialize model
    print("\n🏗️ Step 7: Model Initialization")
    
    model = MultiModalHateSpeechClassifier(
        num_classes=2,
        class_weights=class_weights,
        dropout_rate=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ VisualBERT classifier initialized:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Step 8: Training setup
    print("\n🚀 Step 8: Training Configuration")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # Training hyperparameters
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        
        # Optimization
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        
        # Best model selection
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # Logging
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to="none",
        
        # Performance
        dataloader_num_workers=2,
        fp16=True,
        
        # Reproducibility
        seed=42,
        data_seed=42,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print(f"✅ Trainer initialized")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Learning rate: {training_args.learning_rate}")
    
    # Step 9: Training
    print("\n🚀 Step 9: Model Training")
    print(f"Training on {len(train_dataset)} samples...")
    print(f"Validating on {len(val_dataset)} samples...")
    
    # Train the model
    training_result = trainer.train()
    
    print(f"✅ Training completed!")
    print(f"   Final training loss: {training_result.training_loss:.4f}")
    print(f"   Training time: {training_result.metrics['train_runtime']:.1f} seconds")
    
    # Step 10: Evaluation
    print("\n📊 Step 10: Model Evaluation")
    
    eval_results = trainer.evaluate()
    
    print("🎯 Validation Results:")
    print("=" * 40)
    for metric, value in eval_results.items():
        if metric.startswith('eval_'):
            metric_name = metric.replace('eval_', '').upper()
            if isinstance(value, float):
                print(f"{metric_name:>12}: {value:.4f}")
    
    # Check target accuracy
    target_accuracy = 0.80
    achieved_accuracy = eval_results.get('eval_accuracy', 0)
    print(f"\n🎯 Target vs Achieved:")
    print(f"   Target Accuracy: {target_accuracy:.1%}")
    print(f"   Achieved Accuracy: {achieved_accuracy:.1%}")
    
    if achieved_accuracy >= target_accuracy:
        print("🎉 Target accuracy achieved!")
    else:
        print(f"⚠️ Target accuracy not reached. Gap: {(target_accuracy - achieved_accuracy):.1%}")
    
    # Step 11: Test predictions
    print("\n🔮 Step 11: Test Set Predictions")
    
    test_predictions = trainer.predict(test_dataset)
    test_pred_classes = np.argmax(test_predictions.predictions, axis=1)
    test_pred_probs = torch.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy()
    
    # Create results dataframe
    test_results = test_data.copy()
    test_results['predicted_label'] = test_pred_classes
    test_results['hate_probability'] = test_pred_probs[:, 1]
    test_results['confidence'] = np.max(test_pred_probs, axis=1)
    
    # Save predictions
    predictions_file = f"{OUTPUT_DIR}/test_predictions.csv"
    test_results[['id', 'ocr_clean', 'predicted_label', 'hate_probability', 'confidence']].to_csv(
        predictions_file, index=False
    )
    
    print(f"✅ Test predictions saved to: {predictions_file}")
    print(f"   Predicted as Non-Hate: {(test_pred_classes == 0).sum()}")
    print(f"   Predicted as Hate: {(test_pred_classes == 1).sum()}")
    print(f"   Average confidence: {np.mean(test_results['confidence']):.3f}")
    
    # Step 12: Save model
    print("\n💾 Step 12: Model Saving")
    
    final_model_path = f"{OUTPUT_DIR}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save model configuration
    model_config = {
        'model_type': 'MultiModalHateSpeechClassifier',
        'num_classes': 2,
        'max_length': 128,
        'class_names': ['Non-Hate', 'Hate'],
        'class_weights': class_weights.cpu().tolist(),
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'final_metrics': {
            'accuracy': eval_results['eval_accuracy'],
            'f1': eval_results['eval_f1'],
            'precision': eval_results['eval_precision'],
            'recall': eval_results['eval_recall'],
            'auc': eval_results['eval_auc']
        }
    }
    
    with open(f"{final_model_path}/config.json", 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"💾 Model saved to: {final_model_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("🎯 FINAL SUMMARY")
    print("="*60)
    print(f"🏗️ Architecture: ResNet-50 + BERT + VisualBERT")
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    print(f"🎯 Performance: {achieved_accuracy:.1%} accuracy, {eval_results['eval_f1']:.4f} F1")
    print(f"💾 Outputs: {final_model_path}, {predictions_file}")
    print(f"🛡️ Mission accomplished: Offensive memes have nowhere to hide! 🔍")
    print("="*60)

if __name__ == "__main__":
    main()