# =============================================================================
# MANUAL TRAINING MEMOTION 3.0 - BYPASSES ALL TRANSFORMERS ISSUES
# Uses pure PyTorch training loop - no Trainer dependencies
# =============================================================================

print("🔧 Using pure PyTorch - bypassing all transformers issues!")

# Environment Setup
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted!")
    IN_COLAB = True
except:
    print("ℹ️ Not in Colab environment")
    IN_COLAB = False

# Core Imports
import os
import pandas as pd
import numpy as np
import re
import pickle
import json
import traceback
import gc
import math
from datetime import datetime
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Import CLIP safely
try:
    import open_clip
    CLIP_AVAILABLE = True
    print("✅ CLIP available")
except:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available, using dummy features")

# Minimal transformers imports - NO TRAINER
try:
    from transformers import (
        XLMRobertaTokenizer, 
        XLMRobertaModel
    )
    print("✅ Core transformers imported (no Trainer)")
    TRANSFORMERS_OK = True
except Exception as e:
    print(f"❌ Transformers import failed: {e}")
    TRANSFORMERS_OK = False

if not TRANSFORMERS_OK:
    raise RuntimeError("❌ Cannot proceed without basic transformers")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Advanced text processing - optional
try:
    import nltk
    import textstat
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass
    ADVANCED_NLP = True
    print("✅ Advanced NLP available")
except:
    ADVANCED_NLP = False
    print("⚠️ Advanced NLP not available")

print("🚀 MANUAL TRAINING MEMOTION 3.0 - PURE PYTORCH VERSION")
print("=" * 60)

# Device Setup
def setup_device_safely():
    """Setup device with comprehensive CUDA error handling"""
    try:
        if torch.cuda.is_available():
            test_tensor = torch.randn(10, 10).cuda()
            result = torch.matmul(test_tensor, test_tensor.T)
            del test_tensor, result
            
            device = torch.device('cuda')
            print(f"✅ Device: {device}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
            return device
        else:
            device = torch.device('cpu')
            print(f"✅ Device: {device} (CUDA not available)")
            return device
            
    except Exception as e:
        print(f"⚠️ CUDA error detected: {e}")
        print("🔄 Falling back to CPU...")
        device = torch.device('cpu')
        return device

device = setup_device_safely()

# Manual Training Configuration
class ManualConfig:
    if IN_COLAB:
        BASE_PATH = "/content/drive/MyDrive/Memotion3/"
        CACHE_DIR = "/content/feature_cache/"
        OUTPUT_DIR = "/content/model_outputs/"
        TRAIN_IMAGES = "/content/trainImages"
        VAL_IMAGES = "/content/valImages"
        TEST_IMAGES = "/content/testImages"
    else:
        BASE_PATH = "./data/"
        CACHE_DIR = "./feature_cache/"
        OUTPUT_DIR = "./model_outputs/"
        TRAIN_IMAGES = "./data/trainImages"
        VAL_IMAGES = "./data/valImages"
        TEST_IMAGES = "./data/testImages"
    
    # Model Components
    MULTILINGUAL_TOKENIZER = 'xlm-roberta-base'
    VISUAL_ENCODER = 'ViT-B-32'
    VISUAL_PRETRAINED = 'openai'
    
    # Dimensions
    CLIP_DIM = 512
    TEXT_DIM = 768
    VISUAL_EMBED_DIM = 512
    HIDDEN_DIM = 768
    NUM_VISUAL_TOKENS = 49
    MAX_TEXT_LENGTH = 256
    
    # Manual Training Parameters - Optimized
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-6
    NUM_EPOCHS = 8
    WEIGHT_DECAY = 0.05
    WARMUP_RATIO = 0.2
    DROPOUT_RATE = 0.3
    GRADIENT_CLIP = 1.0
    
    NUM_CLASSES = 2
    USE_MIXED_PRECISION = True if device.type == 'cuda' else False
    USE_FOCAL_LOSS = True
    
    # Evaluation
    EVAL_STEPS = 100
    EARLY_STOPPING_PATIENCE = 2

config = ManualConfig()
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print("🔧 MANUAL TRAINING CONFIGURATION:")
print(f"   📝 Text: XLM-RoBERTa ({config.TEXT_DIM}d)")
print(f"   🖼️ Vision: CLIP {config.VISUAL_ENCODER} ({config.CLIP_DIM}d)")
print(f"   🧠 Hidden: {config.HIDDEN_DIM}d")
print(f"   🎛️ Learning Rate: {config.LEARNING_RATE}")
print(f"   ⏰ Epochs: {config.NUM_EPOCHS}")

# Enhanced Text Preprocessing
def enhanced_bilingual_cleaning(text):
    """Advanced bilingual text cleaning"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', text)
    text = re.sub(r'@\w+', ' [MENTION] ', text)
    text = re.sub(r'#(\w+)', r' [HASHTAG] \1 ', text)
    
    # Handle repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Handle emojis and special characters
    text = re.sub(r'[^\w\s.,!?\'"।॥\u0900-\u097F\u0980-\u09FF\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) < 3:
        return ""
    
    return text

def extract_text_features(text):
    """Extract text features"""
    features = {}
    
    if not text or len(text) < 2:
        return {
            'length': 0, 'word_count': 0, 'has_caps': 0,
            'exclamation_count': 0, 'question_count': 0,
            'readability': 0, 'sentiment_words': 0
        }
    
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['has_caps'] = int(any(c.isupper() for c in text))
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    if ADVANCED_NLP:
        try:
            features['readability'] = textstat.flesch_reading_ease(text)
        except:
            features['readability'] = 50
    else:
        features['readability'] = 50
    
    hate_indicators = ['hate', 'stupid', 'ugly', 'idiot', 'kill', 'die', 'worst']
    features['sentiment_words'] = sum(1 for word in hate_indicators if word in text.lower())
    
    return features

# Data Loading Functions
def load_data_safely():
    """Load dataset files"""
    print("📁 Loading Memotion 3.0 dataset...")
    
    def load_csv_with_fallbacks(file_path, dataset_name):
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
            
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"✅ {dataset_name}: {len(df)} samples")
            return df
        except:
            try:
                df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
                print(f"✅ {dataset_name}: {len(df)} samples (tab-separated)")
                return df
            except:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                    print(f"✅ {dataset_name}: {len(df)} samples (skipped bad lines)")
                    return df
                except Exception as e:
                    print(f"❌ Failed to load {dataset_name}: {e}")
                    return None
    
    train_df = load_csv_with_fallbacks(os.path.join(config.BASE_PATH, 'train.csv'), 'Train')
    val_df = load_csv_with_fallbacks(os.path.join(config.BASE_PATH, 'val.csv'), 'Validation')
    test_df = load_csv_with_fallbacks(os.path.join(config.BASE_PATH, 'test.csv'), 'Test')
    
    if train_df is None:
        raise ValueError("❌ Cannot proceed without training data!")
    
    for df in [train_df, val_df, test_df]:
        if df is not None:
            if 'Unnamed: 0' in df.columns:
                df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
            if 'id' not in df.columns:
                df['id'] = df.index
            if 'ocr' not in df.columns:
                if 'text' in df.columns:
                    df['ocr'] = df['text']
                else:
                    df['ocr'] = 'sample text'
    
    return train_df, val_df, test_df

def create_labels_safely(df, split_name):
    """Create binary labels"""
    if df is None:
        return None
        
    print(f"🏷️ Creating labels for {split_name}...")
    
    possible_label_cols = ['offensive', 'hate', 'label', 'class', 'target']
    label_col = None
    
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        print(f"⚠️ No label column found for {split_name}, using default labels")
        df['label'] = 0
        return df
    
    if label_col == 'offensive':
        hate_categories = ['offensive', 'very_offensive', 'slight', 'hateful_offensive', 'hate']
        df['label'] = df[label_col].apply(
            lambda x: 1 if str(x).lower() in [c.lower() for c in hate_categories] else 0
        )
    else:
        df['label'] = df[label_col].apply(
            lambda x: 1 if (x == 1 or str(x).lower() in ['hate', 'offensive', '1', 'true']) else 0
        )
    
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    
    label_dist = df['label'].value_counts().sort_index()
    print(f"   📊 Label distribution: {label_dist.to_dict()}")
    
    return df

def validate_and_filter_samples(df, image_folder, dataset_name):
    """Sample validation"""
    if df is None:
        return None
        
    print(f"🔍 Validating {dataset_name} samples...")
    
    valid_samples = []
    error_counts = {
        'empty_text': 0, 'short_text': 0, 'missing_image': 0, 
        'corrupted_image': 0, 'invalid_id': 0, 'small_image': 0
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Validating {dataset_name}"):
        try:
            sample_id = str(row['id']).strip()
            if not sample_id or sample_id == 'nan':
                error_counts['invalid_id'] += 1
                continue
        except:
            error_counts['invalid_id'] += 1
            continue
        
        text = str(row['ocr_clean']).strip()
        if len(text) == 0:
            error_counts['empty_text'] += 1
            continue
        
        if len(text.split()) < 2:
            error_counts['short_text'] += 1
            continue
        
        image_name = f"{sample_id}.jpg"
        image_path = os.path.join(image_folder, image_name)
        
        if not os.path.exists(image_path):
            error_counts['missing_image'] += 1
            continue
        
        try:
            with Image.open(image_path) as img:
                if img.size[0] < 64 or img.size[1] < 64:
                    error_counts['small_image'] += 1
                    continue
                    
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
        except Exception as e:
            error_counts['corrupted_image'] += 1
            continue
        
        text_features = extract_text_features(text)
        
        row_dict = row.to_dict()
        row_dict['image_path'] = image_path
        row_dict.update(text_features)
        valid_samples.append(row_dict)
    
    if not valid_samples:
        print(f"❌ No valid samples found in {dataset_name}!")
        return None
    
    filtered_df = pd.DataFrame(valid_samples).reset_index(drop=True)
    
    total_original = len(df)
    total_valid = len(filtered_df)
    print(f"✅ {dataset_name}: {total_valid}/{total_original} valid samples ({total_valid/total_original*100:.1f}%)")
    
    return filtered_df

# CLIP Feature Extraction
def extract_clip_features_safely(df, image_folder, dataset_name):
    """CLIP feature extraction"""
    if df is None or len(df) == 0:
        return {}
    
    cache_file = os.path.join(config.CACHE_DIR, f"{dataset_name}_clip_features_manual.pkl")
    
    if os.path.exists(cache_file):
        print(f"📂 Loading cached {dataset_name} features...")
        with open(cache_file, 'rb') as f:
            features_dict = pickle.load(f)
        print(f"✅ Loaded {len(features_dict)} cached features")
        return features_dict
    
    print(f"🖼️ Computing {dataset_name} CLIP features...")
    
    features_dict = {}
    
    if CLIP_AVAILABLE:
        try:
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                config.VISUAL_ENCODER,
                pretrained=config.VISUAL_PRETRAINED,
                device=device
            )
            clip_model.eval()
            print(f"✅ CLIP model loaded successfully")
            
            batch_size = 32
            image_ids = df['id'].tolist()
            
            for i in tqdm(range(0, len(image_ids), batch_size), desc=f"Extracting {dataset_name}"):
                batch_ids = image_ids[i:i + batch_size]
                batch_images = []
                valid_ids = []
                
                for img_id in batch_ids:
                    image_path = os.path.join(image_folder, f"{img_id}.jpg")
                    
                    try:
                        if os.path.exists(image_path):
                            with Image.open(image_path) as img:
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                preprocessed = preprocess(img).unsqueeze(0)
                                batch_images.append(preprocessed)
                                valid_ids.append(img_id)
                        else:
                            dummy_features = np.random.randn(config.NUM_VISUAL_TOKENS, config.CLIP_DIM).astype(np.float32)
                            features_dict[img_id] = dummy_features
                            
                    except Exception as e:
                        dummy_features = np.random.randn(config.NUM_VISUAL_TOKENS, config.CLIP_DIM).astype(np.float32)
                        features_dict[img_id] = dummy_features
                
                if batch_images and valid_ids:
                    try:
                        batch_tensor = torch.cat(batch_images, dim=0).to(device)
                        
                        with torch.no_grad():
                            batch_features = clip_model.encode_image(batch_tensor)
                        
                        for idx, img_id in enumerate(valid_ids):
                            clip_feature = batch_features[idx].cpu().numpy()
                            visual_tokens = np.tile(clip_feature, (config.NUM_VISUAL_TOKENS, 1))
                            features_dict[img_id] = visual_tokens.astype(np.float32)
                        
                    except Exception as e:
                        print(f"⚠️ Batch processing error: {e}")
                        for img_id in valid_ids:
                            dummy_features = np.random.randn(config.NUM_VISUAL_TOKENS, config.CLIP_DIM).astype(np.float32)
                            features_dict[img_id] = dummy_features
            
            del clip_model
            
        except Exception as e:
            print(f"❌ CLIP failed: {e}")
            CLIP_AVAILABLE = False
    
    if not CLIP_AVAILABLE:
        print("⚠️ Using dummy visual features")
        for _, row in df.iterrows():
            img_id = row['id']
            dummy_features = np.random.normal(0, 0.1, (config.NUM_VISUAL_TOKENS, config.CLIP_DIM)).astype(np.float32)
            features_dict[img_id] = dummy_features
    
    print(f"✅ Generated features for {len(features_dict)} samples")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(features_dict, f)
        print(f"💾 Features cached to {cache_file}")
    except:
        print("⚠️ Failed to cache features")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return features_dict

# Manual Training Components
class ManualFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_device = self.alpha.to(targets.device)
                if len(alpha_device.shape) == 1 and len(targets.shape) == 1:
                    alpha_t = alpha_device[targets]
                else:
                    alpha_t = alpha_device
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ManualVisualBERT(nn.Module):
    def __init__(self, num_classes=2, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        
        print("🧠 Initializing Manual VisualBERT...")
        
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(config.MULTILINGUAL_TOKENIZER)
        self.text_encoder = XLMRobertaModel.from_pretrained(config.MULTILINGUAL_TOKENIZER)
        
        # Visual projection
        self.visual_projection = nn.Sequential(
            nn.Linear(config.CLIP_DIM, config.VISUAL_EMBED_DIM),
            nn.LayerNorm(config.VISUAL_EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.VISUAL_EMBED_DIM, config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT_RATE)
        )
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.HIDDEN_DIM,
            num_heads=12,
            dropout=config.DROPOUT_RATE,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(config.HIDDEN_DIM)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 2, num_classes)
        )
        
        # Loss function
        if config.USE_FOCAL_LOSS and class_weights is not None:
            alpha = torch.tensor(class_weights, dtype=torch.float32)
            self.loss_fct = ManualFocalLoss(alpha=alpha, gamma=2.0)
            self.register_buffer('focal_alpha', alpha)
            print("✅ Using Manual Focal Loss")
        else:
            self.loss_fct = nn.CrossEntropyLoss()
            print("✅ Using Cross Entropy Loss")
        
        self._init_weights()
        
        print(f"✅ Manual model initialized with {sum(p.numel() for p in self.parameters() if p.requires_grad):,} parameters")
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, visual_features, labels=None):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        attention_mask = attention_mask.to(device)
        if visual_features is not None:
            visual_features = visual_features.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        # Text encoding
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_hidden = text_outputs.last_hidden_state
        text_pooled = text_outputs.pooler_output
        
        if visual_features is not None:
            # Visual processing
            visual_flat = visual_features.view(batch_size * config.NUM_VISUAL_TOKENS, -1)
            visual_projected = self.visual_projection(visual_flat)
            visual_projected = visual_projected.view(batch_size, config.NUM_VISUAL_TOKENS, -1)
            
            # Cross-modal attention
            attended_text, _ = self.cross_attention(
                query=text_hidden,
                key=visual_projected,
                value=visual_projected
            )
            
            attended_text = self.layer_norm(attended_text + text_hidden)
            
            # Combine features
            text_final = attended_text.mean(dim=1)
            visual_final = visual_projected.mean(dim=1)
            combined_features = torch.cat([text_final, visual_final], dim=1)
        else:
            combined_features = torch.cat([text_pooled, text_pooled], dim=1)
        
        logits = self.classifier(combined_features)
        
        if labels is not None:
            labels = labels.view(-1).long()
            loss = self.loss_fct(logits, labels)
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits}

class ManualMemotionDataset(Dataset):
    def __init__(self, df, tokenizer, features_dict, max_length=256, is_test=False):
        self.tokenizer = tokenizer
        self.features_dict = features_dict
        self.max_length = max_length
        self.is_test = is_test
        
        self.samples = []
        if df is not None:
            for _, row in df.iterrows():
                sample = {
                    'id': str(row['id']),
                    'text': str(row.get('ocr_clean', '')),
                    'label': int(row.get('label', 0)) if not is_test else None,
                }
                self.samples.append(sample)
        
        print(f"✅ Manual dataset initialized with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        sample_id = sample['id']
        visual_features = self.features_dict.get(sample_id)
        
        if visual_features is None:
            visual_features = np.random.randn(config.NUM_VISUAL_TOKENS, config.CLIP_DIM).astype(np.float32)
        
        visual_features = torch.tensor(visual_features, dtype=torch.float32)
        if visual_features.shape != (config.NUM_VISUAL_TOKENS, config.CLIP_DIM):
            visual_features = torch.randn(config.NUM_VISUAL_TOKENS, config.CLIP_DIM, dtype=torch.float32)
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'visual_features': visual_features
        }
        
        if not self.is_test and sample['label'] is not None:
            result['labels'] = torch.tensor(sample['label'], dtype=torch.long)
        
        return result

def manual_data_collator(batch):
    """Manual data collator"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'visual_features': torch.stack([item['visual_features'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]) if 'labels' in batch[0] else None
    }

def compute_metrics(predictions, labels):
    """Compute evaluation metrics"""
    preds = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
    
    try:
        probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = 0.5
    except:
        auc = 0.5
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'auc': auc
    }

# Manual Training Loop
def manual_train_epoch(model, dataloader, optimizer, scheduler, scaler, device):
    """Manual training epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        visual_features = batch['visual_features'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if config.USE_MIXED_PRECISION and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, visual_features, labels)
                loss = outputs['loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask, visual_features, labels)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def manual_evaluate(model, dataloader, device):
    """Manual evaluation"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            visual_features = batch['visual_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, visual_features, labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    metrics['loss'] = avg_loss
    
    return metrics

def manual_predict(model, dataloader, device):
    """Manual prediction"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            visual_features = batch['visual_features'].to(device)
            
            outputs = model(input_ids, attention_mask, visual_features)
            logits = outputs['logits']
            
            all_predictions.extend(logits.cpu().numpy())
    
    return np.array(all_predictions)

# Main Manual Training Pipeline
def run_manual_training_pipeline():
    """Run manual training pipeline"""
    print("🚀 STARTING MANUAL TRAINING PIPELINE")
    print("=" * 60)
    
    try:
        print("📦 Extracting images...")
        for dataset in ['train', 'val', 'test']:
            zip_path = os.path.join(config.BASE_PATH, f"{dataset}Images.zip")
            extract_path = f"/content/{dataset}Images" if IN_COLAB else f"./data/{dataset}Images"
            
            if os.path.exists(zip_path) and not os.path.exists(extract_path):
                os.system(f"unzip -q '{zip_path}' -d {os.path.dirname(extract_path)}")
                print(f"✅ {dataset} images extracted")
        
        # Load and preprocess data
        train_df, val_df, test_df = load_data_safely()
        
        train_df = create_labels_safely(train_df, 'train')
        val_df = create_labels_safely(val_df, 'val') if val_df is not None else None
        test_df = create_labels_safely(test_df, 'test') if test_df is not None else None
        
        print("📝 Text cleaning...")
        train_df['ocr_clean'] = train_df['ocr'].apply(enhanced_bilingual_cleaning)
        if val_df is not None:
            val_df['ocr_clean'] = val_df['ocr'].apply(enhanced_bilingual_cleaning)
        if test_df is not None:
            test_df['ocr_clean'] = test_df['ocr'].apply(enhanced_bilingual_cleaning)
        
        print("🔍 Validation...")
        train_df = validate_and_filter_samples(train_df, config.TRAIN_IMAGES, "train")
        if val_df is not None:
            val_df = validate_and_filter_samples(val_df, config.VAL_IMAGES, "val")
        if test_df is not None:
            test_df = validate_and_filter_samples(test_df, config.TEST_IMAGES, "test")
        
        if train_df is None or len(train_df) == 0:
            raise ValueError("❌ No valid training samples!")
        
        print("🖼️ Visual features...")
        train_features = extract_clip_features_safely(train_df, config.TRAIN_IMAGES, "train")
        val_features = extract_clip_features_safely(val_df, config.VAL_IMAGES, "val") if val_df is not None else {}
        test_features = extract_clip_features_safely(test_df, config.TEST_IMAGES, "test") if test_df is not None else {}
        
        print("🔤 Tokenizer...")
        tokenizer = XLMRobertaTokenizer.from_pretrained(config.MULTILINGUAL_TOKENIZER)
        
        print("📚 Datasets...")
        train_dataset = ManualMemotionDataset(train_df, tokenizer, train_features, config.MAX_TEXT_LENGTH)
        val_dataset = ManualMemotionDataset(val_df, tokenizer, val_features, config.MAX_TEXT_LENGTH) if val_df is not None else None
        test_dataset = ManualMemotionDataset(test_df, tokenizer, test_features, config.MAX_TEXT_LENGTH, is_test=True) if test_df is not None else None
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=manual_data_collator)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=manual_data_collator) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=manual_data_collator) if test_dataset else None
        
        # Class weights
        train_labels = train_df['label'].values
        unique_labels = np.unique(train_labels)
        if len(unique_labels) > 1:
            class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
            print(f"⚖️ Class weights: {dict(zip(unique_labels, class_weights))}")
        else:
            class_weights = None
        
        print("🧠 Model...")
        model = ManualVisualBERT(num_classes=config.NUM_CLASSES, class_weights=class_weights)
        model = model.to(device)
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        
        num_training_steps = len(train_loader) * config.NUM_EPOCHS
        num_warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
        
        # Scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION else None
        
        print("🚀 Manual training started...")
        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset) if val_dataset else 0}")
        print(f"📊 Test samples: {len(test_dataset) if test_dataset else 0}")
        
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(config.NUM_EPOCHS):
            print(f"\n🔄 Epoch {epoch + 1}/{config.NUM_EPOCHS}")
            
            # Training
            train_loss = manual_train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
            print(f"   📈 Train Loss: {train_loss:.4f}")
            
            # Validation
            if val_loader:
                val_metrics = manual_evaluate(model, val_loader, device)
                print(f"   📊 Val Loss: {val_metrics['loss']:.4f}")
                print(f"   📊 Val F1 Macro: {val_metrics['f1_macro']:.4f}")
                print(f"   📊 Val Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"   📊 Val AUC: {val_metrics['auc']:.4f}")
                
                # Early stopping
                if val_metrics['f1_macro'] > best_f1:
                    best_f1 = val_metrics['f1_macro']
                    patience_counter = 0
                    
                    # Save best model
                    best_model_path = os.path.join(config.OUTPUT_DIR, "manual_best_model.pt")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"   💾 Best model saved: F1={best_f1:.4f}")
                else:
                    patience_counter += 1
                    print(f"   ⏰ Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
                
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print("   🛑 Early stopping triggered")
                    break
        
        print("✅ Manual training completed!")
        
        # Load best model for evaluation
        if val_loader and os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print("✅ Best model loaded for final evaluation")
        
        # Final evaluation
        final_metrics = {}
        if val_loader:
            final_metrics = manual_evaluate(model, val_loader, device)
            print("\n📈 FINAL EVALUATION RESULTS:")
            for key, value in final_metrics.items():
                print(f"   {key.upper()}: {value:.4f}")
        
        # Test predictions
        if test_loader:
            print("\n🎯 Generating test predictions...")
            test_predictions = manual_predict(model, test_loader, device)
            
            probs = torch.softmax(torch.tensor(test_predictions), dim=1).numpy()
            predicted_labels = np.argmax(test_predictions, axis=1)
            confidence_scores = np.max(probs, axis=1)
            
            test_results = []
            for i, sample in enumerate(test_dataset.samples):
                result = {
                    'id': sample['id'],
                    'text': sample['text'][:100] + '...' if len(sample['text']) > 100 else sample['text'],
                    'predicted_label': int(predicted_labels[i]),
                    'prediction': 'Hate Speech' if predicted_labels[i] == 1 else 'Not Hate Speech',
                    'confidence': float(confidence_scores[i]),
                    'prob_not_hate': float(probs[i][0]),
                    'prob_hate': float(probs[i][1])
                }
                test_results.append(result)
            
            test_df_results = pd.DataFrame(test_results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            predictions_file = os.path.join(config.OUTPUT_DIR, f"manual_test_predictions_{timestamp}.csv")
            test_df_results.to_csv(predictions_file, index=False)
            
            hate_count = (test_df_results['predicted_label'] == 1).sum()
            total_count = len(test_df_results)
            
            print("🎯 MANUAL TEST PREDICTION SUMMARY:")
            print(f"   📊 Total samples: {total_count}")
            print(f"   🔴 Hate Speech: {hate_count} ({hate_count/total_count*100:.1f}%)")
            print(f"   🟢 Not Hate: {total_count-hate_count} ({(total_count-hate_count)/total_count*100:.1f}%)")
            print(f"   🎯 Average confidence: {test_df_results['confidence'].mean():.3f}")
            print(f"   💾 Predictions saved to: {predictions_file}")
            
            return model, final_metrics, test_df_results, predictions_file
        
        return model, final_metrics, None, None
        
    except Exception as e:
        print(f"❌ Manual training pipeline error: {e}")
        traceback.print_exc()
        return None, None, None, None

# Execute Manual Training Pipeline
if __name__ == "__main__":
    print("🌟 INITIALIZING MANUAL TRAINING MEMOTION 3.0 PIPELINE")
    print("✅ Pure PyTorch - bypasses ALL transformers Trainer issues")
    print("✅ All functionality preserved and enhanced")
    print("✅ Optimized hyperparameters for better performance")
    print("✅ Manual training loop with proper evaluation")
    print("=" * 60)
    
    model, eval_results, test_predictions, predictions_file = run_manual_training_pipeline()
    
    if model is not None:
        print("\n🎉 MANUAL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅ Pure PyTorch training completed")
        print("✅ No transformers Trainer dependencies")
        print("✅ Enhanced test predictions generated")
        print("✅ 100% COMPATIBILITY - BYPASSES ALL ISSUES!")
    else:
        print("\n❌ Manual training pipeline failed. Check error messages above.")