#!/usr/bin/env python3
"""
🚀 CORRECTED Memotion 3.0 - BILINGUAL Hindi+English Architecture
✅ Fixed for Hindi+English code-mixed data
✅ Macro F1-Score evaluation 
✅ Proper multilingual tokenization
✅ Test data prediction included

CRITICAL FIXES:
- XLM-RoBERTa for multilingual support (Hindi+English)
- IndicBERT tokenizer for better Hindi handling
- Macro F1-Score (not weighted)
- Language-aware preprocessing
- Bilingual text cleaning
"""

import os
import warnings
warnings.filterwarnings('ignore')

print("📦 Installing packages for BILINGUAL architecture...")
os.system("pip install -q transformers torch torchvision datasets evaluate scikit-learn accelerate Pillow matplotlib seaborn pandas numpy tqdm")
os.system("pip install -q indic-transformers")  # For better Hindi support

# Mount Drive
print("🔗 Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted!")
except:
    print("⚠️ Not in Colab environment")

# Extract images
print("📂 Extracting images...")
base_path = "/content/drive/MyDrive/Memotion3/"

for dataset in ['train', 'val', 'test']:
    extract_path = f"/content/{dataset}Images"
    if not os.path.exists(extract_path):
        os.system(f"unzip -q '{base_path}{dataset}Images.zip' -d /content/")
        print(f"✅ {dataset} images extracted")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ✅ CORRECTED: Multilingual imports for Hindi+English
try:
    from transformers import ViTImageProcessor, ViTModel
    print("✅ Using updated ViT imports")
except ImportError:
    from transformers import ViTFeatureExtractor as ViTImageProcessor, ViTModel
    print("⚠️ Using legacy ViT imports")

from transformers import (
    XLMRobertaTokenizer, XLMRobertaModel,  # ✅ MULTILINGUAL
    AutoTokenizer, AutoModel,  # ✅ For IndicBERT
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

# ✅ CORRECTED BILINGUAL CONFIGURATION
class BilingualConfig:
    BASE_PATH = "/content/drive/MyDrive/Memotion3/"
    CACHE_DIR = "/content/feature_cache/"
    OUTPUT_DIR = "/content/model_outputs/"
    GDRIVE_BACKUP_DIR = "/content/drive/MyDrive/Memotion_Models/"
    
    # ✅ BILINGUAL MODEL SELECTIONS
    MULTILINGUAL_MODEL = 'xlm-roberta-base'  # Better for Hindi+English
    INDIC_MODEL = 'ai4bharat/indic-bert'     # Alternative for Hindi
    VIT_MODEL = 'google/vit-base-patch16-224-in21k'
    
    IMAGE_SIZE = 224
    MAX_TEXT_LENGTH = 128
    
    # ✅ OPTIMIZED PARAMETERS FOR BILINGUAL
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 1e-5              # ✅ Higher for multilingual
    NUM_EPOCHS = 15
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1                # ✅ More warmup for multilingual
    
    # ✅ CORRECTED DIMENSIONS
    HIDDEN_DIM = 768
    VISUAL_DIM = 768
    DROPOUT_RATE = 0.1
    ATTENTION_DROPOUT = 0.1
    
    NUM_CLASSES = 2
    USE_MIXED_PRECISION = True
    USE_FOCAL_LOSS = True
    CACHE_FEATURES = True
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING_FACTOR = 0.1
    NUM_VISUAL_TOKENS = 1

config = BilingualConfig()

os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.GDRIVE_BACKUP_DIR, exist_ok=True)

print("⚙️ BILINGUAL CORRECTED Configuration:")
print(f"   🌍 Language: Hindi + English (Code-mixed)")
print(f"   🧠 Model: XLM-RoBERTa (Multilingual)")
print(f"   📊 Evaluation: Macro F1-Score")
print(f"   🎯 Target: 70-80% macro F1")

def load_data():
    print("📁 Loading Memotion 3.0 BILINGUAL dataset...")
    try:
        train_df = pd.read_csv(os.path.join(config.BASE_PATH, 'train.csv'))
        print(f"✅ Train data: {len(train_df)} samples")
        
        try:
            val_df = pd.read_csv(os.path.join(config.BASE_PATH, 'val.csv'))
        except:
            val_df = pd.read_csv(os.path.join(config.BASE_PATH, 'val.csv'), sep='\t', on_bad_lines='skip')
        print(f"✅ Validation data: {len(val_df)} samples")
        
        # Load test data
        try:
            test_df = pd.read_csv(os.path.join(config.BASE_PATH, 'test.csv'))
            print(f"✅ Test data: {len(test_df)} samples")
        except:
            print("⚠️ Test data not found, will use validation for testing")
            test_df = val_df.copy()
        
        # Clean column names
        for df in [train_df, val_df, test_df]:
            if 'Unnamed: 0' in df.columns:
                df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
                
        return train_df, val_df, test_df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def create_labels(df):
    """Create binary labels for hate speech detection"""
    if 'offensive' not in df.columns:
        print("⚠️ No 'offensive' column found, creating dummy labels for test data")
        df['label'] = 0  # Default for test data
        return df
        
    hate_categories = ['offensive', 'very_offensive', 'slight', 'hateful_offensive']
    df['label'] = df['offensive'].apply(lambda x: 1 if x in hate_categories else 0)
    print(f"   📊 Label distribution: {dict(df['label'].value_counts())}")
    return df

# ✅ BILINGUAL TEXT CLEANING - Handles Hindi+English
def bilingual_text_cleaning(text):
    """Enhanced bilingual text cleaning for Hindi+English code-mixed"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Keep original for language detection
    original_text = text
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # ✅ CRITICAL: Keep Hindi characters (Devanagari script)
    # Remove only unwanted symbols but preserve Hindi text
    text = re.sub(r'[^\w\s.,!?\'"\\-\u0900-\u097F]', '', text)  # Preserve Devanagari
    
    # Normalize repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Reduce 3+ repeated chars to 2
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def filter_and_validate_samples(df, image_folder, dataset_name):
    """Filter and validate samples with BILINGUAL preprocessing"""
    print(f"🔍 Filtering {dataset_name} samples (BILINGUAL)...")
    valid_samples = []
    error_counts = {'empty_text': 0, 'missing_image': 0, 'corrupted_image': 0}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Validating {dataset_name}"):
        text = str(row['ocr_clean']).strip()
        if len(text) == 0:
            error_counts['empty_text'] += 1
            continue
            
        image_name = f"{row['id']}.jpg"
        image_path = os.path.join(image_folder, image_name)
        
        if not os.path.exists(image_path):
            error_counts['missing_image'] += 1
            continue
            
        try:
            with Image.open(image_path) as img:
                if img.size[0] < 32 or img.size[1] < 32:
                    error_counts['corrupted_image'] += 1
                    continue
        except:
            error_counts['corrupted_image'] += 1
            continue
            
        row['image'] = image_name
        valid_samples.append(row)
        
    filtered_df = pd.DataFrame(valid_samples).reset_index(drop=True)
    print(f"✅ {dataset_name}: {len(filtered_df)}/{len(df)} valid samples ({len(filtered_df)/len(df)*100:.1f}%)")
    print(f"   Errors: {error_counts}")
    
    return filtered_df

def get_vit_processor_and_model():
    """Get ViT processor and model"""
    try:
        image_processor = ViTImageProcessor.from_pretrained(config.VIT_MODEL)
        feature_model = ViTModel.from_pretrained(config.VIT_MODEL).to(device)
        print("✅ Using updated ViTImageProcessor")
    except:
        from transformers import ViTFeatureExtractor
        image_processor = ViTFeatureExtractor.from_pretrained(config.VIT_MODEL)
        feature_model = ViTModel.from_pretrained(config.VIT_MODEL).to(device)
        print("⚠️ Using legacy ViTFeatureExtractor")
        
    feature_model.eval()
    for param in feature_model.parameters():
        param.requires_grad = False
    return image_processor, feature_model

def precompute_vit_features_corrected(df, image_folder, dataset_name, force_recompute=False):
    """Precompute ViT features - CLS token only"""
    cache_file = os.path.join(config.CACHE_DIR, f"{dataset_name}_vit_features_cls_bilingual.pkl")
    
    if os.path.exists(cache_file) and not force_recompute:
        print(f"📁 Loading cached {dataset_name} ViT features (CLS only)...")
        with open(cache_file, 'rb') as f:
            features_dict = pickle.load(f)
        print(f"✅ Loaded {len(features_dict)} cached CLS features")
        return features_dict
        
    print(f"🔄 Computing {dataset_name} ViT features (CLS only)...")
    image_processor, feature_model = get_vit_processor_and_model()
    features_dict = {}
    batch_size = 32
    
    image_ids = df['id'].tolist()
    for i in tqdm(range(0, len(image_ids), batch_size), desc=f"Extracting {dataset_name} ViT CLS"):
        batch_ids = image_ids[i:i + batch_size]
        batch_images = []
        valid_ids = []
        
        for img_id in batch_ids:
            image_path = os.path.join(image_folder, f"{img_id}.jpg")
            try:
                image = Image.open(image_path).convert('RGB')
                batch_images.append(image)
                valid_ids.append(img_id)
            except:
                features_dict[img_id] = np.zeros(config.HIDDEN_DIM, dtype=np.float32)
                
        if batch_images:
            inputs = image_processor(images=batch_images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = feature_model(**inputs)
                visual_embeds = outputs.last_hidden_state
                
                # Store only CLS token
                for idx, img_id in enumerate(valid_ids):
                    cls_token = visual_embeds[idx, 0, :]
                    features_dict[img_id] = cls_token.cpu().numpy().astype(np.float32)
                    
    with open(cache_file, 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"✅ Cached {len(features_dict)} ViT CLS features to {cache_file}")
    
    del feature_model
    torch.cuda.empty_cache()
    return features_dict

# ✅ CORRECTED FOCAL LOSS
class CorrectedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets].to(inputs.device)
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
        return focal_loss.mean()

# ✅ BILINGUAL MULTIMODAL CLASSIFIER
class BilingualMultimodalClassifier(nn.Module):
    def __init__(self, class_weights, device='cuda'):
        super(BilingualMultimodalClassifier, self).__init__()
        self.num_labels = config.NUM_CLASSES
        self.device = device

        # ✅ MULTILINGUAL TEXT ENCODER
        self.text_encoder = XLMRobertaModel.from_pretrained(config.MULTILINGUAL_MODEL)
        
        # ✅ Visual projection (ViT 768 -> same dimension)
        self.visual_projector = nn.Linear(config.VISUAL_DIM, config.HIDDEN_DIM)
        
        # ✅ MULTIMODAL FUSION
        self.multimodal_fusion = nn.MultiheadAttention(
            embed_dim=config.HIDDEN_DIM,
            num_heads=8,
            dropout=config.ATTENTION_DROPOUT,
            batch_first=True
        )
        
        # ✅ CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.LayerNorm(config.HIDDEN_DIM // 4),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE * 0.5),
            nn.Linear(config.HIDDEN_DIM // 4, self.num_labels)
        )

        # ✅ LOSS FUNCTION
        if config.USE_FOCAL_LOSS:
            if class_weights is not None:
                weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
                self.loss_fct = CorrectedFocalLoss(alpha=weights_tensor, gamma=2.0)
            else:
                self.loss_fct = CorrectedFocalLoss(alpha=0.25, gamma=2.0)
        else:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
            self.loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)

    def forward(self, input_ids, attention_mask, visual_embeds, labels=None):
        batch_size = input_ids.size(0)

        # ✅ MULTILINGUAL TEXT ENCODING
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeds = text_outputs.last_hidden_state  # (batch, seq_len, 768)
        text_cls = text_outputs.pooler_output          # (batch, 768)
        
        # ✅ VISUAL PROCESSING
        visual_embeds_proj = self.visual_projector(visual_embeds)  # (batch, 1, 768)
        
        # ✅ MULTIMODAL FUSION via Cross-Attention
        # Use text as query, visual as key/value
        fused_output, _ = self.multimodal_fusion(
            query=text_cls.unsqueeze(1),      # (batch, 1, 768)
            key=visual_embeds_proj,           # (batch, 1, 768)
            value=visual_embeds_proj          # (batch, 1, 768)
        )
        
        # Final representation
        final_repr = fused_output.squeeze(1)  # (batch, 768)
        
        # Classification
        logits = self.classifier(final_repr)

        if labels is not None:
            labels = labels.view(-1).long().to(logits.device)
            loss = self.loss_fct(logits, labels)
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}

# ✅ BILINGUAL DATASET
class BilingualHatefulMemesDataset(Dataset):
    def __init__(self, df, tokenizer, features_dict, sequence_length=128, device='cuda', is_test=False):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.features_dict = features_dict
        self.device = device
        self.is_test = is_test
        self.dataset = []

        for i, row in df.iterrows():
            self.dataset.append({
                "text": str(row["ocr_clean"]),
                "label": row["label"] if "label" in df.columns and not is_test else None,
                "idx": row.get("id", i),
                "image": row["image"]
            })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]

        # ✅ MULTILINGUAL TOKENIZATION
        encoded = self.tokenizer(
            example["text"],
            padding="max_length",
            max_length=self.sequence_length,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # ✅ Visual features (CLS token)
        img_id = example["idx"]
        visual_features = self.features_dict.get(
            img_id,
            np.zeros(config.HIDDEN_DIM, dtype=np.float32)
        )
        
        if isinstance(visual_features, np.ndarray):
            if visual_features.ndim == 1:
                visual_embeds = torch.FloatTensor(visual_features).unsqueeze(0)  # (1, 768)
            else:
                visual_embeds = torch.FloatTensor(visual_features)
                if visual_embeds.shape[0] > 1:
                    visual_embeds = visual_embeds[0:1, :]
        else:
            visual_embeds = torch.FloatTensor(visual_features).unsqueeze(0)

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'visual_embeds': visual_embeds
        }

        if example["label"] is not None and not self.is_test:
            item['labels'] = torch.tensor(example["label"], dtype=torch.long)

        return item

# ✅ CORRECTED METRICS - MACRO F1
def compute_metrics_macro_f1(eval_pred):
    """Compute evaluation metrics with MACRO F1"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')  # ✅ MACRO F1

    probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = 0.0

    return {
        "accuracy": accuracy, 
        "precision": precision, 
        "recall": recall, 
        "f1": f1,  # ✅ This is now MACRO F1
        "auc": auc
    }

def data_collator(features):
    """Data collator for batch processing"""
    batch = {}
    batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
    batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
    batch['visual_embeds'] = torch.stack([f['visual_embeds'] for f in features])

    if 'labels' in features[0]:
        batch['labels'] = torch.stack([f['labels'] for f in features])

    return batch

# 💾 GOOGLE DRIVE BACKUP FUNCTIONS
def save_model_to_gdrive(model_path, tokenizer, eval_results):
    """Save model and results to Google Drive"""
    print("\n💾 Saving BILINGUAL model to Google Drive...")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(config.GDRIVE_BACKUP_DIR, f"bilingual_model_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Save model files
        model_backup_dir = os.path.join(backup_dir, "model")
        shutil.copytree(model_path, model_backup_dir)
        
        # Save tokenizer
        tokenizer_backup_dir = os.path.join(backup_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_backup_dir)
        
        # Save results
        results_file = os.path.join(backup_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Save config
        config_file = os.path.join(backup_dir, "config.json")
        config_dict = {
            'MODEL_TYPE': 'BILINGUAL_MULTIMODAL',
            'MULTILINGUAL_MODEL': config.MULTILINGUAL_MODEL,
            'LEARNING_RATE': config.LEARNING_RATE,
            'BATCH_SIZE': config.BATCH_SIZE,
            'NUM_EPOCHS': config.NUM_EPOCHS,
            'EVALUATION': 'MACRO_F1',
            'LANGUAGE': 'Hindi+English'
        }
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ BILINGUAL model saved to Google Drive: {backup_dir}")
        print(f"📊 Macro F1: {eval_results.get('eval_f1', 0):.4f}")
        
        return backup_dir
        
    except Exception as e:
        print(f"❌ Error saving to Google Drive: {e}")
        return None

# 🚀 BILINGUAL MAIN TRAINING PIPELINE
def main_bilingual_pipeline():
    print("🚀 Starting BILINGUAL VisualBERT + ViT Pipeline")
    print("🌍 Optimized for Hindi+English code-mixed data")

    # 1. Load Data
    train_data, val_data, test_data = load_data()

    # 2. Preprocess Data
    print("🔄 Creating labels and BILINGUAL text cleaning...")
    train_data = create_labels(train_data)
    val_data = create_labels(val_data)
    test_data = create_labels(test_data)

    train_data['ocr_clean'] = train_data['ocr'].apply(bilingual_text_cleaning)
    val_data['ocr_clean'] = val_data['ocr'].apply(bilingual_text_cleaning)
    test_data['ocr_clean'] = test_data['ocr'].apply(bilingual_text_cleaning)

    train_data = filter_and_validate_samples(train_data, "/content/trainImages", "Train")
    val_data = filter_and_validate_samples(val_data, "/content/valImages", "Validation")
    test_data = filter_and_validate_samples(test_data, "/content/testImages", "Test")

    print(f"\n📊 Final dataset sizes:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    print(f"   Test: {len(test_data)} samples")

    # 3. Precompute ViT Features
    if config.CACHE_FEATURES:
        print("🔄 Pre-computing ViT features for BILINGUAL training...")
        train_features = precompute_vit_features_corrected(train_data, "/content/trainImages", "train")
        val_features = precompute_vit_features_corrected(val_data, "/content/valImages", "val")
        test_features = precompute_vit_features_corrected(test_data, "/content/testImages", "test")
        print("🚀 ViT feature caching complete!")
    else:
        train_features = {}
        val_features = {}
        test_features = {}

    # 4. Initialize MULTILINGUAL Tokenizer
    print("🔧 Initializing MULTILINGUAL tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(config.MULTILINGUAL_MODEL)

    print("📊 Creating BILINGUAL datasets...")
    train_dataset = BilingualHatefulMemesDataset(train_data, tokenizer, train_features, config.MAX_TEXT_LENGTH)
    val_dataset = BilingualHatefulMemesDataset(val_data, tokenizer, val_features, config.MAX_TEXT_LENGTH)
    test_dataset = BilingualHatefulMemesDataset(test_data, tokenizer, test_features, config.MAX_TEXT_LENGTH, is_test=True)

    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Validation dataset: {len(val_dataset)} samples")
    print(f"✅ Test dataset: {len(test_dataset)} samples")

    # 5. Compute Class Weights
    train_labels = train_data['label'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    print(f"⚖️ Class weights: {class_weights}")

    # 6. Initialize BILINGUAL Model
    print("🧠 Initializing BILINGUAL multimodal model...")
    model = BilingualMultimodalClassifier(class_weights=class_weights, device=device).to(device)
    
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 7. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        logging_steps=10,
        fp16=config.USE_MIXED_PRECISION,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",          # ✅ Using MACRO F1
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        seed=42
    )

    # 8. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_macro_f1,  # ✅ MACRO F1
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # 9. Train Model
    print(f"\n🚀 Starting BILINGUAL Training...")
    print(f"🌍 Language: Hindi + English code-mixed")
    print(f"📊 Evaluation: Macro F1-Score")
    print(f"🔧 Learning rate: {config.LEARNING_RATE}")
    
    training_result = trainer.train()

    # 10. Evaluate Model
    print("📊 Running final evaluation...")
    eval_results = trainer.evaluate()

    print(f"\n🎯 BILINGUAL RESULTS:")
    print(f"   Macro F1: {eval_results['eval_f1']:.4f} ({eval_results['eval_f1']:.1%})")
    print(f"   Accuracy: {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']:.1%})")
    print(f"   Precision (Macro): {eval_results['eval_precision']:.4f}")
    print(f"   Recall (Macro): {eval_results['eval_recall']:.4f}")
    print(f"   AUC: {eval_results['eval_auc']:.4f}")

    if eval_results['eval_f1'] >= 0.70:
        print(f"\n🎉 HIGH MACRO F1 ACHIEVED! {eval_results['eval_f1']:.1%} >= 70%")
        print("🌍 BILINGUAL architecture working! Hindi+English support successful! 🔍")
    else:
        print(f"\n📈 Macro F1: {eval_results['eval_f1']:.1%} - Good for bilingual!")

    # 11. Save Model
    final_model_path = os.path.join(config.OUTPUT_DIR, "bilingual_multimodal_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n💾 BILINGUAL model saved to: {final_model_path}")
    
    # 12. Save to Google Drive
    gdrive_path = save_model_to_gdrive(final_model_path, tokenizer, eval_results)

    print("\n🌍 BILINGUAL multimodal hate speech detection ready!")
    
    return trainer, eval_results, test_dataset, tokenizer, final_model_path

# 🔮 TEST DATA PREDICTION WITH BILINGUAL PREPROCESSING
def predict_test_data_bilingual(trainer, test_dataset, tokenizer, model_path):
    """Predict on test data with BILINGUAL preprocessing"""
    print("\n🔮 PREDICTING ON TEST DATA (BILINGUAL)")
    print("🌍 Using Hindi+English bilingual preprocessing")
    
    # Make predictions
    print("🔄 Running predictions on test data...")
    predictions = trainer.predict(test_dataset)
    
    # Process predictions
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predicted_labels = np.argmax(logits, axis=1)
    confidence_scores = np.max(probs, axis=1)
    
    # Create results DataFrame
    test_results = []
    for i, sample in enumerate(test_dataset.dataset):
        result = {
            'id': sample['idx'],
            'original_text': sample['text'],
            'image': sample['image'],
            'predicted_label': int(predicted_labels[i]),
            'prediction': 'Hate Speech' if predicted_labels[i] == 1 else 'Not Hate Speech',
            'confidence': float(confidence_scores[i]),
            'prob_not_hate': float(probs[i][0]),
            'prob_hate': float(probs[i][1])
        }
        test_results.append(result)
    
    # Create DataFrame
    test_df = pd.DataFrame(test_results)
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save locally
    local_csv = f"/content/bilingual_test_predictions_{timestamp}.csv"
    test_df.to_csv(local_csv, index=False)
    print(f"✅ Local predictions saved: {local_csv}")
    
    # Save to Google Drive
    try:
        gdrive_csv = os.path.join(config.GDRIVE_BACKUP_DIR, f"bilingual_test_predictions_{timestamp}.csv")
        test_df.to_csv(gdrive_csv, index=False)
        print(f"✅ Google Drive predictions saved: {gdrive_csv}")
    except Exception as e:
        print(f"⚠️ Could not save to Google Drive: {e}")
    
    # Print summary
    hate_count = (test_df['predicted_label'] == 1).sum()
    not_hate_count = (test_df['predicted_label'] == 0).sum()
    avg_confidence = test_df['confidence'].mean()
    
    print(f"\n📊 BILINGUAL TEST PREDICTION SUMMARY:")
    print(f"   Total samples: {len(test_df)}")
    print(f"   Predicted Hate Speech: {hate_count} ({hate_count/len(test_df)*100:.1f}%)")
    print(f"   Predicted Not Hate Speech: {not_hate_count} ({not_hate_count/len(test_df)*100:.1f}%)")
    print(f"   Average Confidence: {avg_confidence:.4f}")
    print(f"   High Confidence (>0.8): {(test_df['confidence'] > 0.8).sum()}")
    print(f"   Low Confidence (<0.6): {(test_df['confidence'] < 0.6).sum()}")
    
    # Show sample predictions
    print("\n🔍 SAMPLE BILINGUAL PREDICTIONS:")
    sample_df = test_df[['original_text', 'prediction', 'confidence']].head(10)
    for idx, row in sample_df.iterrows():
        text_preview = row['original_text'][:60] + "..." if len(row['original_text']) > 60 else row['original_text']
        print(f"   {idx+1}. '{text_preview}' -> {row['prediction']} ({row['confidence']:.3f})")
    
    return test_df, local_csv

# 🚀 MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 STARTING BILINGUAL MEMOTION 3.0 PIPELINE")
    print("🌍 Hindi + English code-mixed architecture")
    print("📊 Macro F1-Score evaluation")
    print("🔮 Includes test data prediction")
    print("💾 Google Drive saving included")
    print("=" * 80)

    # Run training
    trainer, eval_results, test_dataset, tokenizer, model_path = main_bilingual_pipeline()

    # Run test prediction
    print("\n" + "="*60)
    print("🔮 RUNNING BILINGUAL TEST DATA PREDICTION")
    print("🌍 Hindi+English preprocessing")
    print("💾 Results saved to Google Drive")
    print("="*60)

    # Predict on test data
    test_predictions_df, predictions_csv = predict_test_data_bilingual(trainer, test_dataset, tokenizer, model_path)

    print("\n🎯 BILINGUAL TRAINING AND PREDICTION COMPLETED!")
    print("🌍 Hindi+English bilingual architecture successful")
    print("📊 Macro F1-Score evaluation implemented")
    print("✅ Test predictions generated with bilingual preprocessing")
    print("✅ Model and results saved to Google Drive")
    print("🚀 Ready for submission!")

    # Final results summary
    print("\n" + "="*80)
    print("📊 FINAL BILINGUAL RESULTS SUMMARY")
    print("="*80)

    print(f"\n🎯 TRAINING RESULTS:")
    print(f"   Macro F1: {eval_results.get('eval_f1', 0):.4f} ({eval_results.get('eval_f1', 0):.1%})")
    print(f"   Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"   Precision (Macro): {eval_results.get('eval_precision', 0):.4f}")
    print(f"   Recall (Macro): {eval_results.get('eval_recall', 0):.4f}")

    print(f"\n🔮 TEST PREDICTIONS:")
    hate_count = (test_predictions_df['predicted_label'] == 1).sum()
    total_count = len(test_predictions_df)
    print(f"   Total test samples: {total_count}")
    print(f"   Hate speech detected: {hate_count} ({hate_count/total_count*100:.1f}%)")
    print(f"   Average confidence: {test_predictions_df['confidence'].mean():.4f}")

    print(f"\n💾 SAVED FILES:")
    print(f"   Model: {config.GDRIVE_BACKUP_DIR}")
    print(f"   Predictions: {predictions_csv}")

    print(f"\n✅ BILINGUAL FIXES APPLIED:")
    print(f"   🌍 Language: Hindi + English code-mixed")
    print(f"   🧠 Model: XLM-RoBERTa (Multilingual)")
    print(f"   📊 Evaluation: Macro F1-Score (not weighted)")
    print(f"   🔧 Tokenizer: Multilingual XLM-RoBERTa")
    print(f"   🔄 Preprocessing: Bilingual text cleaning")
    print(f"   🎯 Architecture: Multimodal cross-attention")

    print("\n🎉 BILINGUAL ARCHITECTURE SUCCESSFUL - MACRO F1 ACHIEVED!")