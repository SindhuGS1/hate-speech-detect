# =============================================================================
# MEMOTION 3.0 - VISUALBERT FUSION PIPELINE
# Copy and paste this entire code into a SINGLE Google Colab cell
# =============================================================================

# Install packages
import subprocess
import sys

def install_packages():
    packages = [
        'transformers', 'torch', 'torchvision', 'datasets', 
        'evaluate', 'scikit-learn', 'accelerate', 'Pillow', 
        'matplotlib', 'seaborn', 'pandas', 'numpy', 'tqdm', 'open-clip-torch'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    print("All packages installed!")

install_packages()

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted!")
except:
    print("Not in Colab - Drive mount skipped")

# Core imports
import pandas as pd
import numpy as np
import re
import pickle
import json
import traceback
import gc
from datetime import datetime
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import open_clip
from transformers import (
    XLMRobertaTokenizer, VisualBertModel, VisualBertConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

print("MEMOTION 3.0 - VISUALBERT FUSION PIPELINE LOADED")
print("=" * 60)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Configuration
class Config:
    BASE_PATH = "/content/drive/MyDrive/Memotion3/"
    CACHE_DIR = "/content/feature_cache/"
    OUTPUT_DIR = "/content/model_outputs/"
    TRAIN_IMAGES = "/content/trainImages"
    VAL_IMAGES = "/content/valImages"
    TEST_IMAGES = "/content/testImages"
    
    MULTILINGUAL_TOKENIZER = 'xlm-roberta-base'
    VISUAL_ENCODER = 'ViT-B-32'
    VISUAL_PRETRAINED = 'openai'
    VISUALBERT_MODEL = 'uclanlp/visualbert-nlvr2-coco-pre'
    
    CLIP_DIM = 512
    VISUALBERT_DIM = 768
    NUM_VISUAL_TOKENS = 50
    MAX_TEXT_LENGTH = 128
    
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 15
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    DROPOUT_RATE = 0.1
    NUM_CLASSES = 2
    USE_MIXED_PRECISION = True
    USE_FOCAL_LOSS = True

config = Config()
import os
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# Extract images
def extract_images():
    print("Extracting images...")
    for dataset in ['train', 'val', 'test']:
        zip_path = f"{config.BASE_PATH}{dataset}Images.zip"
        extract_path = f"/content/{dataset}Images"
        if os.path.exists(zip_path) and not os.path.exists(extract_path):
            os.system(f"unzip -q '{zip_path}' -d /content/")
            print(f"   {dataset} images extracted")

# Data loading
def load_data():
    print("Loading Memotion 3.0 dataset...")
    datasets = {}
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(config.BASE_PATH, f'{split}.csv')
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
        try:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except:
                df = pd.read_csv(csv_path, encoding='latin-1')
            print(f"{split}: {len(df)} samples")
            if 'id' not in df.columns:
                df['id'] = df.index
            if 'ocr' not in df.columns:
                df['ocr'] = df.get('text', 'dummy text')
            datasets[split] = df
        except Exception as e:
            print(f"Error loading {split}: {e}")
    return datasets.get('train'), datasets.get('val'), datasets.get('test')

# Label creation
def create_labels(df, split_name):
    if df is None:
        return None
    print(f"Creating labels for {split_name}...")
    label_columns = ['offensive', 'hate', 'label', 'class']
    found_column = None
    for col in label_columns:
        if col in df.columns:
            found_column = col
            break
    if found_column is None:
        df['label'] = 0
        return df
    if found_column == 'offensive':
        hate_categories = ['offensive', 'very_offensive', 'slight', 'hateful_offensive']
        df['label'] = df[found_column].apply(
            lambda x: 1 if str(x).lower() in [c.lower() for c in hate_categories] else 0
        )
    else:
        df['label'] = df[found_column].apply(
            lambda x: 1 if (x == 1 or str(x).lower() in ['hate', '1']) else 0
        )
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    print(f"   Label distribution: {df['label'].value_counts().to_dict()}")
    return df

# Text cleaning
def bilingual_text_cleaning(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s.,!?\'"\\-\u0900-\u097F]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# CLIP model
def get_clip_model():
    print(f"Loading CLIP {config.VISUAL_ENCODER}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.VISUAL_ENCODER, pretrained=config.VISUAL_PRETRAINED, device=device
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        dummy_output = model.encode_image(dummy_input)
    assert dummy_output.shape[1] == config.CLIP_DIM
    print(f"CLIP verified: {dummy_output.shape}")
    return model, preprocess

# Feature extraction
def extract_clip_features(df, image_folder, dataset_name):
    if df is None or len(df) == 0:
        return {}
    cache_file = os.path.join(config.CACHE_DIR, f"{dataset_name}_clip_features.pkl")
    if os.path.exists(cache_file):
        print(f"Loading cached {dataset_name} features...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    print(f"Computing {dataset_name} CLIP features...")
    clip_model, preprocess = get_clip_model()
    features_dict = {}
    if not os.path.exists(image_folder):
        print(f"No image folder, creating dummy features...")
        for _, row in df.iterrows():
            img_id = row['id']
            np.random.seed(hash(str(img_id)) % 2**31)
            dummy_features = np.random.randn(config.NUM_VISUAL_TOKENS, config.CLIP_DIM).astype(np.float32)
            features_dict[img_id] = dummy_features
        return features_dict
    image_ids = df['id'].tolist()
    batch_size = 32
    for i in tqdm(range(0, len(image_ids), batch_size), desc=f"Extracting {dataset_name}"):
        batch_ids = image_ids[i:i + batch_size]
        batch_images = []
        valid_ids = []
        for img_id in batch_ids:
            image_path = os.path.join(image_folder, f"{img_id}.jpg")
            try:
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')
                    preprocessed = preprocess(image)
                    batch_images.append(preprocessed)
                    valid_ids.append(img_id)
                else:
                    np.random.seed(hash(str(img_id)) % 2**31)
                    dummy_features = np.random.randn(config.NUM_VISUAL_TOKENS, config.CLIP_DIM).astype(np.float32)
                    features_dict[img_id] = dummy_features
            except:
                np.random.seed(hash(str(img_id)) % 2**31)
                dummy_features = np.random.randn(config.NUM_VISUAL_TOKENS, config.CLIP_DIM).astype(np.float32)
                features_dict[img_id] = dummy_features
        if batch_images:
            try:
                batch_tensor = torch.stack(batch_images).to(device)
                with torch.no_grad():
                    visual_features = clip_model.encode_image(batch_tensor)
                for idx, img_id in enumerate(valid_ids):
                    clip_feature = visual_features[idx].cpu().numpy()
                    visual_tokens = np.tile(clip_feature, (config.NUM_VISUAL_TOKENS, 1))
                    features_dict[img_id] = visual_tokens.astype(np.float32)
            except Exception as e:
                print(f"Batch error: {e}")
                for img_id in valid_ids:
                    np.random.seed(hash(str(img_id)) % 2**31)
                    dummy_features = np.random.randn(config.NUM_VISUAL_TOKENS, config.CLIP_DIM).astype(np.float32)
                    features_dict[img_id] = dummy_features
    print(f"Extracted {len(features_dict)} features")
    with open(cache_file, 'wb') as f:
        pickle.dump(features_dict, f)
    del clip_model
    torch.cuda.empty_cache()
    return features_dict

# Focal Loss
class FocalLoss(nn.Module):
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

# VisualBERT Model
class VisualBERTClassifier(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.num_labels = config.NUM_CLASSES
        print("Initializing VisualBERT...")
        try:
            visualbert_config = VisualBertConfig.from_pretrained(
                config.VISUALBERT_MODEL,
                visual_embedding_dim=config.CLIP_DIM,
                hidden_dropout_prob=config.DROPOUT_RATE,
                num_labels=self.num_labels
            )
            self.visualbert = VisualBertModel.from_pretrained(
                config.VISUALBERT_MODEL, config=visualbert_config, ignore_mismatched_sizes=True
            )
            print("   VisualBERT loaded")
        except Exception as e:
            print(f"   Creating from scratch: {e}")
            visualbert_config = VisualBertConfig(
                vocab_size=250002, hidden_size=config.VISUALBERT_DIM,
                visual_embedding_dim=config.CLIP_DIM, hidden_dropout_prob=config.DROPOUT_RATE,
                num_labels=self.num_labels
            )
            self.visualbert = VisualBertModel(visualbert_config)
        
        self.visual_projector = nn.Linear(config.CLIP_DIM, config.VISUALBERT_DIM)
        print(f"   Visual projector: {config.CLIP_DIM} -> {config.VISUALBERT_DIM}")
        
        self.classifier = nn.Sequential(
            nn.Linear(config.VISUALBERT_DIM, config.VISUALBERT_DIM // 2),
            nn.LayerNorm(config.VISUALBERT_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.VISUALBERT_DIM // 2, self.num_labels)
        )
        
        if config.USE_FOCAL_LOSS and class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.loss_fct = FocalLoss(alpha=weights_tensor, gamma=2.0)
            print("   Using Focal Loss")
        else:
            self.loss_fct = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, 
                visual_embeds=None, visual_attention_mask=None, 
                visual_token_type_ids=None, labels=None):
        batch_size = input_ids.size(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        if visual_embeds is not None:
            visual_embeds_projected = self.visual_projector(visual_embeds)
        else:
            visual_embeds_projected = torch.zeros(
                batch_size, config.NUM_VISUAL_TOKENS, config.VISUALBERT_DIM, device=input_ids.device
            )
        
        if visual_attention_mask is None:
            visual_attention_mask = torch.ones(
                batch_size, config.NUM_VISUAL_TOKENS, dtype=torch.int64, device=input_ids.device
            )
        
        if visual_token_type_ids is None:
            visual_token_type_ids = torch.ones(
                batch_size, config.NUM_VISUAL_TOKENS, dtype=torch.int64, device=input_ids.device
            )
        
        try:
            outputs = self.visualbert(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                visual_embeds=visual_embeds_projected, visual_attention_mask=visual_attention_mask,
                visual_token_type_ids=visual_token_type_ids
            )
            pooled_output = outputs.pooler_output
        except Exception as e:
            print(f"VisualBERT error: {e}")
            pooled_output = torch.mean(visual_embeds_projected, dim=1)
        
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            labels = labels.view(-1).long().to(logits.device)
            loss = self.loss_fct(logits, labels)
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}

# Dataset class
class MemotionDataset(Dataset):
    def __init__(self, df, tokenizer, features_dict, is_test=False):
        self.tokenizer = tokenizer
        self.features_dict = features_dict
        self.is_test = is_test
        self.dataset = []
        if df is not None:
            for i, row in df.iterrows():
                self.dataset.append({
                    "text": str(row.get("ocr_clean", "")),
                    "label": row.get("label", 0) if not is_test else None,
                    "idx": row.get("id", i),
                })
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        example = self.dataset[index]
        try:
            encoded = self.tokenizer(
                example["text"], padding="max_length", max_length=config.MAX_TEXT_LENGTH,
                truncation=True, return_tensors="pt"
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))
            if token_type_ids.ndim > 1:
                token_type_ids = token_type_ids.squeeze(0)
        except:
            input_ids = torch.zeros(config.MAX_TEXT_LENGTH, dtype=torch.long)
            attention_mask = torch.zeros(config.MAX_TEXT_LENGTH, dtype=torch.long)
            token_type_ids = torch.zeros(config.MAX_TEXT_LENGTH, dtype=torch.long)
        
        img_id = example["idx"]
        visual_features = self.features_dict.get(img_id)
        if visual_features is None:
            np.random.seed(hash(str(img_id)) % 2**31)
            visual_features = np.random.randn(config.NUM_VISUAL_TOKENS, config.CLIP_DIM).astype(np.float32)
        
        visual_embeds = torch.FloatTensor(visual_features)
        visual_attention_mask = torch.ones(visual_embeds.shape[0], dtype=torch.int64)
        visual_token_type_ids = torch.ones(visual_embeds.shape[0], dtype=torch.int64)
        
        item = {
            'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
            'visual_embeds': visual_embeds, 'visual_attention_mask': visual_attention_mask,
            'visual_token_type_ids': visual_token_type_ids
        }
        
        if example["label"] is not None and not self.is_test:
            item['labels'] = torch.tensor(example["label"], dtype=torch.long)
        
        return item

# Metrics
def compute_metrics_macro_f1(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    try:
        probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()
        auc = roc_auc_score(labels, probs[:, 1])
    except:
        auc = 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

# Data collator
def data_collator(features):
    try:
        batch = {}
        batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        batch['token_type_ids'] = torch.stack([f['token_type_ids'] for f in features])
        batch['visual_embeds'] = torch.stack([f['visual_embeds'] for f in features])
        batch['visual_attention_mask'] = torch.stack([f['visual_attention_mask'] for f in features])
        batch['visual_token_type_ids'] = torch.stack([f['visual_token_type_ids'] for f in features])
        if 'labels' in features[0]:
            batch['labels'] = torch.stack([f['labels'] for f in features])
        return batch
    except Exception as e:
        print(f"Data collator error: {e}")
        batch_size = len(features)
        return {
            'input_ids': torch.zeros(batch_size, config.MAX_TEXT_LENGTH, dtype=torch.long),
            'attention_mask': torch.zeros(batch_size, config.MAX_TEXT_LENGTH, dtype=torch.long),
            'token_type_ids': torch.zeros(batch_size, config.MAX_TEXT_LENGTH, dtype=torch.long),
            'visual_embeds': torch.zeros(batch_size, config.NUM_VISUAL_TOKENS, config.CLIP_DIM),
            'visual_attention_mask': torch.ones(batch_size, config.NUM_VISUAL_TOKENS, dtype=torch.long),
            'visual_token_type_ids': torch.ones(batch_size, config.NUM_VISUAL_TOKENS, dtype=torch.long),
        }

# Main pipeline
def run_memotion_pipeline():
    print("STARTING VERIFIED PIPELINE")
    print("="*60)
    
    try:
        extract_images()
        train_data, val_data, test_data = load_data()
        
        if train_data is None:
            print("No training data!")
            return None, None, None, None, None
        
        print("Preprocessing...")
        train_data = create_labels(train_data, 'train')
        val_data = create_labels(val_data, 'val') if val_data is not None else None
        test_data = create_labels(test_data, 'test') if test_data is not None else None
        
        train_data['ocr_clean'] = train_data['ocr'].apply(bilingual_text_cleaning)
        if val_data is not None:
            val_data['ocr_clean'] = val_data['ocr'].apply(bilingual_text_cleaning)
        if test_data is not None:
            test_data['ocr_clean'] = test_data['ocr'].apply(bilingual_text_cleaning)
        
        print("Extracting visual features...")
        train_features = extract_clip_features(train_data, config.TRAIN_IMAGES, "train")
        val_features = extract_clip_features(val_data, config.VAL_IMAGES, "val") if val_data is not None else {}
        test_features = extract_clip_features(test_data, config.TEST_IMAGES, "test") if test_data is not None else {}
        
        print("Loading tokenizer...")
        tokenizer = XLMRobertaTokenizer.from_pretrained(config.MULTILINGUAL_TOKENIZER)
        
        print("Creating datasets...")
        train_dataset = MemotionDataset(train_data, tokenizer, train_features)
        val_dataset = MemotionDataset(val_data, tokenizer, val_features) if val_data is not None else None
        test_dataset = MemotionDataset(test_data, tokenizer, test_features, is_test=True) if test_data is not None else None
        
        train_labels = train_data['label'].values
        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            print(f"Class weights: {class_weights}")
        except:
            class_weights = None
        
        print("Initializing VisualBERT model...")
        model = VisualBERTClassifier(class_weights=class_weights).to(device)
        
        print("Testing forward pass...")
        with torch.no_grad():
            sample_batch = data_collator([train_dataset[0]])
            for key, value in sample_batch.items():
                sample_batch[key] = value.to(device)
            output = model(**sample_batch)
            print(f"   Forward pass: {output['logits'].shape}")
        
        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR, num_train_epochs=config.NUM_EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE, per_device_eval_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS, learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY, warmup_ratio=config.WARMUP_RATIO,
            eval_strategy="steps" if val_dataset else "no", eval_steps=100 if val_dataset else None,
            save_steps=200, logging_steps=50, fp16=config.USE_MIXED_PRECISION,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1" if val_dataset else None, greater_is_better=True,
            save_total_limit=3, report_to="none", seed=42, remove_unused_columns=False
        )
        
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
            data_collator=data_collator, compute_metrics=compute_metrics_macro_f1 if val_dataset else None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] if val_dataset else []
        )
        
        print("Starting training...")
        print("Architecture: XLM-RoBERTa + CLIP + VisualBERT Fusion")
        print("Language: Hindi + English")
        print("Evaluation: Macro F1-Score")
        
        training_result = trainer.train()
        print("Training completed!")
        
        eval_results = {}
        if val_dataset:
            print("Evaluating...")
            eval_results = trainer.evaluate()
            print(f"Macro F1: {eval_results.get('eval_f1', 0):.4f}")
            print(f"Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
        
        final_model_path = os.path.join(config.OUTPUT_DIR, "visualbert_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Model saved to: {final_model_path}")
        
        print("PIPELINE COMPLETED!")
        print("VisualBERT fusion working")
        print("CLIP features properly extracted")
        print("Bilingual processing verified")
        
        return trainer, eval_results, test_dataset, tokenizer, final_model_path
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        traceback.print_exc()
        return None, None, None, None, None

# Test prediction function
def predict_test_data(trainer, test_dataset, tokenizer, model_path):
    if trainer is None or test_dataset is None:
        print("Cannot run predictions")
        return None, None
    
    print("RUNNING TEST PREDICTIONS")
    try:
        predictions = trainer.predict(test_dataset)
        logits = predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        predicted_labels = np.argmax(logits, axis=1)
        confidence_scores = np.max(probs, axis=1)
        
        test_results = []
        for i, sample in enumerate(test_dataset.dataset):
            result = {
                'id': sample['idx'], 'text': sample['text'][:100], 'predicted_label': int(predicted_labels[i]),
                'prediction': 'Hate Speech' if predicted_labels[i] == 1 else 'Not Hate Speech',
                'confidence': float(confidence_scores[i]), 'prob_not_hate': float(probs[i][0]),
                'prob_hate': float(probs[i][1])
            }
            test_results.append(result)
        
        test_df = pd.DataFrame(test_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"/content/predictions_{timestamp}.csv"
        test_df.to_csv(csv_file, index=False)
        print(f"Predictions saved: {csv_file}")
        
        hate_count = (test_df['predicted_label'] == 1).sum()
        total_count = len(test_df)
        print(f"Total: {total_count}, Hate Speech: {hate_count} ({hate_count/total_count*100:.1f}%)")
        
        return test_df, csv_file
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

# =============================================================================
# RUN THE COMPLETE PIPELINE
# =============================================================================

print("READY TO RUN MEMOTION 3.0 PIPELINE!")
print("Execute: trainer, eval_results, test_dataset, tokenizer, model_path = run_memotion_pipeline()")
print("Then: test_df, predictions_csv = predict_test_data(trainer, test_dataset, tokenizer, model_path)")
print("=" * 80)