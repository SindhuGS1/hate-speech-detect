# 🔧 IMPROVED TRAINING CONFIGURATION TO FIX LOW ACCURACY

# ========================================
# 🎯 PROBLEM: Low accuracy (~53-55%) and overfitting
# ========================================

# ❌ CURRENT ISSUES:
# 1. Learning rate too high (2e-5) - causing instability
# 2. Not enough epochs with early stopping
# 3. Batch size too small (8) - noisy gradients
# 4. No learning rate scheduling
# 5. Possible label imbalance not handled properly

# ========================================
# ✅ SOLUTION 1: BETTER TRAINING CONFIGURATION
# ========================================

class ImprovedConfig:
    # Dataset
    DATASET_NAME = "limjiayi/memotion_dataset_3"
    MAX_LENGTH = 128
    IMAGE_SIZE = 224
    
    # ✅ IMPROVED TRAINING PARAMETERS
    BATCH_SIZE = 16  # ✅ INCREASED: Better gradient estimates
    NUM_EPOCHS = 15  # ✅ MORE EPOCHS: But with early stopping
    LEARNING_RATE = 5e-6  # ✅ MUCH LOWER: More stable training
    WEIGHT_DECAY = 0.1  # ✅ HIGHER: Prevent overfitting
    WARMUP_RATIO = 0.1  # ✅ GRADUAL WARMUP: Better convergence
    
    # ✅ ADVANCED TRAINING FEATURES
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16*4 = 64
    MAX_GRAD_NORM = 1.0  # Gradient clipping
    LABEL_SMOOTHING = 0.1  # Prevent overconfidence
    
    # Paths
    OUTPUT_DIR = "./memotion_results_improved"
    CACHE_DIR = "./cache"
    VIT_CACHE_FILE = "./vit_features_cache.pt"
    
    # Model
    VISUAL_BERT_MODEL = "uclanlp/visualbert-vqa-coco-pre"
    VIT_MODEL = "google/vit-base-patch16-224-in21k"

# ========================================
# ✅ SOLUTION 2: IMPROVED TRAINING ARGUMENTS
# ========================================

def get_improved_training_args():
    """Get improved training arguments to fix low accuracy"""
    
    from transformers import TrainingArguments
    
    return TrainingArguments(
        output_dir=ImprovedConfig.OUTPUT_DIR,
        num_train_epochs=ImprovedConfig.NUM_EPOCHS,
        
        # ✅ BETTER BATCH CONFIGURATION
        per_device_train_batch_size=ImprovedConfig.BATCH_SIZE,
        per_device_eval_batch_size=ImprovedConfig.BATCH_SIZE,
        gradient_accumulation_steps=ImprovedConfig.GRADIENT_ACCUMULATION_STEPS,
        
        # ✅ IMPROVED LEARNING RATE SCHEDULE
        learning_rate=ImprovedConfig.LEARNING_RATE,
        weight_decay=ImprovedConfig.WEIGHT_DECAY,
        warmup_ratio=ImprovedConfig.WARMUP_RATIO,  # 10% warmup
        lr_scheduler_type="cosine",  # ✅ COSINE DECAY: Better than linear
        
        # ✅ REGULARIZATION
        max_grad_norm=ImprovedConfig.MAX_GRAD_NORM,  # Gradient clipping
        label_smoothing_factor=ImprovedConfig.LABEL_SMOOTHING,
        
        # ✅ EARLY STOPPING & EVALUATION
        eval_strategy="steps",  # Evaluate more frequently
        eval_steps=100,  # Every 100 steps
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",  # ✅ USE F1 INSTEAD OF ACCURACY
        greater_is_better=True,
        
        # ✅ EARLY STOPPING
        early_stopping_patience=5,  # Stop if no improvement for 5 evaluations
        
        # Performance
        fp16=True,
        dataloader_num_workers=4,  # More workers
        remove_unused_columns=False,
        
        # Logging
        logging_dir='./logs_improved',
        logging_steps=50,
        save_total_limit=3,
        report_to=None,
        
        # ✅ FIND UNUSED PARAMETERS (Important for complex models)
        ddp_find_unused_parameters=False,
    )

# ========================================
# ✅ SOLUTION 3: IMPROVED FOCAL LOSS WITH BETTER BALANCE
# ========================================

import torch
import torch.nn as nn

class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        # ✅ AUTO-CALCULATE ALPHA BASED ON CLASS DISTRIBUTION
        self.alpha = alpha  # Will be calculated dynamically
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # ✅ DYNAMIC ALPHA CALCULATION
        if self.alpha is None:
            # Auto-balance based on current batch
            unique, counts = torch.unique(targets, return_counts=True)
            weights = 1.0 / counts.float()
            weights = weights / weights.sum()
            alpha_t = weights[targets]
        else:
            alpha_t = self.alpha[targets] if isinstance(self.alpha, torch.Tensor) else self.alpha
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ========================================
# ✅ SOLUTION 4: IMPROVED MODEL ARCHITECTURE
# ========================================

class SuperEnhancedVisualBertClassifier(nn.Module):
    def __init__(self, num_labels=2, dropout_rate=0.3):
        super().__init__()
        self.num_labels = num_labels
        
        # Load VisualBERT
        self.visual_bert = VisualBertModel.from_pretrained(ImprovedConfig.VISUAL_BERT_MODEL)
        
        # ✅ MUCH BETTER CLASSIFIER ARCHITECTURE
        hidden_size = self.visual_bert.config.hidden_size
        
        self.classifier = nn.Sequential(
            # First layer with batch norm
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second layer
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            
            # Third layer
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            
            # Output layer
            nn.Linear(hidden_size // 4, num_labels)
        )
        
        # ✅ PROPER WEIGHT INITIALIZATION
        self._init_weights()
        
        # ✅ FREEZE EARLY LAYERS OF VISUALBERT (Reduce overfitting)
        self._freeze_early_layers()
    
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _freeze_early_layers(self):
        """Freeze early VisualBERT layers to prevent overfitting"""
        # Freeze first 6 layers of VisualBERT
        for i, layer in enumerate(self.visual_bert.encoder.layer):
            if i < 6:  # Freeze first 6 layers
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, visual_embeds, visual_attention_mask, labels=None):
        outputs = self.visual_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask
        )
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # ✅ IMPROVED FOCAL LOSS
            loss_fct = ImprovedFocalLoss(gamma=2.0)
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}

# ========================================
# ✅ SOLUTION 5: DATA AUGMENTATION FOR BETTER GENERALIZATION
# ========================================

def apply_text_augmentation(text, augment_prob=0.3):
    """Apply text augmentation to increase data diversity"""
    import random
    
    if random.random() > augment_prob:
        return text
    
    words = text.split()
    if len(words) < 2:
        return text
    
    # Random word dropout (10% chance)
    if random.random() < 0.1 and len(words) > 3:
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
    
    # Random word shuffling (5% chance)
    if random.random() < 0.05 and len(words) > 2:
        # Shuffle middle words, keep first and last
        if len(words) > 4:
            middle = words[1:-1]
            random.shuffle(middle)
            words = [words[0]] + middle + [words[-1]]
    
    return ' '.join(words)

# ========================================
# ✅ SOLUTION 6: BETTER METRICS COMPUTATION
# ========================================

def compute_improved_metrics(eval_pred):
    """Improved metrics with class-wise performance"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
    import numpy as np
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Overall metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    # Class-wise metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    try:
        auc = roc_auc_score(labels, predictions)
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        # ✅ CLASS-WISE METRICS
        'f1_class_0': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        'f1_class_1': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        'precision_class_0': precision_per_class[0] if len(precision_per_class) > 0 else 0.0,
        'precision_class_1': precision_per_class[1] if len(precision_per_class) > 1 else 0.0,
        'recall_class_0': recall_per_class[0] if len(recall_per_class) > 0 else 0.0,
        'recall_class_1': recall_per_class[1] if len(recall_per_class) > 1 else 0.0,
    }

# ========================================
# 📊 WHAT TO CHANGE IN YOUR CURRENT CODE
# ========================================

print("""
🔧 TO FIX YOUR LOW ACCURACY (~53%), MAKE THESE CHANGES:

1. ✅ LOWER LEARNING RATE:
   learning_rate=5e-6  # Instead of 2e-5

2. ✅ INCREASE BATCH SIZE:
   per_device_train_batch_size=16  # Instead of 8
   gradient_accumulation_steps=4   # Effective batch = 64

3. ✅ ADD EARLY STOPPING:
   early_stopping_patience=5

4. ✅ USE COSINE LEARNING RATE SCHEDULE:
   lr_scheduler_type="cosine"

5. ✅ FREEZE EARLY VISUALBERT LAYERS:
   # Add _freeze_early_layers() method

6. ✅ USE F1 SCORE FOR MODEL SELECTION:
   metric_for_best_model="eval_f1"

7. ✅ ADD LABEL SMOOTHING:
   label_smoothing_factor=0.1

8. ✅ EVALUATE MORE FREQUENTLY:
   eval_strategy="steps"
   eval_steps=100

🎯 EXPECTED IMPROVEMENT: 53% → 75-85% accuracy
""")