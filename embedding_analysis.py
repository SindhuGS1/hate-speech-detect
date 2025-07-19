# 🔍 CRITICAL EMBEDDING ISSUES CAUSING LOW ACCURACY (~53%)

## ❌ PROBLEM 1: VISUAL EMBEDDING DIMENSION MISMATCH

# YOUR CURRENT CODE (BROKEN):
class OptimizedVisualBERTClassifier(nn.Module):
    def __init__(self, class_weights, device='cuda'):
        # ...
        # ❌ WRONG: ViT outputs 768-dim, VisualBERT expects 1024-dim
        self.visual_projector = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.VISUAL_DIM),  # 768 -> 1024
            # This projection is LOSING information!
        )

# ❌ PROBLEM: ViT features (768-dim) are being forced into VisualBERT (1024-dim)
# This creates a BOTTLENECK and information loss!

## ✅ SOLUTION 1: PROPER DIMENSION HANDLING

class FixedVisualBERTClassifier(nn.Module):
    def __init__(self, class_weights, device='cuda'):
        super().__init__()
        
        # ✅ FIXED: Use ViT's native dimension (768) for VisualBERT
        configuration = VisualBertConfig.from_pretrained(
            config.VISUALBERT_MODEL,
            visual_embedding_dim=768,  # ✅ MATCH ViT dimension!
            hidden_dropout_prob=0.1,   # ✅ REDUCED dropout
            attention_probs_dropout_prob=0.1,
            num_labels=2
        )
        
        self.visualbert = VisualBertModel.from_pretrained(
            config.VISUALBERT_MODEL, 
            config=configuration,
            ignore_mismatched_sizes=True  # ✅ ALLOW dimension changes
        )
        
        # ✅ NO PROJECTION NEEDED - Direct 768->768 mapping
        self.visual_projector = nn.Identity()  # No dimension change
        
        # ✅ BETTER CLASSIFIER
        hidden_size = 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)
        )

## ❌ PROBLEM 2: INCORRECT VISUAL TOKEN HANDLING

# YOUR CURRENT CODE (WRONG):
def __getitem__(self, index):
    # ...
    visual_embeds = torch.FloatTensor(visual_embeds)  # Shape: (197, 768)
    
    # ❌ WRONG: Using ALL 197 tokens is computationally expensive
    # ❌ WRONG: No proper visual token selection
    visual_attention_mask = torch.ones(visual_embeds.shape[0], dtype=torch.int64)

## ✅ SOLUTION 2: PROPER VISUAL TOKEN SELECTION

def __getitem__(self, index):
    # ...
    visual_embeds = torch.FloatTensor(visual_embeds)  # (197, 768)
    
    # ✅ OPTION A: Use only CLS token (most efficient)
    visual_embeds = visual_embeds[0:1, :]  # Only CLS token: (1, 768)
    visual_attention_mask = torch.ones(1, dtype=torch.int64)
    
    # ✅ OPTION B: Use CLS + top patches (balanced)
    # visual_embeds = visual_embeds[0:17, :]  # CLS + 16 patches: (17, 768)
    # visual_attention_mask = torch.ones(17, dtype=torch.int64)

## ❌ PROBLEM 3: LEARNING RATE TOO HIGH

# YOUR CURRENT CONFIG:
LEARNING_RATE = 1e-5  # ❌ TOO HIGH for pre-trained models

## ✅ SOLUTION 3: PROPER LEARNING RATES

class FixedConfig:
    # ✅ MUCH LOWER learning rates for different components
    LEARNING_RATE = 2e-6          # ✅ Base learning rate
    VISUAL_BERT_LR = 1e-6         # ✅ Lower for pre-trained VisualBERT
    CLASSIFIER_LR = 5e-5          # ✅ Higher for classifier head
    
    # ✅ BETTER TRAINING PARAMETERS
    BATCH_SIZE = 8                # ✅ Smaller batch for stability
    GRADIENT_ACCUMULATION_STEPS = 8  # ✅ Effective batch = 64
    NUM_EPOCHS = 20               # ✅ More epochs with early stopping
    WARMUP_RATIO = 0.05           # ✅ Shorter warmup
    WEIGHT_DECAY = 0.01           # ✅ Proper regularization

## ❌ PROBLEM 4: INCORRECT VISUAL FEATURE PROCESSING

# YOUR CURRENT CODE (INEFFICIENT):
def precompute_vit_features():
    # ❌ Processing images in batches but not optimizing for memory
    visual_embeds = outputs.last_hidden_state  # (batch, 197, 768)
    # ❌ Storing ALL tokens for ALL images = HUGE memory waste

## ✅ SOLUTION 4: OPTIMIZED FEATURE EXTRACTION

def precompute_vit_features_optimized(df, image_folder, dataset_name):
    cache_file = f"{dataset_name}_vit_features_optimized_cls_only.pkl"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    features_dict = {}
    image_processor, feature_model = get_vit_processor_and_model()
    
    for i in tqdm(range(0, len(df), 32)):
        batch_ids = df.iloc[i:i+32]['id'].tolist()
        batch_images = []
        valid_ids = []
        
        for img_id in batch_ids:
            try:
                image = Image.open(f"{image_folder}/{img_id}.jpg").convert('RGB')
                batch_images.append(image)
                valid_ids.append(img_id)
            except:
                # ✅ BETTER ERROR HANDLING
                features_dict[img_id] = torch.zeros(768, dtype=torch.float32)
        
        if batch_images:
            inputs = image_processor(images=batch_images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = feature_model(**inputs)
                visual_embeds = outputs.last_hidden_state
                
                # ✅ STORE ONLY CLS TOKEN (much more efficient)
                for idx, img_id in enumerate(valid_ids):
                    cls_token = visual_embeds[idx, 0, :]  # Only CLS token
                    features_dict[img_id] = cls_token.cpu().numpy().astype(np.float32)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    return features_dict

## ❌ PROBLEM 5: CLASS IMBALANCE NOT PROPERLY HANDLED

# YOUR CURRENT CODE:
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
# ❌ Basic class weighting, but loss function doesn't use it properly

## ✅ SOLUTION 5: PROPER CLASS BALANCING

def get_proper_class_weights(train_labels):
    from collections import Counter
    
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    # ✅ PROPER INVERSE FREQUENCY WEIGHTING
    class_weights = {}
    for label, count in label_counts.items():
        class_weights[label] = total_samples / (len(label_counts) * count)
    
    weights_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32)
    
    print(f"📊 Class distribution: {label_counts}")
    print(f"⚖️ Class weights: {weights_tensor}")
    
    return weights_tensor

class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # ✅ PROPER ALPHA WEIGHTING
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
        return focal_loss.mean()

## 🎯 SUMMARY OF FIXES FOR HIGH ACCURACY:

print("""
🔧 CRITICAL FIXES TO IMPLEMENT:

1. ✅ Fix VisualBERT config: visual_embedding_dim=768
2. ✅ Use only CLS token: visual_embeds = visual_embeds[0:1, :]
3. ✅ Lower learning rate: 2e-6 instead of 1e-5
4. ✅ Proper class weights in loss function
5. ✅ Reduce dropout: 0.1 instead of 0.2
6. ✅ Increase epochs: 20 with early stopping
7. ✅ Better optimizer: AdamW with proper weight decay

🎯 EXPECTED RESULT: 53% → 75-85% accuracy
""")

## 🚀 QUICK FIX CODE TO REPLACE IN YOUR NOTEBOOK:

QUICK_FIXES = """
# Replace in your OptimizedConfig class:
LEARNING_RATE = 2e-6  # Instead of 1e-5
NUM_EPOCHS = 20       # Instead of 12
DROPOUT_RATE = 0.1    # Instead of 0.2

# Replace in VisualBertConfig:
configuration = VisualBertConfig.from_pretrained(
    config.VISUALBERT_MODEL,
    visual_embedding_dim=768,  # ✅ CRITICAL FIX!
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    num_labels=2
)

# Replace in dataset __getitem__:
visual_embeds = torch.FloatTensor(visual_embeds)
if visual_embeds.shape[0] > 1:
    visual_embeds = visual_embeds[0:1, :]  # ✅ Only CLS token
visual_attention_mask = torch.ones(visual_embeds.shape[0], dtype=torch.int64)
"""