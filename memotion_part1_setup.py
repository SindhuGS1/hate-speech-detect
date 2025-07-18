"""
🚀 OPTIMIZED Memotion 3.0 - VisualBERT + ViT Architecture - PART 1
🎯 Target: 90% Accuracy | ⚡ Feature Caching | 🛡️ Error-Free
"""

print("🚀 OPTIMIZED VisualBERT + ViT Memotion Detection")
print("🎯 Target: 90% Accuracy with YOUR original architecture")
print("=" * 60)

# ============================================================================
# 📦 SETUP & INSTALLATION (Colab Ready)
# ============================================================================

import os
import warnings
warnings.filterwarnings('ignore')

# Install packages
print("📦 Installing packages...")
os.system("pip install -q transformers==4.36.0 torch torchvision datasets evaluate scikit-learn accelerate Pillow matplotlib seaborn pandas numpy tqdm")

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

# ============================================================================
# 📚 IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import json
from pathlib import Path
from tqdm.auto import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Image processing - FIXED deprecated warnings
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Updated ViT imports (FIXED your deprecation warnings)
try:
    from transformers import ViTImageProcessor, ViTModel
    print("✅ Using updated ViT imports")
except ImportError:
    from transformers import ViTFeatureExtractor as ViTImageProcessor, ViTModel
    print("⚠️ Using legacy ViT imports")

# VisualBERT imports (KEEPING your original architecture)
from transformers import (
    BertTokenizer, VisualBertModel, VisualBertConfig,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup
)

# Evaluation
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

# Enable optimizations
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

print("✅ Part 1 (Setup & Imports) Complete!")
print("📋 Next: Run Part 2 (Configuration & Data Processing)")