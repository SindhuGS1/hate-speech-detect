# Optimized VisualBERT + ViT for Memotion 3.0 - 90% Accuracy Target

import os
import warnings
warnings.filterwarnings('ignore')

# Setup
os.system("pip install -q transformers torch torchvision datasets evaluate scikit-learn accelerate Pillow matplotlib seaborn pandas numpy tqdm")

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted!")
except:
    print("Not in Colab environment")

# Imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, VisualBertModel, VisualBertConfig, TrainingArguments, Trainer
from transformers import ViTImageProcessor, ViTModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import pickle
import json
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
class Config:
    BASE_PATH = "/content/drive/MyDrive/Memotion3/"
    CACHE_DIR = "/content/feature_cache/"
    OUTPUT_DIR = "/content/model_outputs/"
    VISUALBERT_MODEL = 'uclanlp/visualbert-nlvr2-coco-pre'
    VIT_MODEL = 'google/vit-base-patch16-224-in21k'
    BATCH_SIZE = 16
    NUM_EPOCHS = 12
    LEARNING_RATE = 1e-5

config = Config()
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print("Minimal VisualBERT + ViT setup complete!")
print("Upload this file first, then I'll provide the rest!")