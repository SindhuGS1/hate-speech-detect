#!/usr/bin/env python3
"""
Inference Demo for Hate Speech Detection
Demonstrates how to use the trained model for predictions on new samples
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import re
import json
import os

# Import our model classes (assuming they're in the main script)
try:
    from memotion_hate_speech_detection import (
        ResNetFeatureExtractor, 
        MultiModalHateSpeechClassifier,
        clean_ocr_text
    )
except ImportError:
    print("⚠️ Please ensure memotion_hate_speech_detection.py is in the same directory")
    exit(1)

class HateSpeechPredictor:
    """Easy-to-use predictor for hate speech detection in memes"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize the predictor with trained model
        
        Args:
            model_path (str): Path to the saved model directory
            device (str): Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load configuration
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            print("⚠️ Config file not found. Using default configuration.")
            self.config = {
                'num_classes': 2,
                'max_length': 128,
                'class_names': ['Non-Hate', 'Hate']
            }
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded")
        
        # Initialize ResNet feature extractor
        self.resnet_extractor = ResNetFeatureExtractor('resnet50').to(self.device)
        self.resnet_extractor.eval()
        print("✅ ResNet feature extractor loaded")
        
        # Load trained model
        self.model = MultiModalHateSpeechClassifier(
            num_classes=self.config['num_classes'],
            class_weights=None,  # No weights needed for inference
            dropout_rate=0.3
        ).to(self.device)
        
        # Load model weights
        model_weights_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(model_weights_path):
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            print("✅ Model weights loaded")
        else:
            print("⚠️ Model weights not found. Please ensure the model is properly saved.")
        
        self.model.eval()
        
        # Image preprocessing
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("🔍 Hate Speech Predictor ready!")
    
    def predict_single(self, image_path, text):
        """
        Predict hate speech for a single meme
        
        Args:
            image_path (str): Path to the meme image
            text (str): OCR text from the meme
            
        Returns:
            dict: Prediction results
        """
        self.model.eval()
        self.resnet_extractor.eval()
        
        with torch.no_grad():
            # Process text
            text_clean = clean_ocr_text(text)
            encoding = self.tokenizer(
                text_clean,
                padding='max_length',
                max_length=self.config['max_length'],
                truncation=True,
                return_tensors='pt'
            )
            
            # Process image
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.image_transforms(image).unsqueeze(0).to(self.device)
                visual_features = self.resnet_extractor(image_tensor).cpu().squeeze(0)
            except Exception as e:
                print(f"⚠️ Error processing image: {e}")
                visual_features = torch.zeros(768)
            
            # Prepare input tensors
            input_data = {
                'input_ids': encoding['input_ids'].to(self.device),
                'attention_mask': encoding['attention_mask'].to(self.device),
                'token_type_ids': encoding.get('token_type_ids', 
                                             torch.zeros_like(encoding['input_ids'])).to(self.device),
                'visual_embeds': visual_features.unsqueeze(0).to(self.device)
            }
            
            # Get prediction
            outputs = self.model(**input_data)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
            
            # Extract results
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = torch.max(probabilities).item()
            hate_probability = probabilities[0, 1].item()
            
            return {
                'predicted_class': predicted_class,
                'predicted_label': self.config['class_names'][predicted_class],
                'confidence': confidence,
                'hate_probability': hate_probability,
                'non_hate_probability': probabilities[0, 0].item(),
                'processed_text': text_clean,
                'text_length': len(text_clean),
                'risk_level': self._get_risk_level(hate_probability)
            }
    
    def predict_batch(self, samples):
        """
        Predict hate speech for multiple samples
        
        Args:
            samples (list): List of dicts with 'image_path' and 'text' keys
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        print(f"🔄 Processing {len(samples)} samples...")
        
        for i, sample in enumerate(samples):
            try:
                result = self.predict_single(sample['image_path'], sample['text'])
                result['sample_id'] = i
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(samples)} samples")
                    
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                results.append({
                    'sample_id': i,
                    'predicted_class': -1,
                    'predicted_label': 'ERROR',
                    'confidence': 0.0,
                    'hate_probability': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def _get_risk_level(self, hate_probability):
        """Get risk level based on hate probability"""
        if hate_probability < 0.3:
            return "LOW"
        elif hate_probability < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def analyze_results(self, results):
        """Analyze batch prediction results"""
        valid_results = [r for r in results if r['predicted_class'] != -1]
        
        if not valid_results:
            print("⚠️ No valid predictions to analyze")
            return
        
        hate_count = sum(1 for r in valid_results if r['predicted_class'] == 1)
        non_hate_count = len(valid_results) - hate_count
        
        avg_confidence = np.mean([r['confidence'] for r in valid_results])
        avg_hate_prob = np.mean([r['hate_probability'] for r in valid_results])
        
        risk_levels = [r['risk_level'] for r in valid_results]
        risk_counts = {level: risk_levels.count(level) for level in ['LOW', 'MEDIUM', 'HIGH']}
        
        print("\n📊 Batch Analysis Results:")
        print("=" * 40)
        print(f"📈 Predictions:")
        print(f"   Non-Hate: {non_hate_count} ({non_hate_count/len(valid_results)*100:.1f}%)")
        print(f"   Hate: {hate_count} ({hate_count/len(valid_results)*100:.1f}%)")
        
        print(f"\n📊 Confidence & Risk:")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Average Hate Probability: {avg_hate_prob:.3f}")
        print(f"   Risk Distribution:")
        for level, count in risk_counts.items():
            print(f"     {level}: {count} ({count/len(valid_results)*100:.1f}%)")

def demo():
    """Demonstration of the hate speech predictor"""
    
    print("🔍🛡️ Hate Speech Detection - Inference Demo")
    print("=" * 50)
    
    # Configuration
    MODEL_PATH = "/content/model_outputs/final_model"  # Adjust this path
    SAMPLE_IMAGES_PATH = "/content/memotion_data/testImages"  # Adjust this path
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model not found at {MODEL_PATH}")
        print("💡 Please train the model first or adjust the MODEL_PATH")
        return
    
    # Initialize predictor
    try:
        predictor = HateSpeechPredictor(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return
    
    # Example 1: Single prediction
    print("\n🧪 Example 1: Single Prediction")
    print("-" * 30)
    
    sample_image = os.path.join(SAMPLE_IMAGES_PATH, "0.jpg")
    sample_text = "This is a sample meme text for demonstration"
    
    if os.path.exists(sample_image):
        result = predictor.predict_single(sample_image, sample_text)
        
        print(f"📄 Input:")
        print(f"   Text: {sample_text}")
        print(f"   Image: {sample_image}")
        
        print(f"\n🎯 Prediction:")
        print(f"   Label: {result['predicted_label']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Hate Probability: {result['hate_probability']:.3f}")
        print(f"   Risk Level: {result['risk_level']}")
    else:
        print(f"⚠️ Sample image not found: {sample_image}")
    
    # Example 2: Batch prediction
    print("\n🧪 Example 2: Batch Prediction")
    print("-" * 30)
    
    # Create sample batch (you can modify this)
    sample_batch = []
    for i in range(min(5, len(os.listdir(SAMPLE_IMAGES_PATH)) if os.path.exists(SAMPLE_IMAGES_PATH) else 0)):
        sample_batch.append({
            'image_path': os.path.join(SAMPLE_IMAGES_PATH, f"{i}.jpg"),
            'text': f"Sample meme text {i} for batch demonstration"
        })
    
    if sample_batch:
        batch_results = predictor.predict_batch(sample_batch)
        
        print(f"\n📋 Individual Results:")
        for result in batch_results[:3]:  # Show first 3 results
            if result['predicted_class'] != -1:
                print(f"   Sample {result['sample_id']}: {result['predicted_label']} "
                      f"(confidence: {result['confidence']:.3f}, risk: {result['risk_level']})")
        
        predictor.analyze_results(batch_results)
    else:
        print("⚠️ No sample images found for batch prediction")
    
    print("\n✅ Demo completed!")
    print("💡 You can now use the predictor for your own memes!")

if __name__ == "__main__":
    demo()