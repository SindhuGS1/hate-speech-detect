#!/usr/bin/env python3
"""
Setup Environment for Memotion 3.0 Hate Speech Detection
Installs all required dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")

def main():
    print("🔧 Setting up environment for Memotion 3.0 Hate Speech Detection")
    print("=" * 60)
    
    packages = [
        "transformers==4.35.0",
        "torch>=2.0.0",
        "torchvision",
        "datasets",
        "evaluate", 
        "scikit-learn",
        "accelerate",
        "Pillow",
        "matplotlib",
        "seaborn", 
        "pandas",
        "numpy",
        "tqdm",
        "jupyter",
        "ipywidgets"
    ]
    
    print(f"📦 Installing {len(packages)} packages...")
    
    for package in packages:
        install_package(package)
    
    print("\n✅ Environment setup complete!")
    print("🚀 You can now run the hate speech detection pipeline")

if __name__ == "__main__":
    main()