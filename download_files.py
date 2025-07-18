#!/usr/bin/env python3
"""
📥 Download All Memotion Project Files
Run this script to download all project files as a zip
"""

import zipfile
import os
from pathlib import Path

def create_project_zip():
    """Create a zip file with all project files"""
    
    # Files to include in the zip
    files_to_zip = [
        "memotion_colab_ready.py",
        "memotion_hate_speech_detection.py", 
        "inference_demo.py",
        "setup_environment.py",
        "README.md",
        "requirements.txt",
        "enhanced_memotion.py"
    ]
    
    zip_filename = "Memotion_Optimized_Project_Complete.zip"
    
    print("📦 Creating project zip file...")
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for filename in files_to_zip:
            if os.path.exists(filename):
                zipf.write(filename)
                print(f"✅ Added {filename}")
            else:
                print(f"⚠️ {filename} not found")
    
    print(f"\n🎉 Created {zip_filename}")
    print(f"📁 File size: {os.path.getsize(zip_filename) / 1024:.1f} KB")
    
    return zip_filename

if __name__ == "__main__":
    zip_file = create_project_zip()
    print(f"\n📥 Download ready: {zip_file}")