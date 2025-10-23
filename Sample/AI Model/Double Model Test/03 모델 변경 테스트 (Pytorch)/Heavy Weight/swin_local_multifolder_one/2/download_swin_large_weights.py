"""
Swin Transformer Large Pretrained Weights Downloader

This script downloads ImageNet-22k pretrained weights for Swin Transformer Large
and saves them as 'swin_large.pth' for use in training recipes.

Swin Transformer Large Features:
- 200M+ parameters (~800MB)
- ImageNet-22k pretraining + ImageNet-1k fine-tuning
- 224x224 or 384x384 input resolution
- Hierarchical Transformer with shifted windows

Usage:
    python download_swin_large_weights.py
"""

import os
import sys
import torch

try:
    import timm
except ImportError:
    print("❌ Could not import timm library")
    print("Please install timm: pip install timm")
    sys.exit(1)

def download_swin_large_weights():
    """Download and save Swin Transformer Large pretrained weights"""
    
    print("🚀 Downloading Swin Transformer Large ImageNet-22k pretrained weights...")
    print("⚠️  This is a large model (~800MB) and may take a few minutes to download.")
    print("💡 Make sure you have sufficient disk space and a stable internet connection.")
    
    try:
        # Available Swin Large models
        model_variants = {
            'swin_large_patch4_window12_384_in22k': 'Swin-L (ImageNet-22k, 384x384)',
            'swin_large_patch4_window12_384': 'Swin-L (ImageNet-1k, 384x384)',
            'swin_large_patch4_window7_224_in22k': 'Swin-L (ImageNet-22k, 224x224)',
            'swin_large_patch4_window7_224': 'Swin-L (ImageNet-1k, 224x224)',
            'swinv2_large_window12to24_192to384_22kft1k': 'Swin-V2-L (22k→1k, 384x384)',
        }
        
        print("\n📋 Available Swin Large variants:")
        for i, (model_name, description) in enumerate(model_variants.items(), 1):
            print(f"{i}. {model_name}: {description}")
        
        # Use ImageNet-22k 384 version by default (best performance)
        model_name = 'swin_large_patch4_window12_384_in22k'
        print(f"\n🎯 Downloading: {model_name}")
        
        # Create model with pretrained weights
        print("📥 Loading model and downloading weights...")
        model = timm.create_model(model_name, pretrained=True)
        
        # Get current directory and create weights directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(current_dir, exist_ok=True)
        weight_path = os.path.join(current_dir, "swin_large.pth")
        
        # Save the entire model state dict
        print(f"💾 Saving weights to: {weight_path}")
        torch.save(model.state_dict(), weight_path)
        
        # Get file size in MB
        file_size = os.path.getsize(weight_path) / (1024 * 1024)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n✅ Successfully downloaded and saved Swin Transformer Large weights!")
        print(f"📁 File: swin_large.pth")
        print(f"📏 Size: {file_size:.1f} MB")
        print(f"🔢 Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"🎓 Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"🖼️  Input size: 384x384")
        print(f"🏆 Model: {model_name}")
        
        # Model architecture info
        print(f"\n📊 Model Architecture:")
        print(f"   - Family: Swin Transformer")
        print(f"   - Size: Large")
        print(f"   - Patch size: 4x4")
        print(f"   - Window size: 12x12")
        print(f"   - Pretraining: ImageNet-22k")
        print(f"   - Architecture: Hierarchical Transformer")
        
        # Usage example
        print(f"\n💡 Usage Example:")
        print(f"```python")
        print(f"import torch")
        print(f"import timm")
        print(f"")
        print(f"# Load model")
        print(f"model = timm.create_model('{model_name}', pretrained=False, num_classes=1000)")
        print(f"")
        print(f"# Load saved weights")
        print(f"state_dict = torch.load('swin_large.pth')")
        print(f"model.load_state_dict(state_dict)")
        print(f"")
        print(f"# For fine-tuning, reset classifier")
        print(f"model.reset_classifier(num_classes=YOUR_NUM_CLASSES)")
        print(f"")
        print(f"# Resize input if needed")
        print(f"# model.set_input_size((224, 224))  # For 224x224 input")
        print(f"```")
        
        print(f"\n🎉 You can now use this file for fine-tuning or inference!")
        
    except Exception as e:
        print(f"❌ Error downloading weights: {e}")
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~1GB)")
        print("3. Try updating timm: pip install --upgrade timm")
        print("4. Check if the model name is correct")
        sys.exit(1)

def list_available_swin_models():
    """List all available Swin Transformer models in timm"""
    print("🔍 Searching for all Swin Transformer models in timm...")
    
    try:
        # Get all Swin models
        swin_models = timm.list_models('swin*', pretrained=True)
        
        print(f"\n📋 Available Swin Transformer models with pretrained weights ({len(swin_models)} total):")
        
        # Group by version and size
        versions = [('swin_', 'Swin V1'), ('swinv2_', 'Swin V2')]
        sizes = ['tiny', 'small', 'base', 'large']
        
        for version_prefix, version_name in versions:
            version_models = [m for m in swin_models if m.startswith(version_prefix)]
            if version_models:
                print(f"\n🔸 {version_name}:")
                for size in sizes:
                    size_models = [m for m in version_models if size in m]
                    if size_models:
                        print(f"   📌 {size.upper()}:")
                        for model in sorted(size_models)[:5]:  # Show first 5 to avoid clutter
                            print(f"     - {model}")
                        if len(size_models) > 5:
                            print(f"     ... and {len(size_models) - 5} more")
    
    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_available_swin_models()
    else:
        download_swin_large_weights() 