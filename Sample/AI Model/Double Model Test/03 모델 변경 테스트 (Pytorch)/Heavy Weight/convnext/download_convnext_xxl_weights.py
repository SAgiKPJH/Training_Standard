"""
ConvNeXt-XXL Pretrained Weights Downloader

This script downloads ImageNet-22k pretrained weights for ConvNeXt-XXL
and saves them as 'convnext_xxl.pth' for use in training recipes.

ConvNeXt-XXL Features:
- 800M+ parameters (~3.2GB)
- ImageNet-22k pretraining + ImageNet-1k fine-tuning
- 384x384 input resolution
- State-of-the-art performance

Usage:
    python download_convnext_xxl_weights.py
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

def download_convnext_xxl_weights():
    """Download and save ConvNeXt-XXL pretrained weights"""
    
    print("🚀 Downloading ConvNeXt-XXL ImageNet-22k pretrained weights...")
    print("⚠️  This is a large model (~3.2GB) and may take several minutes to download.")
    print("💡 Make sure you have sufficient disk space and a stable internet connection.")
    
    try:
        # Available ConvNeXt-XXL models
        model_variants = {
            'convnext_`large_384_in22ft1k': 'ConvNeXt-XXL (ImageNet-22k→1k, 384x384)',
            'convnext_large_384_in22k': 'ConvNeXt-XXL (ImageNet-22k only, 384x384)',
        }
        
        print("\n📋 Available ConvNeXt-XXL variants:")
        for i, (model_name, description) in enumerate(model_variants.items(), 1):
            print(f"{i}. {model_name}: {description}")
        
        # Use the fine-tuned version by default (best for downstream tasks)
        model_name = 'convnext_large_384_in22ft1k'
        print(f"\n🎯 Downloading: {model_name}")
        
        # Create model with pretrained weights
        print("📥 Loading model and downloading weights...")
        model = timm.create_model(model_name, pretrained=True)
        
        # Get current directory and create weights directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(current_dir, exist_ok=True)
        weight_path = os.path.join(current_dir, "convnext_xxl.pth")
        
        # Save the entire model state dict
        print(f"💾 Saving weights to: {weight_path}")
        torch.save(model.state_dict(), weight_path)
        
        # Get file size in MB
        file_size = os.path.getsize(weight_path) / (1024 * 1024)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n✅ Successfully downloaded and saved ConvNeXt-XXL weights!")
        print(f"📁 File: convnext_xxl.pth")
        print(f"📏 Size: {file_size:.1f} MB")
        print(f"🔢 Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"🎓 Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"🖼️  Input size: 384x384")
        print(f"🏆 Model: {model_name}")
        
        # Model architecture info
        print(f"\n📊 Model Architecture:")
        print(f"   - Family: ConvNeXt")
        print(f"   - Size: XXL (Extra Extra Large)")
        print(f"   - Pretraining: ImageNet-22k")
        print(f"   - Fine-tuning: ImageNet-1k")
        print(f"   - Resolution: 384x384")
        
        # Usage example
        print(f"\n💡 Usage Example:")
        print(f"```python")
        print(f"import torch")
        print(f"import timm")
        print(f"")
        print(f"# Load model")
        print(f"model = timm.create_model('{model_name}', pretrained=False)")
        print(f"")
        print(f"# Load saved weights")
        print(f"state_dict = torch.load('convnext_xxl.pth')")
        print(f"model.load_state_dict(state_dict)")
        print(f"```")
        
        print(f"\n🎉 You can now use this file for fine-tuning or inference!")
        
    except Exception as e:
        print(f"❌ Error downloading weights: {e}")
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~4GB)")
        print("3. Try updating timm: pip install --upgrade timm")
        print("4. Check if the model name is correct")
        sys.exit(1)

def list_available_convnext_models():
    """List all available ConvNeXt models in timm"""
    print("🔍 Searching for all ConvNeXt models in timm...")
    
    try:
        # Get all ConvNeXt models
        convnext_models = timm.list_models('convnext*', pretrained=True)
        
        print(f"\n📋 Available ConvNeXt models with pretrained weights ({len(convnext_models)} total):")
        
        # Group by size
        sizes = ['tiny', 'small', 'base', 'large', 'xlarge', 'xxlarge']
        for size in sizes:
            size_models = [m for m in convnext_models if size in m]
            if size_models:
                print(f"\n🔸 {size.upper()}:")
                for model in sorted(size_models):
                    print(f"   - {model}")
    
    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_available_convnext_models()
    else:
        download_convnext_xxl_weights() 