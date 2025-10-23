"""
Vision Transformer Huge (ViT-Huge) Pretrained Weights Downloader

This script downloads ImageNet-21k pretrained weights for ViT-Huge
and saves them as 'vit_huge.pth' for use in training recipes.

ViT-Huge Features:
- 630M+ parameters (~2.5GB)
- ImageNet-21k pretraining + ImageNet-1k fine-tuning
- 224x224 or 384x384 input resolution
- Transformer-based architecture

Usage:
    python download_vit_huge_weights.py
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

def download_vit_huge_weights():
    """Download and save ViT-Huge pretrained weights"""
    
    print("🚀 Downloading ViT-Huge ImageNet-21k pretrained weights...")
    print("⚠️  This is a large model (~2.5GB) and may take several minutes to download.")
    print("💡 Make sure you have sufficient disk space and a stable internet connection.")
    
    try:
        # Available ViT-Huge models
        model_variants = {
            'vit_huge_patch14_224_in21k': 'ViT-Huge/14 (ImageNet-21k, 224x224)',
            'vit_huge_patch14_224': 'ViT-Huge/14 (ImageNet-1k, 224x224)',
            'vit_so150m2_patch16_reg1_gap_448.sbb_e200_in12k_ft_in1k': 'ViT-SO150M2 (SBB recipe, 448x448)',
        }
        
        print("\n📋 Available ViT-Huge variants:")
        for i, (model_name, description) in enumerate(model_variants.items(), 1):
            print(f"{i}. {model_name}: {description}")
        
        # Use ImageNet-21k version by default (best for transfer learning)
        model_name = 'vit_huge_patch14_224_in21k'
        print(f"\n🎯 Downloading: {model_name}")
        
        # Create model with pretrained weights
        print("📥 Loading model and downloading weights...")
        model = timm.create_model(model_name, pretrained=True)
        
        # Get current directory and create weights directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(current_dir, exist_ok=True)
        weight_path = os.path.join(current_dir, "vit_huge.pth")
        
        # Save the entire model state dict
        print(f"💾 Saving weights to: {weight_path}")
        torch.save(model.state_dict(), weight_path)
        
        # Get file size in MB
        file_size = os.path.getsize(weight_path) / (1024 * 1024)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n✅ Successfully downloaded and saved ViT-Huge weights!")
        print(f"📁 File: vit_huge.pth")
        print(f"📏 Size: {file_size:.1f} MB")
        print(f"🔢 Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"🎓 Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"🖼️  Input size: 224x224")
        print(f"🏆 Model: {model_name}")
        
        # Model architecture info
        print(f"\n📊 Model Architecture:")
        print(f"   - Family: Vision Transformer (ViT)")
        print(f"   - Size: Huge")
        print(f"   - Patch size: 14x14")
        print(f"   - Pretraining: ImageNet-21k")
        print(f"   - Architecture: Transformer encoder")
        
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
        print(f"state_dict = torch.load('vit_huge.pth')")
        print(f"model.load_state_dict(state_dict)")
        print(f"")
        print(f"# For fine-tuning, reset classifier")
        print(f"model.reset_classifier(num_classes=YOUR_NUM_CLASSES)")
        print(f"```")
        
        print(f"\n🎉 You can now use this file for fine-tuning or inference!")
        
    except Exception as e:
        print(f"❌ Error downloading weights: {e}")
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~3GB)")
        print("3. Try updating timm: pip install --upgrade timm")
        print("4. Check if the model name is correct")
        sys.exit(1)

def list_available_vit_models():
    """List all available ViT models in timm"""
    print("🔍 Searching for all ViT models in timm...")
    
    try:
        # Get all ViT models
        vit_models = timm.list_models('vit*', pretrained=True)
        
        print(f"\n📋 Available ViT models with pretrained weights ({len(vit_models)} total):")
        
        # Group by size
        sizes = ['tiny', 'small', 'base', 'large', 'huge']
        for size in sizes:
            size_models = [m for m in vit_models if size in m]
            if size_models:
                print(f"\n🔸 {size.upper()}:")
                for model in sorted(size_models)[:10]:  # Show first 10 to avoid clutter
                    print(f"   - {model}")
                if len(size_models) > 10:
                    print(f"   ... and {len(size_models) - 10} more")
    
    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_available_vit_models()
    else:
        download_vit_huge_weights() 