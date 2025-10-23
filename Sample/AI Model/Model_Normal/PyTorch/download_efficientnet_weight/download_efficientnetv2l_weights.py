"""
EfficientNetV2-L Pretrained Weights Downloader

This script downloads ImageNet pretrained weights for EfficientNetV2-L
and saves them as 'efficientnetv2l_notop.h5' for use in the training recipe.

Usage:
    python download_efficientnetv2l_weights.py
"""

import os
import sys

try:
    # Try different import methods for Keras/TensorFlow
    try:
        from keras.applications.efficientnet_v2 import EfficientNetV2L
    except ImportError:
        try:
            from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
        except ImportError:
            import tensorflow as tf
            EfficientNetV2L = tf.keras.applications.EfficientNetV2L
except ImportError as e:
    print(f"❌ Could not import EfficientNetV2L: {e}")
    print("Please install TensorFlow: pip install tensorflow")
    sys.exit(1)

def download_efficientnetv2l_weights():
    """Download and save EfficientNetV2-L pretrained weights"""
    
    print("🚀 Downloading EfficientNetV2-L ImageNet pretrained weights...")
    print("This may take a few minutes depending on your internet connection.")
    
    try:
        # Create EfficientNetV2L model with ImageNet weights
        # Using 480x480 input size for optimal performance
        model = EfficientNetV2L(
            input_shape=(480, 480, 3), 
            weights='imagenet', 
            include_top=False
        )
        
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(current_dir, "efficientnetv2l_notop.h5")
        
        # Save weights
        print(f"💾 Saving weights to: {weight_path}")
        model.save_weights(weight_path)
        
        # Get file size in MB
        file_size = os.path.getsize(weight_path) / (1024 * 1024)
        
        print(f"✅ Successfully downloaded and saved EfficientNetV2-L weights!")
        print(f"📁 File: efficientnetv2l_notop.h5")
        print(f"📏 Size: {file_size:.1f} MB")
        print(f"🔢 Parameters: {model.count_params():,}")
        print(f"🎯 Input size: 480x480")
        
        print("\n🎉 You can now use this file with train_recipe_efficientnetv2l.py!")
        
    except Exception as e:
        print(f"❌ Error downloading weights: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    download_efficientnetv2l_weights() 