import numpy as np
import pickle
from model_handler import ModelHandler

# Create a test image with the problematic size (600x600x3)
test_image = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
print(f"Original image shape: {test_image.shape}")

# Serialize the image data (as it would be in your pipeline)
test_data = pickle.dumps(test_image)

# Test the ModelHandler with the fixed code
try:
    handler = ModelHandler(None, None)
    result, context = handler(test_data, {})
    print("✅ Success! The image resizing fix works correctly.")
    print(f"Result keys: {list(pickle.loads(result).keys())}")
except Exception as e:
    print(f"❌ Error: {e}") 