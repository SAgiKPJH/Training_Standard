import json
import os
import pickle
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import tensorflow as tf
import json

## Gpu Memory 제한
memory_limit = 3
gpus = tf.config.list_physical_devices('GPU')
if gpus :
    try : 
        for gpu in gpus :
            tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit*1024)])
    except RuntimeError as e :
        print(e)
        pass


class ModelHandler:
    def __init__(self, data=None, context=None):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "data", "model.pth")

        label_path = os.path.join(base_path, "data", "label.txt")
        with open(label_path, 'r') as f:
            self.label_list = f.read().strip().split(',')

        # TorchScript JIT 모델 로드
        self.model = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()

        self.input_size = 299
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, data, context):
        # 데이터 역직렬화 및 전처리
        image_np = pickle.loads(data)  # H x W x 3
        image_tensor = self.transform(image_np).unsqueeze(0)  # (1, 3, 299, 299)

        # 추론
        with torch.no_grad():
            outputs = self.model(image_tensor)
            preds = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(preds, dim=1).item()
            score = preds[0][pred_idx].item()

        result = {
            'result_code': self.label_list[pred_idx],
            'score': score
        }

        return pickle.dumps(result), context
        
if __name__ == "__main__":
    import cv2
    test_image = cv2.imread("test_image.png", cv2.IMREAD_COLOR)
    test_image = cv2.resize(test_image, (299, 299))
    data = pickle.dumps(test_image)

    handler = ModelHandler()
    output, _ = handler(data, context={})
    
    result = pickle.loads(output)
    print(result)