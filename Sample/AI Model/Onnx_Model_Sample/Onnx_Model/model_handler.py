import os
import pickle
import numpy as np
import onnxruntime as ort
import cv2
import json

class ModelHandler:
    def __init__(self, data=None, context=None):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "data", "model.onnx")
        
        # ONNX 모델 로드
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # 입력/출력 정보 가져오기
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 라벨 로드
        label_path = os.path.join(base_path, "data", "label.txt")
        with open(label_path, 'r') as f:
            self.label_list = f.read().strip().split(',')
        
        self.input_size = 299
        
    def preprocess_image(self, image_np):
        """이미지 전처리"""
        # BGR to RGB
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # 리사이즈
        image_resized = cv2.resize(image_rgb, (self.input_size, self.input_size))
        
        # 정규화 (0-1 범위로)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 정규화 (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        image_normalized = (image_normalized - 0.5) / 0.5
        
        # 차원 변환 (H, W, C) -> (1, C, H, W)
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, axis=0)
        
        return image_tensor

    def __call__(self, data, context):
        # 데이터 역직렬화 및 전처리
        image_np = pickle.loads(data)  # H x W x 3
        input_tensor = self.preprocess_image(image_np)
        
        # ONNX 추론
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        output = outputs[0]
        
        # 결과 후처리
        preds = self.softmax(output)
        pred_idx = np.argmax(preds, axis=1)[0]
        score = preds[0][pred_idx]
        
        result = {
            'result_code': self.label_list[pred_idx],
            'score': float(score)
        }
        
        return pickle.dumps(result), context
    
    def softmax(self, x):
        """Softmax 함수"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
if __name__ == "__main__":
    handler = ModelHandler()
    test_image = cv2.imread("./data/test_image.png")
    if test_image is not None:
        test_image = cv2.resize(test_image, (299, 299))
        data = pickle.dumps(test_image)
        output, _ = handler(data, context={})
        result = pickle.loads(output)
        print(result)
    else:
        print("테스트 이미지를 찾을 수 없습니다.")

