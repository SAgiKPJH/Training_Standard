import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

def load_pytorch_model(model_path):
    """PyTorch 모델 로딩"""
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found.")
        return None
    
    try:
        # 모델 직접 로딩
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 만약 state_dict만 저장되어 있는 경우
        if isinstance(model, dict) and 'model_state_dict' in model:
            print("Warning: Only state_dict found. Model architecture is needed.")
            return None
            
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        print(f"Model type: {type(model)}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_path, target_size=(299, 299)):
    """이미지 전처리"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def test_single_image(model, image_path, save_path=None):
    """단일 이미지 테스트"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 이미지 전처리
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # 예측
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            
            # 출력 형태에 따라 처리
            if len(outputs.shape) == 2:  # (batch_size, num_classes)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            else:
                # 다른 형태의 출력인 경우
                probabilities = outputs
                predicted_class = torch.argmax(outputs).item()
                confidence = torch.max(outputs).item()
        
        print(f"Model output shape: {outputs.shape}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        # 결과 시각화
        plt.figure(figsize=(12, 4))
        
        # 원본 이미지
        plt.subplot(1, 2, 1)
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f'Original Image\nPredicted: Class {predicted_class}\nConfidence: {confidence:.4f}')
        plt.axis('off')
        
        # 예측 확률 분포
        plt.subplot(1, 2, 2)
        if len(outputs.shape) == 2:
            probs = probabilities[0].cpu().numpy()
            classes = range(len(probs))
            plt.bar(classes, probs)
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('Prediction Probabilities')
        else:
            # 다른 형태의 출력 시각화
            output_flat = outputs.cpu().numpy().flatten()
            plt.plot(output_flat)
            plt.xlabel('Output Index')
            plt.ylabel('Value')
            plt.title('Model Output')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'pytorch_prediction.png'), dpi=300, bbox_inches='tight')
            print(f"Results saved to {save_path}/pytorch_prediction.png")
        
        plt.tight_layout()
        plt.show()
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None



def main():
    """메인 함수"""
    # 설정
    model_path = "./model.pth"  # PyTorch 모델 경로
    test_image_path = "./image.png"
    save_path = "."
    
    print("=== PyTorch Model Testing ===")
    
    # 모델 로딩
    model = load_pytorch_model(model_path)
    
    if model is None:
        print("Failed to load model. Please check the model file.")
        return
    
    # 단일 이미지 테스트
    if os.path.exists(test_image_path):
        print("\n=== Single Image Test ===")
        test_single_image(model, test_image_path, save_path)
    else:
        print(f"Test image not found: {test_image_path}")
    
    print("\n=== PyTorch Model Testing Complete ===")

if __name__ == "__main__":
    main() 