import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from PIL import Image

# TensorFlow 로그 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_folder(directory):
    """폴더 생성 함수"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print(f'Error: Creating directory. {directory}')

def load_tensorflow_model(model_path):
    """TensorFlow/Keras 모델 로딩"""
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_path, target_size=(256, 256)):
    """이미지 전처리"""
    try:
        # OpenCV로 이미지 읽기
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # 크기 조정
        img = cv2.resize(img, target_size)
        
        # BGR에서 RGB로 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 정규화
        img = img.astype(np.float32) / 255.0
        
        # 배치 차원 추가
        img_input = np.expand_dims(img, axis=0)
        
        return img_input, img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

def test_single_image(model, image_path, save_path=None):
    """단일 이미지 테스트"""
    # 이미지 전처리
    img_input, img_original = preprocess_image(image_path)
    
    if img_input is None:
        print("Failed to preprocess image")
        return None, None
    
    # 예측
    try:
        predictions = model.predict(img_input, verbose=0)
        
        # 예측 결과 처리
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # 다중 클래스 분류
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            probabilities = predictions[0]
        else:
            # 이진 분류 또는 회귀
            predicted_class = 0 if predictions[0][0] < 0.5 else 1
            confidence = predictions[0][0]
            probabilities = predictions[0]
        
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Prediction shape: {predictions.shape}")
        
        # 결과 시각화
        plt.figure(figsize=(15, 5))
        
        # 원본 이미지
        plt.subplot(1, 3, 1)
        plt.imshow(img_original)
        plt.title(f'Original Image\nPredicted: Class {predicted_class}\nConfidence: {confidence:.4f}')
        plt.axis('off')
        
        # 예측 확률 분포
        plt.subplot(1, 3, 2)
        if len(probabilities.shape) == 0:
            # 단일 값인 경우
            plt.bar([0, 1], [1-confidence, confidence])
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('Prediction Probabilities')
        else:
            # 다중 클래스인 경우
            classes = range(len(probabilities))
            plt.bar(classes, probabilities)
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('Prediction Probabilities')
        plt.grid(True, alpha=0.3)
        
        # 모델 구조 요약
        plt.subplot(1, 3, 3)
        plt.text(0.1, 0.9, f"Model Summary:", fontsize=12, fontweight='bold')
        plt.text(0.1, 0.8, f"Input Shape: {model.input_shape}", fontsize=10)
        plt.text(0.1, 0.7, f"Output Shape: {model.output_shape}", fontsize=10)
        plt.text(0.1, 0.6, f"Parameters: {model.count_params():,}", fontsize=10)
        plt.text(0.1, 0.5, f"Layers: {len(model.layers)}", fontsize=10)
        plt.text(0.1, 0.4, f"Prediction: {predictions[0]}", fontsize=8)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        if save_path:
            create_folder(save_path)
            plt.savefig(os.path.join(save_path, 'tensorflow_prediction.png'), dpi=300, bbox_inches='tight')
            print(f"Results saved to {save_path}/tensorflow_prediction.png")
        
        plt.tight_layout()
        plt.show()
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def test_model_with_multiple_images(model, image_folder, save_path=None):
    """여러 이미지로 모델 테스트"""
    if not os.path.exists(image_folder):
        print(f"Image folder not found: {image_folder}")
        return
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in the folder")
        return
    
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        print(f"\n=== Testing {img_file} ===")
        
        predicted_class, confidence = test_single_image(model, img_path)
        
        if predicted_class is not None:
            results.append({
                'image': img_file,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
    
    # 결과 요약
    print(f"\n=== Summary of {len(results)} images ===")
    for result in results:
        print(f"{result['image']}: Class {result['predicted_class']} (Confidence: {result['confidence']:.4f})")

def analyze_model_architecture(model):
    """모델 구조 분석"""
    print("\n=== Model Architecture Analysis ===")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {np.sum([keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    print(f"Non-trainable parameters: {np.sum([keras.backend.count_params(w) for w in model.non_trainable_weights]):,}")
    
    print("\n=== Layer Information ===")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
        if hasattr(layer, 'output_shape'):
            print(f"  Output shape: {layer.output_shape}")
        if hasattr(layer, 'count_params'):
            print(f"  Parameters: {layer.count_params():,}")
    
    # 모델 요약 플롯
    try:
        keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
        print("Model architecture saved to model_architecture.png")
    except Exception as e:
        print(f"Could not save model architecture plot: {e}")

def main():
    """메인 함수"""
    # 설정
    model_path = "./model.h5"  # TensorFlow 모델 경로
    test_image_path = "./image.png"
    save_path = "."
    
    print("=== TensorFlow Model Testing ===")
    
    # 모델 로딩
    model = load_tensorflow_model(model_path)
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # 모델 구조 분석
    analyze_model_architecture(model)
    
    # 단일 이미지 테스트
    if os.path.exists(test_image_path):
        print("\n=== Single Image Test ===")
        test_single_image(model, test_image_path, save_path)
    else:
        print(f"Test image not found: {test_image_path}")
    
    # 여러 이미지 테스트 (이미지 폴더가 있는 경우)
    test_image_folder = "../Edge_Model_Save_Recipe/Model_Validation/"
    if os.path.exists(test_image_folder):
        print("\n=== Multiple Images Test ===")
        test_model_with_multiple_images(model, test_image_folder, save_path)
    
    print("\n=== TensorFlow Model Testing Complete ===")

if __name__ == "__main__":
    main() 