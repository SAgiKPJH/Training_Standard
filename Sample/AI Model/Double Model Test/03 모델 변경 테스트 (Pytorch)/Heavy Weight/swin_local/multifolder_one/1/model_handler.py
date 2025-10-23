import os
import pickle
import json
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import time
import logging
import sys
import gc
from timm import create_model

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

## GPU Memory 제한
memory_limit = 3 * 1024

def print_gpu_memory_info(stage=""):
    """GPU 메모리 사용량 출력 (GPU 0, 1 둘 다)"""
    try:
        if torch.cuda.is_available():
            # GPU 0과 1 둘 다 확인
            for gpu_idx in range(min(2, torch.cuda.device_count())):  # 최대 2개 GPU까지 확인
                try:
                    # PyTorch 메모리 정보 가져오기
                    allocated = torch.cuda.memory_allocated(gpu_idx) / (1024**2)  # MB
                    reserved = torch.cuda.memory_reserved(gpu_idx) / (1024**2)  # MB
                    max_allocated = torch.cuda.max_memory_allocated(gpu_idx) / (1024**2)  # MB
                    
                    # 전체 메모리는 설정된 제한값 사용
                    total = memory_limit  # MB
                    free = total - allocated
                    
                    logger.info(f"[GPU{gpu_idx} MEMORY {stage}] 사용: {allocated:.1f}MB, 여유: {free:.1f}MB, 전체: {total:.1f}MB, 피크: {max_allocated:.1f}MB")
                except Exception as gpu_e:
                    logger.info(f"[GPU{gpu_idx} MEMORY {stage}] GPU{gpu_idx} 메모리 정보 가져오기 실패: {gpu_e}")
        else:
            logger.info(f"[GPU MEMORY {stage}] GPU 없음")
    except Exception as e:
        logger.info(f"[GPU MEMORY {stage}] 전체 메모리 정보 가져오기 실패: {e}")

class ModelHandler:
    def __init__(self, data, context):
        print_gpu_memory_info("Swin 모델 로딩 전")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Label 로드
        label = open(os.path.join(self.base_path, "label.txt"), 'r')
        self.label_list = label.read().replace('\n','').split(',')
        
        # 모델 상태 추적
        self.model = None
        self.is_model_loaded = False
        
        # Swin Transformer용 Transform 설정 (224x224)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])  # ImageNet 정규화
        ])
        
        # 초기 모델 로딩
        self.initialize_model()
        
        print_gpu_memory_info("Swin 모델 로딩 완료 후")

    def clear_model(self):
        """현재 Swin 모델을 메모리에서 완전히 제거"""
        if self.model is not None:
            logger.info(f"[MODEL CLEAR] 기존 Swin 모델 메모리에서 제거 (폴더 1)")
            # 모델을 CPU로 이동 후 삭제
            self.model.cpu()
            del self.model
            self.model = None
            self.is_model_loaded = False
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            print_gpu_memory_info("Swin 모델 제거 후")

    def initialize_model(self):
        """Swin Transformer 모델을 새로 초기화 (기존 모델 제거 후)"""
        logger.info(f"[MODEL INIT] 폴더 1 - Swin Transformer 모델 초기화 시작")
        
        # 기존 모델 제거
        self.clear_model()
        
        try:
            # 모델 파일 로딩 (model.pth) - 만약 없으면 다운로드된 가중치 사용
            model_path = os.path.join(self.base_path, "model.pth")
            if os.path.exists(model_path):
                self.model = torch.load(model_path, map_location=self.device)
                logger.info(f"[MODEL INIT] 폴더 1 - 기존 Swin model.pth 로드")
            else:
                # 기본 Swin-Large 모델 생성
                self.model = create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=10)
                self.model.to(self.device)
                logger.info(f"[MODEL INIT] 폴더 1 - 기본 Swin-Large 모델 생성")
            
            self.model.eval()
            self.is_model_loaded = True
            logger.info(f"[MODEL INIT] 폴더 1 - Swin Transformer 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"[MODEL INIT] 폴더 1 - Swin Transformer 모델 초기화 실패: {e}")
            raise

    def test(self):        ## 이후에 추론 하는 모든 데이터에 대해서 빠르게 진행 될 수 있도록 하는 부분
        """모델 테스트 - float32와 mixed precision 모두 테스트"""
        logger.info("[SWIN TEST] 폴더 1 - Swin Transformer 모델 테스트 시작")
        
        # float32 테스트
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input)
            logger.info("[SWIN TEST] 모델 테스트 성공 (float32)")
        except Exception as e:
            logger.warning(f"[SWIN TEST] float32 테스트 실패: {e}")
        
        # mixed precision 테스트
        try:
            with torch.cuda.amp.autocast():
                output = self.model(dummy_input)
            logger.info("[SWIN TEST] 모델 테스트 성공 (mixed precision)")
        except Exception as e2:
            logger.error(f"[SWIN TEST] mixed precision 테스트도 실패: {e2}")
        
        logger.info("[SWIN TEST] 폴더 1 - Swin Transformer 모델 테스트 완료")

    def __call__(self, data, context):
        """Swin Transformer 추론 수행"""
        start_time = time.time()
        
        try:
            # 데이터 전처리
            if isinstance(data, bytes):
                image_data = pickle.loads(data)
            else:
                image_data = data
            
            # OpenCV 이미지로 변환
            if isinstance(image_data, np.ndarray):
                image = image_data
            else:
                image = np.array(image_data)
            
            # BGR to RGB 변환
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 리사이즈 (224x224)
            image = cv2.resize(image, (224, 224))
            
            # Transform 적용
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 추론
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # 결과 생성
            result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "label": self.label_list[predicted_class] if predicted_class < len(self.label_list) else "unknown"
            }
            
            # 성능 정보 추가
            inference_time = time.time() - start_time
            context["inference_time"] = str(inference_time)
            context["model_type"] = "swin_transformer"
            context["folder"] = "1"
            
            logger.info(f"[SWIN INFERENCE] 폴더 1 - 클래스: {result['label']}, 신뢰도: {confidence:.4f}, 시간: {inference_time:.4f}s")
            
            return result, context
            
        except Exception as e:
            logger.error(f"[SWIN INFERENCE] 폴더 1 - 추론 실패: {e}")
            raise

    def __del__(self):
        """소멸자에서 모델 정리"""
        try:
            self.clear_model()
            logger.info("[CLEANUP] 폴더 1 - Swin Transformer ModelHandler 정리 완료")
        except Exception as e:
            logger.error(f"[CLEANUP ERROR] 폴더 1 - Swin 정리 중 오류: {e}")

if __name__ == "__main__":
    # Swin Transformer Test 실행
    import pickle
    import numpy as np
    
    # 더미 데이터 생성 (224x224 크기의 랜덤 이미지)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    data = pickle.dumps(dummy_image)
    
    try:
        handler = ModelHandler(None, {})
        handler.test()
        
        result_dict, context = handler(data=data, context={})
        logger.info(f"[SWIN TEST RESULT] 결과: {result_dict['result_code']}, "
                   f"클래스: {result_dict['label']}, 신뢰도: {result_dict['confidence']:.4f}")
        
    except Exception as e:
        logger.error(f"[SWIN TEST ERROR] 테스트 실행 중 오류: {e}")
