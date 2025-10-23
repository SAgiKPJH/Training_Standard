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
        print_gpu_memory_info("ViT 모델 로딩 전")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Label 로드
        label = open(os.path.join(self.base_path, "label.txt"), 'r')
        self.label_list = label.read().replace('\n','').split(',')
        
        # 모델 상태 추적
        self.model = None
        self.is_model_loaded = False
        
        # Vision Transformer용 Transform 설정 (224x224)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])  # ImageNet 정규화
        ])
        
        # 초기 모델 로딩
        self.initialize_model()
        
        print_gpu_memory_info("ViT 모델 로딩 완료 후")

    def clear_model(self):
        """현재 ViT 모델을 메모리에서 완전히 제거"""
        if self.model is not None:
            logger.info(f"[MODEL CLEAR] 기존 ViT 모델 메모리에서 제거 (폴더 1)")
            # 모델을 CPU로 이동 후 삭제
            self.model.cpu()
            del self.model
            self.model = None
            self.is_model_loaded = False
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            print_gpu_memory_info("ViT 모델 제거 후")

    def initialize_model(self):
        """Vision Transformer 모델을 새로 초기화 (기존 모델 제거 후)"""
        logger.info(f"[MODEL INIT] 폴더 1 - Vision Transformer 모델 초기화 시작")
        
        # 기존 모델 제거
        self.clear_model()
        
        try:
            # 모델 파일 로딩 (model.pth) - 만약 없으면 다운로드된 가중치 사용
            model_path = os.path.join(self.base_path, "model.pth")
            if os.path.exists(model_path):
                self.model = torch.load(model_path, map_location=self.device, weights_only=False)
                logger.info(f"[MODEL INIT] 폴더 1 - 기존 ViT model.pth 로드")
            else:
                # 다운로드된 가중치가 있는지 확인
                download_weight_path = None
                for file in os.listdir(self.base_path):
                    if file.endswith('.pth') and 'vit' in file.lower():
                        download_weight_path = os.path.join(self.base_path, file)
                        break
                
                if download_weight_path and os.path.exists(download_weight_path):
                    logger.info(f"[MODEL INIT] 폴더 1 - 다운로드된 가중치 사용: {download_weight_path}")
                    # 다운로드된 가중치는 state_dict이므로 모델 구조를 먼저 생성
                    self.model = create_model('vit_huge_patch14_224_in21k', pretrained=False, num_classes=len(self.label_list))
                    state_dict = torch.load(download_weight_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(f"[MODEL INIT] 폴더 1 - ViT 다운로드 가중치 로드 완료")
                else:
                    # 가중치가 없으면 사전 훈련된 모델 생성
                    logger.info(f"[MODEL INIT] 폴더 1 - 사전 훈련된 ViT 모델 생성")
                    self.model = create_model('vit_huge_patch14_224_in21k', 
                                            pretrained=True, 
                                            num_classes=len(self.label_list))
            
            self.model.to(self.device)
            self.model.eval()
            self.is_model_loaded = True
            
            logger.info(f"[MODEL INIT] 폴더 1 - Vision Transformer 모델 초기화 완료")
            print_gpu_memory_info("ViT 모델 초기화 후")
            
        except Exception as e:
            logger.error(f"[MODEL INIT ERROR] 폴더 1 - Vision Transformer 모델 초기화 실패: {e}")
            self.is_model_loaded = False
            raise

    def test(self):        ## 이후에 추론 하는 모든 데이터에 대해서 빠르게 진행 될 수 있도록 하는 부분
        print_gpu_memory_info("ViT 모델 테스트 전")
        
        # 더미 데이터로 워밍업 (ViT: 224x224)
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        try:
            with torch.no_grad():
                # autocast 없이 먼저 시도
                _ = self.model(dummy_input)
                logger.info("[VIT TEST] 모델 테스트 성공 (float32)")
        except Exception as e:
            logger.warning(f"[VIT TEST] float32 테스트 실패: {e}")
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True):
                        _ = self.model(dummy_input)
                logger.info("[VIT TEST] 모델 테스트 성공 (mixed precision)")
            except Exception as e2:
                logger.error(f"[VIT TEST] mixed precision 테스트도 실패: {e2}")
        
        print_gpu_memory_info("ViT 모델 테스트 완료 후")

    def __call__(self, data, context):
        print_gpu_memory_info("ViT 모델 추론 전")
        
        ## Expand_dims
        data = pickle.loads(data)
        images = np.array(data)
        
        # Validate input image
        if len(images.shape) != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got shape: {images.shape}")
        
        # Resize images to match ViT's expected input size (224x224)
        expected_height, expected_width = 224, 224
        if images.shape[:2] != (expected_height, expected_width):
            logger.info(f"Resizing image from {images.shape[:2]} to ({expected_height}, {expected_width})")
            images = cv2.resize(images, (expected_width, expected_height))
        
        # Convert to RGB and apply normalization
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        images = self.transform(images).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        result = {}

        ## Predict
        predict_start_time = time.time()
        
        try:
            with torch.no_grad():
                # autocast 없이 먼저 시도 (float32)
                predict_result = self.model(images)
                predict_result = predict_result.cpu().numpy()
            logger.info(f"[VIT PREDICT] 추론 성공 (float32)")
        except Exception as e:
            logger.warning(f"[VIT PREDICT] float32 추론 실패: {e}, mixed precision으로 재시도")
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True):
                        predict_result = self.model(images)
                        predict_result = predict_result.cpu().numpy()
                logger.info(f"[VIT PREDICT] 추론 성공 (mixed precision)")
            except Exception as e2:
                logger.error(f"[VIT PREDICT] mixed precision 추론도 실패: {e2}")
                raise e2
        
        predict_end_time = time.time()
        predict_duration = predict_end_time - predict_start_time
        logger.info(f"[VIT PREDICT TIME] 추론 시간: {predict_duration:.4f}초")
        
        print_gpu_memory_info("ViT 모델 추론 후")
        
        adc_code = self.label_list[int(np.argmax(predict_result))]
        top1_score = float(np.max(predict_result))
        
        result['result_code'] = adc_code
        result['score'] = top1_score
        result['model_type'] = 'vision_transformer'
        result['folder'] = '1'
        
        result = pickle.dumps(result)
        
        print_gpu_memory_info("ViT 완료 후")
       
        return result, context

    def __del__(self):
        """소멸자에서 ViT 모델 정리"""
        try:
            self.clear_model()
            logger.info("[CLEANUP] 폴더 1 - ViT ModelHandler 정리 완료")
        except Exception as e:
            logger.error(f"[CLEANUP ERROR] 폴더 1 - ViT 정리 중 오류: {e}")
        
if __name__ == "__main__":
    # 더미 데이터 생성 (224x224 크기의 랜덤 이미지)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    data = pickle.dumps(dummy_image)
    
    # Vision Transformer Test 실행
    handler = ModelHandler(None, None)
    
    # 테스트 실행
    result, context = handler(data=data, context={})
    result_dict = pickle.loads(result)
    logger.info(f"[VIT TEST RESULT] 결과: {result_dict['result_code']}, "
               f"점수: {result_dict['score']:.4f}, "
               f"모델: {result_dict['model_type']}")