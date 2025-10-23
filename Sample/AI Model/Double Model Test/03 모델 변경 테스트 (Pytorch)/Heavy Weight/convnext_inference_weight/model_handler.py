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
        print_gpu_memory_info("모델 로딩 전")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Label 로드
        label = open(os.path.join(self.base_path, "label.txt"), 'r')
        self.label_list = label.read().replace('\n','').split(',')
        
        # 가중치 파일 경로들
        self.model_path = os.path.join(self.base_path, "model.pth")
        self.weight_path = os.path.join(self.base_path, "weight.pth")
        
        # 단일 모델 생성
        logger.info(f"[MODEL INIT] ConvNeXt 모델 생성 시작...")
        self.model = create_model(
            'convnext_large_384_in22ft1k',
            pretrained=False,
            num_classes=len(self.label_list),
            drop_path_rate=0.0
        )
        
        # 모델을 device로 이동
        self.model.to(self.device)
        logger.info(f"[MODEL INIT] 모델을 {self.device}로 이동 완료")
        
        # 초기 가중치 로드 (model.pth)
        self.current_weight_type = None
        self.load_weights('model')
        logger.info(f"[MODEL INIT] 초기 가중치 로드 완료 (model.pth)")
        print_gpu_memory_info("초기 모델 로딩 후")
        
        # 가중치 변경 성능 통계
        self.weight_change_stats = {
            'total_changes': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0
        }
        
        # Transform 설정
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                              std=[0.5, 0.5, 0.5])
        ])
        
        logger.info(f"[MODEL INIT] Weight Swap 모드 초기화 완료")
        print_gpu_memory_info("모델 로딩 완료 후")

    def determine_weight_type(self):
        """현재 시간의 밀리초를 기준으로 가중치 타입 결정"""
        current_time = time.time()
        milliseconds = int((current_time * 1000) % 1000)  # 밀리초 부분 추출
        weight_selector = milliseconds % 2
        
        if weight_selector == 0:
            weight_type = 'model'
        else:
            weight_type = 'weight'
            
        logger.info(f"[WEIGHT SELECT] 시간: {current_time:.3f}, 밀리초: {milliseconds}, %2 = {weight_selector} -> {weight_type}")
        return weight_type

    def load_weights(self, target_weight_type):
        """지정된 가중치 타입 로드"""
        logger.info(f"[WEIGHT LOAD] 가중치 로드 시작: {target_weight_type}")
        load_start = time.time()
        
        try:
            # 가중치 파일 경로 결정
            if target_weight_type == 'model':
                weight_file_path = self.model_path
            else:  # weight
                weight_file_path = self.weight_path
            
            # 체크포인트 로드 (버전 호환성 처리)
            try:
                checkpoint = torch.load(weight_file_path, map_location=self.device)
            except TypeError:
                # 이전 PyTorch 버전 지원
                checkpoint = torch.load(weight_file_path, map_location=self.device)
            
            # 가중치만 교체
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 가중치 로드 시간 측정
            load_end = time.time()
            load_duration = load_end - load_start
            
            # 통계 업데이트 (초기 로딩이 아닌 경우에만)
            if self.current_weight_type is not None:
                self.weight_change_stats['total_changes'] += 1
                self.weight_change_stats['total_time'] += load_duration
                self.weight_change_stats['min_time'] = min(self.weight_change_stats['min_time'], load_duration)
                self.weight_change_stats['max_time'] = max(self.weight_change_stats['max_time'], load_duration)
                self.weight_change_stats['avg_time'] = self.weight_change_stats['total_time'] / self.weight_change_stats['total_changes']
                
                logger.info(f"[WEIGHT CHANGE TIME] 가중치 변경 시간: {load_duration:.4f}초")
                logger.info(f"[WEIGHT CHANGE STATS] 총 변경: {self.weight_change_stats['total_changes']}회, "
                           f"평균: {self.weight_change_stats['avg_time']:.4f}초, "
                           f"최소: {self.weight_change_stats['min_time']:.4f}초, "
                           f"최대: {self.weight_change_stats['max_time']:.4f}초")
            
            # 상태 업데이트
            self.current_weight_type = target_weight_type
            logger.info(f"[WEIGHT LOAD] {target_weight_type}.pth 가중치 로드 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"[WEIGHT LOAD ERROR] 가중치 로드 실패: {e}")
            return False

    def load_weight_if_needed(self, target_weight_type):
        """필요한 경우에만 가중치 로드"""
        # 동일한 가중치 타입이면 로딩 건너뛰기
        if self.current_weight_type == target_weight_type:
            logger.info(f"[WEIGHT SKIP] 동일한 가중치 타입 ({target_weight_type}) - 로딩 건너뛰기")
            return True
        
        logger.info(f"[WEIGHT CHANGE] 가중치 변경 시작: {self.current_weight_type} -> {target_weight_type}")
        return self.load_weights(target_weight_type)

    def get_current_model(self):
        """현재 시간 기반으로 적절한 가중치 로드하고 모델 반환"""
        target_weight_type = self.determine_weight_type()
        
        if self.load_weight_if_needed(target_weight_type):
            return self.model
        else:
            # 가중치 로드 실패시 현재 모델 반환
            logger.warning(f"[WEIGHT FALLBACK] 가중치 로드 실패, 현재 가중치 유지: {self.current_weight_type}")
            return self.model

    def test(self):
        """이후에 추론 하는 모든 데이터에 대해서 빠르게 진행 될 수 있도록 하는 부분"""
        # 현재 모델로 test 실행
        current_model = self.get_current_model()
        dummy_input = torch.randn(1, 3, 384, 384).to(self.device)
        try:
            with torch.no_grad():
                # autocast 없이 먼저 시도
                _ = current_model(dummy_input)
                logger.info("[TEST] 모델 테스트 성공 (float32)")
        except Exception as e:
            logger.warning(f"[TEST] float32 테스트 실패: {e}")
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True):
                        _ = current_model(dummy_input)
                logger.info("[TEST] 모델 테스트 성공 (mixed precision)")
            except Exception as e2:
                logger.error(f"[TEST] mixed precision 테스트도 실패: {e2}")
        
        # 가중치 변경 테스트
        logger.info("[TEST] 가중치 변경 테스트 시작...")
        for i in range(5):
            time.sleep(0.1)  # 밀리초 변경을 위한 대기
            current_model = self.get_current_model()
            logger.info(f"[TEST] {i+1}번째 테스트 - 가중치 타입: {self.current_weight_type}")
        logger.info("[TEST] 가중치 변경 테스트 완료")

    def __call__(self, data, context):
        print_gpu_memory_info("모델 추론 전")
        
        ## Expand_dims
        data = pickle.loads(data)
        images = np.array(data)
        
        # Validate input image
        if len(images.shape) != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got shape: {images.shape}")
        
        # Resize images to match ConvNeXt's expected input size (384x384)
        expected_height, expected_width = 384, 384
        if images.shape[:2] != (expected_height, expected_width):
            logger.info(f"Resizing image from {images.shape[:2]} to ({expected_height}, {expected_width})")
            images = cv2.resize(images, (expected_width, expected_height))
        
        # Convert to RGB and apply normalization
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        images = self.transform(images).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        result = {}

        ## 현재 시간 기반으로 가중치 선택 및 로드
        current_model = self.get_current_model()
        logger.info(f"[CURRENT MODEL] 현재 사용 중인 가중치: {self.current_weight_type}")
        
        ## Predict
        predict_start_time = time.time()
        
        try:
            with torch.no_grad():
                # autocast 없이 먼저 시도 (float32)
                predict_result = current_model(images)
                predict_result = predict_result.cpu().numpy()
            logger.info(f"[PREDICT] 추론 성공 (float32)")
        except Exception as e:
            logger.warning(f"[PREDICT] float32 추론 실패: {e}, mixed precision으로 재시도")
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True):
                        predict_result = current_model(images)
                        predict_result = predict_result.cpu().numpy()
                logger.info(f"[PREDICT] 추론 성공 (mixed precision)")
            except Exception as e2:
                logger.error(f"[PREDICT] mixed precision 추론도 실패: {e2}")
                raise e2
        
        predict_end_time = time.time()
        predict_duration = predict_end_time - predict_start_time
        logger.info(f"[PREDICT TIME] 추론 시간: {predict_duration:.4f}초")
        
        print_gpu_memory_info("모델 추론 후")
        
        adc_code = self.label_list[int(np.argmax(predict_result))]
        top1_score = float(np.max(predict_result))
        
        result['result_code'] = adc_code
        result['score'] = top1_score
        result['current_weight'] = self.current_weight_type
        result['weight_change_stats'] = self.weight_change_stats.copy()
        
        result = pickle.dumps(result)
        
        print_gpu_memory_info("완료 후")
       
        return result, context

    def __del__(self):
        """소멸자에서 모델 정리"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                self.model.cpu()
                del self.model
                logger.info("[CLEANUP] 모델 정리 완료")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info("[CLEANUP] ModelHandler 정리 완료")
        except Exception as e:
            logger.error(f"[CLEANUP ERROR] 정리 중 오류: {e}")
        
if __name__ == "__main__":
    # 더미 데이터 생성 (384x384 크기의 랜덤 이미지)
    dummy_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    data = pickle.dumps(dummy_image)
    
    # Weight Swap Test 실행
    handler = ModelHandler(None, None)
    
    # 여러 번 실행하여 동적 가중치 변경 테스트
    for i in range(10):
        logger.info(f"\n[TEST RUN] {i+1}번째 테스트 실행")
        result, context = handler(data=data, context={})
        result_dict = pickle.loads(result)
        logger.info(f"[TEST RESULT] 결과: {result_dict['result_code']}, "
                   f"점수: {result_dict['score']:.4f}, "
                   f"가중치: {result_dict['current_weight']}")
        time.sleep(0.5)  # 0.5초 대기로 밀리초 변경 확인