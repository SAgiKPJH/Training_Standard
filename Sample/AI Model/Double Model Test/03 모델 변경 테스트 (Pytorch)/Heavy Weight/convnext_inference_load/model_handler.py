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
        
        # 모델 파일 경로들
        self.model0_path = os.path.join(self.base_path, "model0.pth")
        self.model1_path = os.path.join(self.base_path, "model1.pth")
        
        # 모델 사전 로딩
        logger.info(f"[PRELOAD] 모델 사전 로딩 시작...")
        self.model0 = None
        self.model1 = None
        self.current_model_type = None
        
        # Model0 로딩
        self.load_model0()
        print_gpu_memory_info("Model0 로딩 후")
        
        # Model1 로딩
        self.load_model1()
        print_gpu_memory_info("Model1 로딩 후")
        
        # 모델 선택 성능 통계
        self.model_select_stats = {
            'total_selects': 0,
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
        
        logger.info(f"[PRELOAD] 모든 모델 사전 로딩 완료")
        print_gpu_memory_info("모든 모델 로딩 완료 후")

    def load_model0(self):
        """Model0 로딩"""
        logger.info(f"[PRELOAD] Model0 로딩 시작...")
        load_start = time.time()
        
        try:
            # Model0 생성
            self.model0 = create_model(
                'convnext_large_384_in22ft1k',
                pretrained=False,
                num_classes=len(self.label_list),
                drop_path_rate=0.0
            )
            
            # 체크포인트 로드 (버전 호환성 처리)
            try:
                checkpoint = torch.load(self.model0_path, map_location=self.device)
            except TypeError:
                # 이전 PyTorch 버전 지원
                checkpoint = torch.load(self.model0_path, map_location=self.device)
            
            self.model0.load_state_dict(checkpoint['model_state_dict'])
            self.model0.to(self.device)
            self.model0.eval()
            
            load_end = time.time()
            load_duration = load_end - load_start
            logger.info(f"[PRELOAD] Model0 로딩 완료 - 시간: {load_duration:.4f}초")
            
        except Exception as e:
            logger.error(f"[PRELOAD ERROR] Model0 로딩 실패: {e}")
            self.model0 = None

    def load_model1(self):
        """Model1 로딩"""
        logger.info(f"[PRELOAD] Model1 로딩 시작...")
        load_start = time.time()
        
        try:
            # Model1 생성
            self.model1 = create_model(
                'convnext_large_384_in22ft1k',
                pretrained=False,
                num_classes=len(self.label_list),
                drop_path_rate=0.0
            )
            
            # 체크포인트 로드 (버전 호환성 처리)
            try:
                checkpoint = torch.load(self.model1_path, map_location=self.device)
            except TypeError:
                # 이전 PyTorch 버전 지원
                checkpoint = torch.load(self.model1_path, map_location=self.device)
            
            self.model1.load_state_dict(checkpoint['model_state_dict'])
            self.model1.to(self.device)
            self.model1.eval()
            
            load_end = time.time()
            load_duration = load_end - load_start
            logger.info(f"[PRELOAD] Model1 로딩 완료 - 시간: {load_duration:.4f}초")
            
        except Exception as e:
            logger.error(f"[PRELOAD ERROR] Model1 로딩 실패: {e}")
            self.model1 = None

    def determine_model_type(self):
        """현재 시간의 밀리초를 기준으로 모델 타입 결정"""
        current_time = time.time()
        milliseconds = int((current_time * 1000) % 1000)  # 밀리초 부분 추출
        model_selector = milliseconds % 2
        
        if model_selector == 0:
            model_type = 'model0'
        else:
            model_type = 'model1'
            
        logger.info(f"[MODEL SELECT] 시간: {current_time:.3f}, 밀리초: {milliseconds}, %2 = {model_selector} -> {model_type}")
        return model_type

    def get_current_model(self):
        """현재 시간 기반으로 사전 로딩된 모델 선택"""
        select_start = time.time()
        
        target_model_type = self.determine_model_type()
        
        # 모델 선택
        if target_model_type == 'model0':
            if self.model0 is not None:
                selected_model = self.model0
                logger.info(f"[MODEL SELECT] Model0 선택됨")
            else:
                logger.warning(f"[MODEL SELECT] Model0이 없음, Model1 사용")
                selected_model = self.model1
                target_model_type = 'model1'
        else:  # model1
            if self.model1 is not None:
                selected_model = self.model1
                logger.info(f"[MODEL SELECT] Model1 선택됨")
            else:
                logger.warning(f"[MODEL SELECT] Model1이 없음, Model0 사용")
                selected_model = self.model0
                target_model_type = 'model0'
        
        # 선택 시간 측정
        select_end = time.time()
        select_duration = select_end - select_start
        
        # 통계 업데이트
        self.model_select_stats['total_selects'] += 1
        self.model_select_stats['total_time'] += select_duration
        self.model_select_stats['min_time'] = min(self.model_select_stats['min_time'], select_duration)
        self.model_select_stats['max_time'] = max(self.model_select_stats['max_time'], select_duration)
        self.model_select_stats['avg_time'] = self.model_select_stats['total_time'] / self.model_select_stats['total_selects']
        
        # 모델 변경 여부 확인
        if self.current_model_type != target_model_type:
            logger.info(f"[MODEL CHANGE] 모델 변경: {self.current_model_type} -> {target_model_type}")
            self.current_model_type = target_model_type
        else:
            logger.info(f"[MODEL KEEP] 동일한 모델 유지: {target_model_type}")
        
        logger.info(f"[MODEL SELECT TIME] 모델 선택 시간: {select_duration:.6f}초")
        logger.info(f"[MODEL SELECT STATS] 총 선택: {self.model_select_stats['total_selects']}회, "
                   f"평균: {self.model_select_stats['avg_time']:.6f}초, "
                   f"최소: {self.model_select_stats['min_time']:.6f}초, "
                   f"최대: {self.model_select_stats['max_time']:.6f}초")
        
        return selected_model

    def test(self):
        """이후에 추론 하는 모든 데이터에 대해서 빠르게 진행 될 수 있도록 하는 부분"""
        # 두 모델 모두로 test 실행
        dummy_input = torch.randn(1, 3, 384, 384).to(self.device)
        
        logger.info("[TEST] 사전 로딩된 모델들 테스트 시작...")
        
        # Model0 테스트
        if self.model0 is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    _ = self.model0(dummy_input)
            logger.info("[TEST] Model0 테스트 완료")
        
        # Model1 테스트
        if self.model1 is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    _ = self.model1(dummy_input)
            logger.info("[TEST] Model1 테스트 완료")
        
        # 모델 선택 테스트
        logger.info("[TEST] 모델 선택 테스트 시작...")
        for i in range(5):
            time.sleep(0.1)  # 밀리초 변경을 위한 대기
            current_model = self.get_current_model()
            logger.info(f"[TEST] {i+1}번째 테스트 - 선택된 모델: {self.current_model_type}")
        logger.info("[TEST] 모델 선택 테스트 완료")

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

        ## 사전 로딩된 모델 중에서 선택
        current_model = self.get_current_model()
        logger.info(f"[CURRENT MODEL] 현재 사용 중인 모델: {self.current_model_type}")
        
        ## Predict
        predict_start_time = time.time()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                predict_result = current_model(images)
                predict_result = predict_result.cpu().numpy()
        
        predict_end_time = time.time()
        predict_duration = predict_end_time - predict_start_time
        logger.info(f"[PREDICT TIME] 추론 시간: {predict_duration:.4f}초")
        
        print_gpu_memory_info("모델 추론 후")
        
        adc_code = self.label_list[int(np.argmax(predict_result))]
        top1_score = float(np.max(predict_result))
        
        result['result_code'] = adc_code
        result['score'] = top1_score
        result['current_model'] = self.current_model_type
        result['model_select_stats'] = self.model_select_stats.copy()
        result['models_loaded'] = {
            'model0': self.model0 is not None,
            'model1': self.model1 is not None
        }
        
        result = pickle.dumps(result)
        
        print_gpu_memory_info("완료 후")
       
        return result, context

    def __del__(self):
        """소멸자에서 모델들 정리"""
        try:
            if self.model0 is not None:
                self.model0.cpu()
                del self.model0
                logger.info("[CLEANUP] Model0 정리 완료")
            
            if self.model1 is not None:
                self.model1.cpu()
                del self.model1
                logger.info("[CLEANUP] Model1 정리 완료")
            
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
    
    # Preload Test 실행
    handler = ModelHandler(None, None)
    
    # 여러 번 실행하여 동적 모델 선택 테스트
    for i in range(10):
        logger.info(f"\n[TEST RUN] {i+1}번째 테스트 실행")
        result, context = handler(data=data, context={})
        result_dict = pickle.loads(result)
        logger.info(f"[TEST RESULT] 결과: {result_dict['result_code']}, "
                   f"점수: {result_dict['score']:.4f}, "
                   f"모델: {result_dict['current_model']}, "
                   f"로딩상태: {result_dict['models_loaded']}")
        time.sleep(0.5)  # 0.5초 대기로 밀리초 변경 확인