import os
import pickle
import json
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import time
import logging

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

## GPU Memory 제한
memory_limit = 300

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
        
        # 모델 파일 경로
        self.model10_path = os.path.join(self.base_path, "model10.pth")
        self.model4_path = os.path.join(self.base_path, "model4.pth")
        
        # 레이블 파일 로드
        self.label10_path = os.path.join(self.base_path, "label10.txt")
        self.label4_path = os.path.join(self.base_path, "label4.txt")
        
        # 현재 모델 상태
        self.current_model = None
        self.current_model_type = None  # 'model10' 또는 'model4'
        self.current_labels = None
        
        # 가중치 변경 통계
        self.weight_change_stats = {
            'total_changes': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0
        }
        
        # 기본 모델(model10) 로드
        self.load_model('model4')
        logger.info(f"[LOG] 초기 모델(model10) 로드 완료")
        print_gpu_memory_info("초기 모델 로딩 후")

    def load_labels(self, model_type):
        """레이블 파일 로드"""
        label_path = self.label10_path if model_type == 'model10' else self.label4_path
        with open(label_path, 'r') as f:
            labels = f.read().replace('\n', '').split(',')
        return labels

    def load_model(self, model_type):
        """모델 로드 (model10 또는 model4)"""
        if self.current_model_type == model_type:
            logger.info(f"[WEIGHT SKIP] 동일한 모델 타입 ({model_type}) - 로딩 건너뛰기")
            return
        
        logger.info(f"[MODEL CHANGE] 모델 변경 시작: {self.current_model_type} -> {model_type}")
        model_change_start = time.time()
        
        try:
            # 모델 파일 경로 결정
            model_path = self.model10_path if model_type == 'model10' else self.model4_path
            
            # 새 모델 로드
            self.current_model = torch.load(model_path, map_location=self.device, weights_only=False)
            self.current_model.to(self.device)
            self.current_model.eval()
            
            # 레이블 로드
            self.current_labels = self.load_labels(model_type)
            
            # 상태 업데이트
            self.current_model_type = model_type
            
            # 모델 변경 시간 측정
            model_change_end = time.time()
            change_duration = model_change_end - model_change_start
            
            # 통계 업데이트
            self.weight_change_stats['total_changes'] += 1
            self.weight_change_stats['total_time'] += change_duration
            self.weight_change_stats['min_time'] = min(self.weight_change_stats['min_time'], change_duration)
            self.weight_change_stats['max_time'] = max(self.weight_change_stats['max_time'], change_duration)
            self.weight_change_stats['avg_time'] = self.weight_change_stats['total_time'] / self.weight_change_stats['total_changes']
            
            logger.info(f"[MODEL CHANGE TIME] 모델 변경 시간: {change_duration:.4f}초")
            logger.info(f"[MODEL CHANGE STATS] 총 변경: {self.weight_change_stats['total_changes']}회, "
                       f"평균: {self.weight_change_stats['avg_time']:.4f}초, "
                       f"최소: {self.weight_change_stats['min_time']:.4f}초, "
                       f"최대: {self.weight_change_stats['max_time']:.4f}초")
            
        except Exception as e:
            logger.error(f"[MODEL CHANGE ERROR] 모델 변경 실패: {e}")
            raise

    def test(self):
        """이후에 추론 하는 모든 데이터에 대해서 빠르게 진행 될 수 있도록 하는 부분"""
        # 현재 모델로 test 실행
        dummy_input = torch.randn(1, 3, 299, 299).to(self.device)
        with torch.no_grad():
            _ = self.current_model(dummy_input)
        
        # 모델 변경 테스트
        logger.info("[TEST] 모델 변경 테스트 시작...")
        test_types = ['model10', 'model4', 'model10', 'model4', 'model10']
        for i, model_type in enumerate(test_types):
            self.load_model(model_type)
            logger.info(f"[TEST] {i+1}번째 테스트 - 모델 타입: {model_type}")
        logger.info("[TEST] 모델 변경 테스트 완료")

    def determine_model_type(self):
        """현재 시간의 밀리초를 기준으로 모델 타입 결정"""
        current_time = time.time()
        milliseconds = int((current_time * 1000) % 1000)  # 밀리초 부분 추출
        model_selector = milliseconds % 2
        
        if model_selector == 1:
            model_type = 'model10'
        else:
            model_type = 'model4'
            
        logger.info(f"[MODEL SELECT] 시간: {current_time:.3f}, 밀리초: {milliseconds}, %2 = {model_selector} -> {model_type}")
        return model_type

    def get_current_model(self):
        """현재 시간 기반으로 적절한 모델 선택"""
        target_model_type = self.determine_model_type()
        self.load_model(target_model_type)
        return self.current_model

    def __call__(self, data, context):
        print_gpu_memory_info("모델 추론 전")
        
        ## Expand_dims
        data = pickle.loads(data)
        images = np.array(data)
        
        # Validate input image
        if len(images.shape) != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got shape: {images.shape}")
        
        # Resize images to match model's expected input size (299x299)
        expected_height, expected_width = 299, 299
        if images.shape[:2] != (expected_height, expected_width):
            logger.info(f"Resizing image from {images.shape[:2]} to ({expected_height}, {expected_width})")
            images = cv2.resize(images, (expected_width, expected_height))
        
        # Convert to RGB and normalize
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        images = images.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor
        images = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        result = {}
        
        ## 현재 시간 기반으로 모델 선택
        current_model = self.get_current_model()
        logger.info(f"[CURRENT MODEL] 현재 사용 중인 모델: {self.current_model_type}")
        
        ## Predict
        predict_start_time = time.time()
        
        with torch.no_grad():
            predict_result = current_model(images)
            predict_result = predict_result.cpu().numpy()
        
        predict_end_time = time.time()
        predict_duration = predict_end_time - predict_start_time
        logger.info(f"[PREDICT TIME] 추론 시간: {predict_duration:.4f}초")
        
        print_gpu_memory_info("모델 추론 후")
        
        adc_code = self.current_labels[int(np.argmax(predict_result))]
        top1_score = float(np.max(predict_result))
        
        result['result_code'] = adc_code
        result['score'] = top1_score
        result['model_type'] = self.current_model_type
        result['weight_change_stats'] = self.weight_change_stats.copy()
        
        result = pickle.dumps(result)
        
        print_gpu_memory_info("완료 후")
       
        return result, context

if __name__ == "__main__":
    # 더미 데이터 생성
    dummy_image = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
    data = pickle.dumps(dummy_image)
    
    # ModelHandler 인스턴스 생성
    handler = ModelHandler(None, None)
    
    # 테스트 실행 (밀리초 기반 자동 모델 선택)
    logger.info("\n=== 밀리초 기반 모델 전환 테스트 시작 ===\n")
    
    for i in range(10):
        logger.info(f"\n[TEST RUN] {i+1}번째 테스트 실행")
        result, _ = handler(data=data, context={})
        result_dict = pickle.loads(result)
        logger.info(f"[TEST RESULT] 결과: {result_dict['result_code']}, "
                   f"점수: {result_dict['score']:.4f}, "
                   f"모델: {result_dict['model_type']}")
        time.sleep(0.1)  # 0.1초 대기로 밀리초 변경 확인
