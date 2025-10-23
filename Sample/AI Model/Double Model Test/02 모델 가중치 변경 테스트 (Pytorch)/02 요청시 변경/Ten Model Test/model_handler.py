import os
import pickle
import json
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import time
import logging
import random

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

## GPU Memory 제한
#memory_limit = 3 * 1024
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
        
        # 사용 가능한 모델 파일들 확인
        self.available_models = {}
        self.model_files = []
        
        logger.info(f"[LOG] 10개 모델 파일 확인 중...")
        
        for i in range(10):
            model_file = f"model{i}.pth"
            model_path = os.path.join(self.base_path, model_file)
            
            if os.path.exists(model_path):
                self.model_files.append(model_file)
                logger.info(f"[LOG] {model_file} 발견")
            else:
                logger.info(f"[WARNING] {model_file} 파일이 없습니다.")
        
        if not self.model_files:
            raise ValueError("사용 가능한 모델 파일이 없습니다.")
        
        # Live Test 설정
        self.current_model = None
        self.current_model_index = 0
        self.last_model_change_time = time.time()
        self.model_change_interval = 1.5  # 1.5초마다 모델 변경
        
        # 모델 변경 성능 통계
        self.model_change_stats = {
            'total_changes': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0
        }
        
        logger.info(f"[LOG] 총 {len(self.model_files)}개 모델 파일 발견")
        logger.info(f"[LOG] Live Test 모드 초기화 완료")
        print_gpu_memory_info("모델 로딩 완료 후")

    def change_model_dynamically(self, force_change=False):
        """동적으로 모델을 변경하고 시간을 측정"""
        current_time = time.time()
        
        # 일정 시간마다 또는 강제로 모델 변경
        if not force_change and (current_time - self.last_model_change_time) < self.model_change_interval:
            return False
            
        logger.info(f"[MODEL CHANGE] 모델 변경 시작...")
        model_change_start = time.time()
        
        # 다음 모델 파일 선택
        next_model_index = (self.current_model_index + 1) % len(self.model_files)
        model_file = self.model_files[next_model_index]
        model_path = os.path.join(self.base_path, model_file)
        
        try:
            # 기존 모델 정리
            if self.current_model is not None:
                del self.current_model
                torch.cuda.empty_cache()  # GPU 메모리 정리
            
            # 새 모델 로드
            self.current_model = torch.load(model_path, map_location=self.device)
            self.current_model.to(self.device)
            self.current_model.eval()
            
            # 모델 변경 시간 측정
            model_change_end = time.time()
            change_duration = model_change_end - model_change_start
            
            # 통계 업데이트
            self.model_change_stats['total_changes'] += 1
            self.model_change_stats['total_time'] += change_duration
            self.model_change_stats['min_time'] = min(self.model_change_stats['min_time'], change_duration)
            self.model_change_stats['max_time'] = max(self.model_change_stats['max_time'], change_duration)
            self.model_change_stats['avg_time'] = self.model_change_stats['total_time'] / self.model_change_stats['total_changes']
            
            logger.info(f"[MODEL CHANGE] {model_file} 로드 완료")
            logger.info(f"[MODEL CHANGE TIME] 모델 변경 시간: {change_duration:.4f}초")
            logger.info(f"[MODEL CHANGE STATS] 총 변경: {self.model_change_stats['total_changes']}회, "
                       f"평균: {self.model_change_stats['avg_time']:.4f}초, "
                       f"최소: {self.model_change_stats['min_time']:.4f}초, "
                       f"최대: {self.model_change_stats['max_time']:.4f}초")
            
            # 상태 업데이트
            self.current_model_index = next_model_index
            self.last_model_change_time = current_time
            
            print_gpu_memory_info(f"모델 변경 후 ({model_file})")
            
            return True
            
        except Exception as e:
            logger.error(f"[MODEL CHANGE ERROR] 모델 변경 실패: {e}")
            return False

    def get_current_model(self):
        """현재 활성 모델 반환 (필요시 모델 변경)"""
        # 동적 모델 변경 체크
        self.change_model_dynamically()
        
        if self.current_model is None:
            # 초기 모델 설정
            self.change_model_dynamically(force_change=True)
            
        return self.current_model

    def test(self):
        """이후에 추론 하는 모든 데이터에 대해서 빠르게 진행 될 수 있도록 하는 부분"""
        # 현재 모델로 test 실행
        current_model = self.get_current_model()
        dummy_input = torch.randn(1, 3, 299, 299).to(self.device)
        with torch.no_grad():
            _ = current_model(dummy_input)
        
        # 모델 변경 테스트
        logger.info("[TEST] 모델 변경 테스트 시작...")
        for i in range(3):
            self.change_model_dynamically(force_change=True)
            time.sleep(0.1)  # 짧은 대기
        logger.info("[TEST] 모델 변경 테스트 완료")

    def __call__(self, data, context):
        print_gpu_memory_info("모델 추론 전")
        
        ## Expand_dims
        data = pickle.loads(data)
        images = np.array(data) # input_shape[0],input_shape[1],3
        
        # Validate input image
        if len(images.shape) != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got shape: {images.shape}")
        
        # 현재 모델 가져오기 (동적 모델 변경 포함)
        current_model = self.get_current_model()
        
        current_model_file = self.model_files[self.current_model_index]
        logger.info(f"[CURRENT MODEL] 현재 사용 중인 모델: {current_model_file}")
        
        # Resize images to match model's expected input size (299x299)
        expected_height, expected_width = 299, 299
        if images.shape[:2] != (expected_height, expected_width):
            logger.info(f"Resizing image from {images.shape[:2]} to ({expected_height}, {expected_width})")
            images = cv2.resize(images, (expected_width, expected_height))
        
        # Convert to PyTorch tensor and normalize
        # OpenCV loads images in BGR format, convert to RGB
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        images = images.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Convert to PyTorch tensor and add batch dimension
        images = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        result = {}
        
        ## Label Data Load
        label = open(os.path.join(self.base_path, "label.txt"),'r')
        label_list = label.read().replace('\n','')
        label_list = label_list.split(',')

        ## Predict
        predict_start_time = time.time()
        
        with torch.no_grad():
            predict_result = current_model(images)
            # Convert to numpy for processing
            predict_result = predict_result.cpu().numpy()
        
        predict_end_time = time.time()
        
        predict_duration = predict_end_time - predict_start_time
        logger.info(f"[PREDICT TIME] 추론 시간: {predict_duration:.4f}초")
        
        print_gpu_memory_info("모델 추론 후")
        
        adc_code = label_list[int(np.argmax(predict_result))]  # np.argmax 대신 int로 변환
        top1_score = float(np.max(predict_result))  # np.max 대신 float로 변환
        
        result['result_code'] = adc_code
        result['score'] = top1_score
        result['current_model'] = current_model_file
        result['model_change_stats'] = self.model_change_stats.copy()
        result['available_models'] = self.model_files.copy()
        
        result = pickle.dumps(result)
        
        print_gpu_memory_info("완료 후")
       
        return result, context
        
if __name__ == "__main__":
    img = cv2.imread(os.path.join(r'D:\ADC60_Trainset\01','2_000_MCP20503A00-004_MS2360033-02A_00-39.png'))
    if img is None:
        # 테스트용 더미 이미지 생성
        img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)
        logger.info("[TEST] 더미 이미지 생성")
    else:
        img = cv2.resize(img,(299,299))
        
    data = pickle.dumps(img)
    
    # Live Test 실행
    handler = ModelHandler(None, None)
    
    # 여러 번 실행하여 동적 모델 변경 테스트
    for i in range(8):
        logger.info(f"[TEST RUN] {i+1}번째 테스트 실행")
        result, context = handler(data=data, context={})
        result_dict = pickle.loads(result)
        logger.info(f"[TEST RESULT] 결과: {result_dict['result_code']}, 점수: {result_dict['score']:.4f}, 모델: {result_dict['current_model']}")
        logger.info(f"[AVAILABLE MODELS] 사용 가능한 모델: {result_dict['available_models']}")
        time.sleep(0.8)  # 0.8초 대기 