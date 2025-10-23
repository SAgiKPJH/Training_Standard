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
        
        # 두 모델을 모두 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model0.pth")
        self.model1 = torch.load(model1_path, map_location=self.device)
        self.model1.to(self.device)  # 명시적으로 GPU로 이동
        self.model1.eval()
        logger.info(f"[LOG] model0.pth 로드 성공")
        print_gpu_memory_info("model0.pth 로딩 후")
        
        model2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model1.pth")
        self.model2 = torch.load(model2_path, map_location=self.device)
        self.model2.to(self.device)  # 명시적으로 GPU로 이동
        self.model2.eval()
        logger.info(f"[LOG] model1.pth 로드 성공")
        print_gpu_memory_info("model1.pth 로딩 후")
        
        logger.info(f"[LOG] 총 2개 모델 로드 완료")
        print_gpu_memory_info("모델 로딩 완료 후")

    def test(self):        ## 이후에 추론 하는 모든 데이터에 대해서 빠르게 진행 될 수 있도록 하는 부분
        # 두 모델 모두 test 실행
        for model in [self.model1, self.model2]:
            dummy_input = torch.randn(1, 3, 299, 299).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)

    def get_current_model(self):
        """현재 시간의 소수점 이하 부분을 *100해서 %2로 모델 선택"""
        current_time = time.time()
        decimal_part = current_time - int(current_time)  # 소수점 이하 부분
        decimal_x100 = int(decimal_part * 100)
        selected_model_idx = decimal_x100 % 2
        
        if selected_model_idx == 1:  # 홀수
            logger.info(f"[LOG] 소수점 이하: {decimal_part:.4f} -> {decimal_x100} % 2 = {selected_model_idx} -> model0.pth 사용")
            return self.model1
        else:  # 짝수
            logger.info(f"[LOG] 소수점 이하: {decimal_part:.4f} -> {decimal_x100} % 2 = {selected_model_idx} -> model1.pth 사용")
            return self.model2

    def __call__(self, data, context):
        print_gpu_memory_info("모델 추론 전")
        
        ## Expand_dims
        data = pickle.loads(data)
        images = np.array(data) # input_shape[0],input_shape[1],3
        
        # Validate input image
        if len(images.shape) != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got shape: {images.shape}")
        
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
        label = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"label.txt"),'r')
        label_list = label.read().replace('\n','')
        label_list = label_list.split(',')

        ## 현재 시간에 따라 모델 선택
        current_model = self.get_current_model()
        
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
        
        result = pickle.dumps(result)
        
        print_gpu_memory_info("완료 후")
       
        return result, context
        
if __name__ == "__main__":
    img = cv2.imread(os.path.join(r'D:\ADC60_Trainset\01','2_000_MCP20503A00-004_MS2360033-02A_00-39.png'))
    img = cv2.resize(img,(299,299))
    data = pickle.dumps(img)
    ModelHandler(None, None).__call__(data=data,context={}) 