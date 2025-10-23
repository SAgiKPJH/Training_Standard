import os
import pickle
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from keras.models import *
import cv2
import time

import tensorflow as tf
import json

## Gpu Memory 제한
#memory_limit = 3 * 1024
memory_limit = 300
gpus = tf.config.list_physical_devices('GPU')
if gpus :
    try : 
        for gpu in gpus :
            tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
    except RuntimeError as e :
        print(e)
        pass

def print_gpu_memory_info(stage=""):
    """GPU 메모리 사용량 출력 (GPU 0, 1 둘 다)"""
    try:
        if gpus:
            # GPU 0과 1 둘 다 확인
            for gpu_idx in range(min(2, len(gpus))):  # 최대 2개 GPU까지 확인
                try:
                    gpu_info = tf.config.experimental.get_memory_info(f'GPU:{gpu_idx}')
                    current = gpu_info['current'] / (1024**2)  # MB
                    peak = gpu_info['peak'] / (1024**2)  # MB
                    
                    # 전체 메모리는 설정된 제한값 사용
                    total = memory_limit  # MB
                    free = total - current
                    
                    print(f"[GPU{gpu_idx} MEMORY {stage}] 사용: {current:.1f}MB, 여유: {free:.1f}MB, 전체: {total:.1f}MB, 피크: {peak:.1f}MB")
                except Exception as gpu_e:
                    print(f"[GPU{gpu_idx} MEMORY {stage}] GPU{gpu_idx} 메모리 정보 가져오기 실패: {gpu_e}")
        else:
            print(f"[GPU MEMORY {stage}] GPU 없음")
    except Exception as e:
        print(f"[GPU MEMORY {stage}] 전체 메모리 정보 가져오기 실패: {e}")


class ModelHandler:
    def __init__(self, data, context):
        print_gpu_memory_info("모델 로딩 전")
        
        # 30개의 모델을 모두 로드 (model0.h5 ~ model29.h5)
        self.models = {}
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        print(f"[LOG] 30개 모델 로드 시작...")
        
        for i in range(30):
            model_file = f"model{i}.h5"
            model_path = os.path.join(base_path, model_file)
            
            if os.path.exists(model_path):
                try:
                    self.models[i] = load_model(model_path)
                    print(f"[LOG] {model_file} 로드 성공")
                    if i < 10:  # 처음 10개만 GPU 메모리 모니터링
                        print_gpu_memory_info(f"모델{i} 로딩 후")
                except Exception as e:
                    print(f"[ERROR] {model_file} 로드 실패: {e}")
            else:
                print(f"[WARNING] {model_file} 파일이 없습니다.")
        
        print(f"[LOG] 총 {len(self.models)}개 모델 로드 완료")
        print_gpu_memory_info("모델 로딩 완료 후")

    def test(self) :        ## 이후에 추론 하는 모든 데이터에 대해서 빠르게 진행 될 수 있도록 하는 부분
        # 모든 모델에 대해 test 실행
        for i, model in self.models.items():
            try:
                first_layer = model.layers[0]
                input_shape = first_layer.input_shape[0][1:]
                data = np.zeros(input_shape)    # input_shape[0],input_shape[1],3
                data = np.expand_dims(data,0)   ## 1,input_shape[0],input_shape[1],3
                model.predict(data,batch_size=1)
                print(f"[LOG] model{i}.h5 test 완료")
            except Exception as e:
                print(f"[ERROR] model{i}.h5 test 실패: {e}")

    def get_current_model(self):
        """현재 시간의 소수점 이하 부분을 *100해서 %30으로 모델 선택 (0~29)"""
        current_time = time.time()
        decimal_part = current_time - int(current_time)  # 소수점 이하 부분
        decimal_x100 = int(decimal_part * 100)
        selected_model_idx = decimal_x100 % 30
        
        print(f"[LOG] 소수점 이하: {decimal_part:.4f} -> {decimal_x100} % 30 = {selected_model_idx} -> model{selected_model_idx}.h5 사용")
        
        if selected_model_idx in self.models:
            return self.models[selected_model_idx]
        else:
            # 해당 모델이 없으면 첫 번째 사용 가능한 모델 사용
            if self.models:
                fallback_idx = list(self.models.keys())[0]
                print(f"[WARNING] model{selected_model_idx}.h5가 없어서 model{fallback_idx}.h5 사용")
                return self.models[fallback_idx]
            else:
                raise ValueError("사용 가능한 모델이 없습니다.")

    def __call__(self, data, context):
        print_gpu_memory_info("모델 추론 전")
        
        ## Expand_dims
        data = pickle.loads(data)
        images = np.array(data) # input_shape[0],input_shape[1],3
        
        # Validate input image
        if len(images.shape) != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got shape: {images.shape}")
        
        # 현재 시간에 따라 모델 선택
        current_model = self.get_current_model()
        
        # Get model's expected input shape dynamically
        first_layer = current_model.layers[0]
        expected_height, expected_width = first_layer.input_shape[0][1:3]
        
        # Resize images to match model's expected input size
        if images.shape[:2] != (expected_height, expected_width):
            print(f"Resizing image from {images.shape[:2]} to ({expected_height}, {expected_width})")
            images = cv2.resize(images, (expected_width, expected_height))
        
        images = np.expand_dims(images,0) ## 1,input_shape[0],input_shape[1],3
        result = {}
        
        ## Label Data Load
        label = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"label.txt"),'r')
        label_list = label.read().replace('\n','')
        label_list = label_list.split(',')

        ## Predict
        predict_start_time = time.time()
        predict_result = current_model.predict(images)
        predict_end_time = time.time()
        
        predict_duration = predict_end_time - predict_start_time
        print(f"[PREDICT TIME] 추론 시간: {predict_duration:.4f}초")
        
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
    img = cv2.resize(img,(224,224))
    data = pickle.dumps(img)
    ModelHandler(None, None).__call__(data=data,context={}) 