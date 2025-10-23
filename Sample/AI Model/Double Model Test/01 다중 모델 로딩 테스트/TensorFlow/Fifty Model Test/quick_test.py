import os
import sys
import cv2
import numpy as np
import pickle
import time
from model_handler import ModelHandler, print_gpu_memory_info

def quick_test():
    """Fifty Model Test 빠른 테스트 (GPU 메모리 모니터링 포함)"""
    
    print("="*60)
    print("FIFTY MODEL TEST - 빠른 테스트")
    print("="*60)
    
    # 시작 전 GPU 메모리 상태
    print_gpu_memory_info("테스트 시작 전")
    
    # 1. 더미 이미지 생성
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_img[:, :, 0] = 100  # Red channel
    dummy_img[:, :, 1] = 150  # Green channel  
    dummy_img[:, :, 2] = 200  # Blue channel
    
    data = pickle.dumps(dummy_img)
    
    # 2. ModelHandler 초기화 (모든 모델 로드)
    print("\n[테스트] ModelHandler 초기화 중...")
    handler = ModelHandler(None, None)
    
    # 3. 모델 테스트
    print("\n[테스트] 모델 테스트 실행...")
    handler.test()
    
    # 4. 추론 테스트 (10번 실행)
    print("\n[테스트] 추론 테스트 10번 실행...")
    for i in range(10):
        print(f"\n--- 추론 테스트 {i+1}/10 ---")
        
        # 시간 간격을 두어 다른 모델이 선택되도록
        time.sleep(0.1)
        
        try:
            result, context = handler(data, {})
            result_dict = pickle.loads(result)
            
            print(f"[결과] 추론 성공 - 코드: {result_dict['result_code']}, 점수: {result_dict['score']:.4f}")
            
        except Exception as e:
            print(f"[ERROR] 추론 실패: {e}")
    
    # 5. 모델 선택 분포 테스트
    print("\n[테스트] 모델 선택 분포 테스트 (50번 실행)...")
    model_usage = {}
    
    for i in range(50):
        current_time = time.time()
        decimal_part = current_time - int(current_time)
        decimal_x100 = int(decimal_part * 100)
        selected_model_idx = decimal_x100 % 50
        
        model_usage[selected_model_idx] = model_usage.get(selected_model_idx, 0) + 1
        
        # 짧은 대기 시간
        time.sleep(0.02)
    
    print(f"\n[결과] 사용된 모델 개수: {len(model_usage)}/50")
    print(f"[결과] 모델 사용 분포 (처음 10개):")
    for i, (model_idx, count) in enumerate(sorted(model_usage.items())[:10]):
        print(f"  model{model_idx}.h5: {count}회")
    
    # 최종 GPU 메모리 상태
    print_gpu_memory_info("테스트 완료 후")
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)

if __name__ == "__main__":
    quick_test() 