import os
import pickle
import cv2
import numpy as np
import time
from model_handler import ModelHandler

def quick_decimal_test():
    """소수점 이하 부분을 사용한 모델 선택 테스트"""
    print("="*50)
    print("Two Model Test - 소수점 이하 부분 모델 선택 테스트")
    print("="*50)
    
    # 더미 이미지 생성
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_img[:, :, 0] = 100
    dummy_img[:, :, 1] = 150
    dummy_img[:, :, 2] = 200
    
    data = pickle.dumps(dummy_img)
    
    try:
        print("ModelHandler 초기화 중...")
        handler = ModelHandler(None, None)
        
        print("\n5번 테스트 (0.1초 간격):")
        for i in range(5):
            print(f"\n--- {i+1}번째 테스트 ---")
            
            current_time = time.time()
            decimal_part = current_time - int(current_time)
            decimal_x100 = int(decimal_part * 100)
            expected_model = decimal_x100 % 2
            
            print(f"현재 시간: {current_time:.4f}")
            print(f"소수점 이하: {decimal_part:.4f}")
            print(f"decimal_x100: {decimal_x100}")
            print(f"예상 모델: {expected_model} ({'model.h5' if expected_model == 1 else 'model2.h5'})")
            
            try:
                result, context = handler.__call__(data, {})
                result_dict = pickle.loads(result)
                print(f"예측 결과: {result_dict['result_code']}, 점수: {result_dict['score']:.4f}")
            except Exception as e:
                print(f"예측 실패: {e}")
            
            time.sleep(0.1)
            
        # 특정 소수점 시뮬레이션
        print("\n특정 소수점 시뮬레이션:")
        test_decimals = [0.123, 0.456, 0.789, 0.012, 0.999]
        for decimal in test_decimals:
            original_time = time.time
            time.time = lambda: 1234 + decimal
            
            try:
                current_model = handler.get_current_model()
                decimal_x100 = int(decimal * 100)
                expected_model = decimal_x100 % 2
                print(f"소수점 {decimal:.3f} -> {decimal_x100} % 2 = {expected_model}")
            except Exception as e:
                print(f"소수점 {decimal:.3f} -> 오류: {e}")
            finally:
                time.time = original_time
                
    except Exception as e:
        print(f"테스트 실패: {e}")

if __name__ == "__main__":
    quick_decimal_test()
    print("\n테스트 완료!") 