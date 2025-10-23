import os
import pickle
import cv2
import numpy as np
import time
from model_handler import ModelHandler

def quick_decimal_test():
    """소수점 이하 부분을 사용한 모델 선택 빠른 테스트"""
    print("="*50)
    print("소수점 이하 부분 모델 선택 빠른 테스트")
    print("="*50)
    
    # 더미 이미지 생성 (224x224x3)
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_img[:, :, 0] = 100  # 빨간색 채널
    dummy_img[:, :, 1] = 150  # 초록색 채널
    dummy_img[:, :, 2] = 200  # 파란색 채널
    
    # 이미지 데이터를 pickle로 직렬화
    data = pickle.dumps(dummy_img)
    
    try:
        # ModelHandler 초기화
        print("ModelHandler 초기화 중...")
        handler = ModelHandler(None, None)
        
        # 5번 빠른 테스트 (0.05초 간격)
        print("\n5번 빠른 테스트 (0.05초 간격):")
        for i in range(5):
            print(f"\n--- {i+1}번째 테스트 ---")
            
            # 현재 시간 출력
            current_time = time.time()
            decimal_part = current_time - int(current_time)
            decimal_x100 = int(decimal_part * 100)
            selected_model = decimal_x100 % 100
            print(f"현재 시간: {current_time:.4f}")
            print(f"소수점 이하: {decimal_part:.4f} -> {decimal_x100} % 100 = {selected_model}")
            print(f"선택된 모델: model{selected_model}.h5")
            
            # 모델 예측 실행
            try:
                result, context = handler.__call__(data, {})
                result_dict = pickle.loads(result)
                print(f"예측 결과: {result_dict['result_code']}, 점수: {result_dict['score']:.4f}")
            except Exception as e:
                print(f"예측 실패: {e}")
            
            # 0.05초 대기
            time.sleep(0.05)
            
        # 특정 소수점 시뮬레이션
        print("\n특정 소수점 시뮬레이션:")
        test_decimals = [0.123, 0.456, 0.789, 0.012, 0.999]
        for decimal in test_decimals:
            original_time = time.time
            time.time = lambda: 1234 + decimal
            
            try:
                current_model = handler.get_current_model()
                decimal_x100 = int(decimal * 100)
                expected_model = decimal_x100 % 100
                print(f"소수점 {decimal:.3f} -> {decimal_x100} % 100 = {expected_model}")
            except Exception as e:
                print(f"소수점 {decimal:.3f} -> 오류: {e}")
            finally:
                time.time = original_time
                
    except Exception as e:
        print(f"테스트 실패: {e}")

if __name__ == "__main__":
    quick_decimal_test()
    print("\n빠른 테스트 완료!") 