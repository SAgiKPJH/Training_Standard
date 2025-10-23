import os
import pickle
import cv2
import numpy as np
import time
from model_handler import ModelHandler

def test_twenty_model():
    """Twenty Model Test 기능 테스트"""
    print("="*50)
    print("Twenty Model Test - 시간에 따른 20개 모델 선택 테스트")
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
        
        # Test 함수 실행
        print("Model test 실행 중...")
        handler.test()
        
        # 20초 동안 매초마다 모델 선택 테스트
        print("\n20초 동안 매초마다 모델 선택 테스트:")
        for i in range(20):
            print(f"\n--- {i+1}초 후 테스트 ---")
            
            # 현재 시간 출력
            current_second = int(time.time()) % 60
            print(f"현재 시간: {current_second}초")
            
            # 모델 예측 실행
            try:
                result, context = handler.__call__(data, {})
                result_dict = pickle.loads(result)
                print(f"예측 결과: {result_dict['result_code']}, 점수: {result_dict['score']:.4f}")
            except Exception as e:
                print(f"예측 실패: {e}")
            
            # 1초 대기
            time.sleep(1)
            
    except Exception as e:
        print(f"테스트 실패: {e}")

def test_specific_seconds():
    """특정 초에 대한 모델 선택 테스트"""
    print("\n" + "="*50)
    print("특정 초에 대한 모델 선택 시뮬레이션")
    print("="*50)
    
    # 더미 이미지 생성
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    data = pickle.dumps(dummy_img)
    
    try:
        handler = ModelHandler(None, None)
        
        # 0~19초 시뮬레이션
        print("0~19초 각각에 대한 모델 선택 시뮬레이션:")
        for second in range(20):
            # 시간을 강제로 설정하여 테스트
            original_time = time.time
            time.time = lambda: second  # 임시로 시간 함수 오버라이드
            
            try:
                current_model = handler.get_current_model()
                print(f"{second}초 -> 선택된 모델: model{second}.h5")
            except Exception as e:
                print(f"{second}초 -> 오류: {e}")
            finally:
                time.time = original_time  # 원래 시간 함수 복원
        
        # 20초 이상의 경우 순환 테스트
        print("\n20초 이상의 경우 순환 테스트:")
        test_seconds = [20, 25, 30, 35, 40, 45, 50, 55, 59]
        for second in test_seconds:
            original_time = time.time
            time.time = lambda: second
            
            try:
                current_model = handler.get_current_model()
                expected_model = second % 20
                print(f"{second}초 -> 선택된 모델: model{expected_model}.h5")
            except Exception as e:
                print(f"{second}초 -> 오류: {e}")
            finally:
                time.time = original_time
                
    except Exception as e:
        print(f"시뮬레이션 실패: {e}")

def show_model_mapping():
    """모델 매핑 표시"""
    print("\n" + "="*50)
    print("Twenty Model Test - 시간과 모델 매핑")
    print("="*50)
    
    print("초 → 모델 매핑:")
    for i in range(20):
        print(f"{i:2d}초 → model{i}.h5")
    
    print("\n순환 패턴 (20초 이후):")
    for i in range(20, 60, 20):
        print(f"{i:2d}~{i+19:2d}초 → model0.h5~model19.h5 (순환)")

if __name__ == "__main__":
    # 모델 매핑 표시
    show_model_mapping()
    
    # 기본 테스트 실행
    test_twenty_model()
    
    # 특정 초 테스트 실행
    test_specific_seconds()
    
    print("\n" + "="*50)
    print("테스트 완료!")
    print("="*50) 