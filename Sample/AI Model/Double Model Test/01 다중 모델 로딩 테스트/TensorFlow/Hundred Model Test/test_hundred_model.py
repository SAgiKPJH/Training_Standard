import os
import pickle
import cv2
import numpy as np
import time
from model_handler import ModelHandler

def test_hundred_model():
    """Hundred Model Test 기능 테스트"""
    print("="*60)
    print("Hundred Model Test - 밀리초에 따른 100개 모델 선택 테스트")
    print("="*60)
    
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
        
        # 10번 테스트 (0.1초 간격으로 밀리초 단위 모델 선택 테스트)
        print("\n10번 테스트 (0.1초 간격으로 밀리초 단위 모델 선택 테스트):")
        for i in range(10):
            print(f"\n--- {i+1}번째 테스트 ---")
            
            # 현재 시간 출력
            current_time = time.time()
            current_millisecond = int(current_time * 1000) % 1000
            selected_model = int(current_time * 1000) % 100
            print(f"현재 시간: {current_millisecond}ms (전체: {int(current_time * 1000)}) -> 예상 모델: model{selected_model}.h5")
            
            # 모델 예측 실행
            try:
                result, context = handler.__call__(data, {})
                result_dict = pickle.loads(result)
                print(f"예측 결과: {result_dict['result_code']}, 점수: {result_dict['score']:.4f}")
            except Exception as e:
                print(f"예측 실패: {e}")
            
            # 0.1초 대기
            time.sleep(0.1)
            
    except Exception as e:
        print(f"테스트 실패: {e}")

def test_specific_hundred_milliseconds():
    """특정 밀리초에 대한 100개 모델 선택 테스트"""
    print("\n" + "="*60)
    print("특정 밀리초에 대한 100개 모델 선택 시뮬레이션")
    print("="*60)
    
    # 더미 이미지 생성
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    data = pickle.dumps(dummy_img)
    
    try:
        handler = ModelHandler(None, None)
        
        # 0~99 밀리초 시뮬레이션 (샘플링)
        print("0~99 밀리초에 대한 모델 선택 시뮬레이션 (20개 샘플):")
        test_ms = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
        
        for test_time_ms in test_ms:
            # 시간을 강제로 설정하여 테스트 (밀리초를 초로 변환)
            original_time = time.time
            time.time = lambda: test_time_ms / 1000.0  # 밀리초를 초로 변환
            
            try:
                current_model = handler.get_current_model()
                expected_model = test_time_ms % 100
                print(f"밀리초 {test_time_ms:2d} -> 선택된 모델: model{expected_model}.h5")
            except Exception as e:
                print(f"밀리초 {test_time_ms:2d} -> 오류: {e}")
            finally:
                time.time = original_time  # 원래 시간 함수 복원
        
        # 100 이상의 경우 순환 테스트
        print("\n100 이상의 밀리초 순환 테스트:")
        test_ms_high = [100, 150, 200, 250, 300, 500, 750, 850, 950, 999]
        for test_time_ms in test_ms_high:
            original_time = time.time
            time.time = lambda: test_time_ms / 1000.0
            
            try:
                current_model = handler.get_current_model()
                expected_model = test_time_ms % 100
                print(f"밀리초 {test_time_ms:3d} -> 선택된 모델: model{expected_model:2d}.h5")
            except Exception as e:
                print(f"밀리초 {test_time_ms:3d} -> 오류: {e}")
            finally:
                time.time = original_time
                
    except Exception as e:
        print(f"시뮬레이션 실패: {e}")

def show_hundred_model_mapping():
    """100개 모델 매핑 표시"""
    print("\n" + "="*60)
    print("Hundred Model Test - 밀리초와 모델 매핑")
    print("="*60)
    
    print("밀리초 → 모델 매핑 (처음 20개 예시):")
    for i in range(20):
        print(f"밀리초 {i:2d} → model{i:2d}.h5")
    
    print("...")
    print("밀리초 → 모델 매핑 (마지막 20개 예시):")
    for i in range(80, 100):
        print(f"밀리초 {i:2d} → model{i:2d}.h5")
    
    print(f"\n순환 패턴:")
    print(f"- 밀리초 0~99 → model0.h5~model99.h5")
    print(f"- 밀리초 100~199 → model0.h5~model99.h5 (순환)")
    print(f"- 밀리초 200~299 → model0.h5~model99.h5 (순환)")
    print(f"- ... (밀리초 % 100으로 모델 선택)")

def performance_test():
    """성능 테스트"""
    print("\n" + "="*60)
    print("성능 테스트 - 100개 모델 로드 및 초기화 시간")
    print("="*60)
    
    # 더미 이미지 생성
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    data = pickle.dumps(dummy_img)
    
    try:
        # 초기화 시간 측정
        start_time = time.time()
        handler = ModelHandler(None, None)
        init_time = time.time() - start_time
        
        print(f"초기화 시간: {init_time:.2f}초")
        print(f"로드된 모델 개수: {len(handler.models)}개")
        
        # 테스트 시간 측정
        start_time = time.time()
        handler.test()
        test_time = time.time() - start_time
        
        print(f"모델 테스트 시간: {test_time:.2f}초")
        
        # 예측 시간 측정 (10회 평균)
        prediction_times = []
        for i in range(10):
            start_time = time.time()
            result, context = handler.__call__(data, {})
            prediction_time = time.time() - start_time
            prediction_times.append(prediction_time)
        
        avg_prediction_time = np.mean(prediction_times)
        print(f"평균 예측 시간: {avg_prediction_time:.4f}초 (10회 평균)")
        
    except Exception as e:
        print(f"성능 테스트 실패: {e}")

if __name__ == "__main__":
    # 모델 매핑 표시
    show_hundred_model_mapping()
    
    # 성능 테스트
    performance_test()
    
    # 기본 테스트 실행
    test_hundred_model()
    
    # 특정 밀리초 테스트 실행
    test_specific_hundred_milliseconds()
    
    print("\n" + "="*60)
    print("모든 테스트 완료!")
    print("="*60) 