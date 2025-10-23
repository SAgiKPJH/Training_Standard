import os
import sys
import numpy as np
import pickle
import time
from model_handler import ModelHandler, print_gpu_memory_info

def simple_test():
    """간단한 모델 로딩 테스트"""
    
    print("="*50)
    print("FIFTY MODEL TEST - 간단한 모델 로딩 테스트")
    print("="*50)
    
    # 시작 전 GPU 메모리 상태
    print_gpu_memory_info("테스트 시작 전")
    
    # ModelHandler 초기화 (모든 모델 로드)
    print("\n[테스트] ModelHandler 초기화 중...")
    handler = ModelHandler(None, None)
    
    # 로드된 모델 개수 확인
    print(f"\n[결과] 로드된 모델 개수: {len(handler.models)}/50")
    
    # 로드된 모델 인덱스 확인
    loaded_models = sorted(handler.models.keys())
    print(f"[결과] 로드된 모델 인덱스: {loaded_models[:10]}{'...' if len(loaded_models) > 10 else ''}")
    
    # 모델 선택 테스트
    print("\n[테스트] 모델 선택 테스트...")
    for i in range(5):
        try:
            current_model = handler.get_current_model()
            print(f"[결과] 모델 선택 성공")
            time.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] 모델 선택 실패: {e}")
    
    # 최종 GPU 메모리 상태
    print_gpu_memory_info("테스트 완료 후")
    
    print("\n" + "="*50)
    print("테스트 완료!")
    print("="*50)

if __name__ == "__main__":
    simple_test() 