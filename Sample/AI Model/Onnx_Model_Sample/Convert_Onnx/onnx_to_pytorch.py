import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os
import argparse
from onnx2pytorch import ConvertModel

def onnx_to_pytorch(onnx_path, pytorch_path=None):
    """
    ONNX 모델을 PyTorch로 변환
    
    Args:
        onnx_path: ONNX 모델 파일 경로
        pytorch_path: 저장할 PyTorch 모델 파일 경로 (None이면 반환만)
        
    Returns:
        PyTorch 모델
    """
    
    # ONNX 모델 로드
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 모델 검증 완료")
    
    # ONNX를 PyTorch로 변환
    pytorch_model = ConvertModel(onnx_model)
    print("ONNX 모델이 PyTorch로 변환되었습니다.")
    
    # PyTorch 모델을 평가 모드로 설정
    pytorch_model.eval()
    
    # 모델 저장
    if pytorch_path:
        torch.save(pytorch_model, pytorch_path)
        print(f"PyTorch 모델이 {pytorch_path}로 저장되었습니다.")
    
    return pytorch_model

def verify_conversion(onnx_path, pytorch_model, input_shape=(1, 3, 299, 299)):
    """
    변환된 PyTorch 모델 검증
    
    Args:
        onnx_path: 원본 ONNX 모델 파일 경로
        pytorch_model: 변환된 PyTorch 모델
        input_shape: 입력 텐서의 형태
    """
    
    # ONNX Runtime으로 추론
    ort_session = ort.InferenceSession(onnx_path)
    
    # 더미 입력 생성
    dummy_input = np.random.random(input_shape).astype(np.float32)
    
    # ONNX 추론
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # PyTorch 추론
    with torch.no_grad():
        pytorch_input = torch.from_numpy(dummy_input)
        pytorch_output = pytorch_model(pytorch_input)
        pytorch_output_np = pytorch_output.numpy()
    
    # 출력 비교
    np.testing.assert_allclose(
        ort_output, pytorch_output_np, rtol=1e-05, atol=1e-05
    )
    print("ONNX와 PyTorch 출력이 일치합니다.")

def main():
    parser = argparse.ArgumentParser(description='ONNX 모델을 PyTorch로 변환')
    parser.add_argument('--onnx_path', type=str, required=True,
                       help='ONNX 모델 파일 경로')
    parser.add_argument('--pytorch_path', type=str, default=None,
                       help='저장할 PyTorch 모델 파일 경로')
    parser.add_argument('--input_shape', type=str, default='1,3,299,299',
                       help='입력 텐서 형태 (쉼표로 구분)')
    parser.add_argument('--verify', action='store_true',
                       help='변환 후 모델 검증 수행')
    
    args = parser.parse_args()
    
    # 입력 형태 파싱
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # 파일 존재 확인
    if not os.path.exists(args.onnx_path):
        print(f"ONNX 모델 파일을 찾을 수 없습니다: {args.onnx_path}")
        return
    
    # ONNX를 PyTorch로 변환
    pytorch_model = onnx_to_pytorch(args.onnx_path, args.pytorch_path)
    
    # 검증 수행
    if args.verify:
        verify_conversion(args.onnx_path, pytorch_model, input_shape)

if __name__ == "__main__":
    main()

