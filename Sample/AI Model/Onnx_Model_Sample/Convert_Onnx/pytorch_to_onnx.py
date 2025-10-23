import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort
import numpy as np
import os
import argparse

def create_sample_model():
    """샘플 PyTorch 모델 생성 (실제 사용시에는 본인의 모델로 교체)"""
    model = models.resnet50(pretrained=False)
    # 마지막 레이어를 9개 클래스로 수정
    model.fc = nn.Linear(model.fc.in_features, 9)
    return model

def pytorch_to_onnx(pytorch_model, onnx_path, input_shape=(1, 3, 299, 299), 
                    dynamic_axes=None, opset_version=11):
    """
    PyTorch 모델을 ONNX로 변환
    
    Args:
        pytorch_model: PyTorch 모델
        onnx_path: 저장할 ONNX 파일 경로
        input_shape: 입력 텐서의 형태
        dynamic_axes: 동적 축 설정
        opset_version: ONNX opset 버전
    """
    
    # 모델을 평가 모드로 설정
    pytorch_model.eval()
    
    # 더미 입력 생성
    dummy_input = torch.randn(input_shape)
    
    # 동적 축이 설정되지 않은 경우 기본값 설정
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # ONNX로 변환
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"PyTorch 모델이 {onnx_path}로 변환되었습니다.")

def verify_onnx_model(onnx_path, pytorch_model, input_shape=(1, 3, 299, 299)):
    """
    변환된 ONNX 모델 검증
    
    Args:
        onnx_path: ONNX 모델 파일 경로
        pytorch_model: 원본 PyTorch 모델
        input_shape: 입력 텐서의 형태
    """
    
    # ONNX 모델 로드
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 모델 검증 완료")
    
    # ONNX Runtime으로 추론
    ort_session = ort.InferenceSession(onnx_path)
    
    # 더미 입력 생성
    dummy_input = torch.randn(input_shape)
    
    # PyTorch 추론
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input)
    
    # ONNX 추론
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # 출력 비교
    pytorch_output_np = pytorch_output.numpy()
    
    # 정확도 비교 (소수점 5자리까지)
    np.testing.assert_allclose(
        ort_output, pytorch_output_np, rtol=1e-05, atol=1e-05
    )
    print("PyTorch와 ONNX 출력이 일치합니다.")

def main():
    parser = argparse.ArgumentParser(description='PyTorch 모델을 ONNX로 변환')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='PyTorch 모델 파일 경로 (.pth)')
    parser.add_argument('--onnx_path', type=str, default='model.onnx',
                       help='저장할 ONNX 파일 경로')
    parser.add_argument('--input_shape', type=str, default='1,3,299,299',
                       help='입력 텐서 형태 (쉼표로 구분)')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset 버전')
    parser.add_argument('--verify', action='store_true',
                       help='변환 후 모델 검증 수행')
    
    args = parser.parse_args()
    
    # 입력 형태 파싱
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # 모델 로드 또는 생성
    if args.model_path and os.path.exists(args.model_path):
        print(f"PyTorch 모델을 로드합니다: {args.model_path}")
        model = torch.load(args.model_path, map_location='cpu', weights_only=False)
        if isinstance(model, dict):
            # state_dict인 경우
            if 'model' in model:
                model = model['model']
            else:
                # 첫 번째 키를 사용
                first_key = list(model.keys())[0]
                model = model[first_key]
    else:
        print("샘플 PyTorch 모델을 생성합니다.")
        model = create_sample_model()
    
    # ONNX로 변환
    pytorch_to_onnx(model, args.onnx_path, input_shape, opset_version=args.opset_version)
    
    # 검증 수행
    if args.verify:
        verify_onnx_model(args.onnx_path, model, input_shape)

if __name__ == "__main__":
    main()

