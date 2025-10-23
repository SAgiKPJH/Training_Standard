import torch
import torch.nn as nn
import os
import argparse
from collections import OrderedDict
import torchvision.models as models
import json
import logging

# Logging 설정
logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def extract_model_info(model_path):
    """
    PyTorch 모델 파일에서 모델 정보를 추출합니다.
    """
    try:
        # 모델 파일 로드
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 체크포인트 구조 확인
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                model_info = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                model_info = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
            else:
                # 전체가 state_dict인 경우
                state_dict = checkpoint
                model_info = {}
        else:
            # 모델 객체 자체인 경우
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
            model_info = {}
        
        return state_dict, model_info
    except Exception as e:
        logger.info(f"모델 로드 실패: {e}")
        return None, None

def analyze_model_architecture(state_dict):
    """
    State dict에서 모델 아키텍처 정보를 추출합니다.
    """
    architecture_info = {
        'layer_names': list(state_dict.keys()),
        'layer_shapes': {k: list(v.shape) for k, v in state_dict.items()},
        'total_parameters': sum(v.numel() for v in state_dict.values()),
        'layer_types': {}
    }
    
    # 레이어 타입 추정 (간단한 휴리스틱)
    for name, tensor in state_dict.items():
        if 'weight' in name:
            if len(tensor.shape) == 4:  # Conv2d
                architecture_info['layer_types'][name] = 'Conv2d'
            elif len(tensor.shape) == 2:  # Linear
                architecture_info['layer_types'][name] = 'Linear'
            elif len(tensor.shape) == 1:  # BatchNorm or bias
                architecture_info['layer_types'][name] = 'BatchNorm/Bias'
        elif 'bias' in name:
            architecture_info['layer_types'][name] = 'Bias'
        elif 'running_mean' in name or 'running_var' in name:
            architecture_info['layer_types'][name] = 'BatchNorm_Stats'
    
    return architecture_info

def separate_model_weights(input_path, output_dir):
    """
    PyTorch 모델에서 네트워크 구조와 가중치를 분리합니다.
    """
    logger.info(f"모델 분리 시작: {input_path}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 정보 추출
    state_dict, model_info = extract_model_info(input_path)
    
    if state_dict is None:
        logger.info("모델 로드 실패")
        return False
    
    # 아키텍처 정보 분석
    architecture_info = analyze_model_architecture(state_dict)
    
    # 1. 네트워크 구조 정보 저장 (model.pth)
    model_structure = {
        'architecture_info': architecture_info,
        'model_info': model_info,
        'input_model_path': input_path
    }
    
    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(model_structure, model_path)
    logger.info(f"모델 구조 저장 완료: {model_path}")
    
    # 2. 가중치 정보만 저장 (weight.pth)
    weight_path = os.path.join(output_dir, 'weight.pth')
    torch.save(state_dict, weight_path)
    logger.info(f"가중치 저장 완료: {weight_path}")
    
    # 3. 아키텍처 정보를 JSON으로도 저장
    architecture_json_path = os.path.join(output_dir, 'architecture_info.json')
    with open(architecture_json_path, 'w', encoding='utf-8') as f:
        json.dump(architecture_info, f, indent=2, ensure_ascii=False)
    logger.info(f"아키텍처 정보 저장 완료: {architecture_json_path}")
    
    # 4. 분리 정보 요약
    logger.info(f"=== 모델 분리 완료 ===")
    logger.info(f"총 파라미터 수: {architecture_info['total_parameters']:,}")
    logger.info(f"레이어 수: {len(architecture_info['layer_names'])}")
    logger.info(f"생성된 파일:")
    logger.info(f"  - {model_path}")
    logger.info(f"  - {weight_path}")
    logger.info(f"  - {architecture_json_path}")
    
    return True

def load_separated_model(model_path, weight_path):
    """
    분리된 모델 구조와 가중치를 로드합니다.
    """
    try:
        # 모델 구조 정보 로드
        model_structure = torch.load(model_path, map_location='cpu')
        
        # 가중치 로드
        state_dict = torch.load(weight_path, map_location='cpu')
        
        logger.info("분리된 모델 로드 완료")
        logger.info(f"총 파라미터 수: {model_structure['architecture_info']['total_parameters']:,}")
        
        return model_structure, state_dict
    except Exception as e:
        logger.info(f"분리된 모델 로드 실패: {e}")
        return None, None

# python ./torch_model_weight_separator.py --input=./model0.pth --output=.
def main():
    parser = argparse.ArgumentParser(description='PyTorch 모델 네트워크/가중치 분리 도구')
    parser.add_argument('--input', '-i', required=True, help='입력 모델 파일 경로 (.pth)')
    parser.add_argument('--output', '-o', required=True, help='출력 디렉토리 경로')
    parser.add_argument('--test', '-t', action='store_true', help='분리 후 로드 테스트')
    
    args = parser.parse_args()
    
    # 입력 파일 검증
    if not os.path.exists(args.input):
        logger.info(f"입력 파일을 찾을 수 없습니다: {args.input}")
        return
    
    # 모델 분리 실행
    success = separate_model_weights(args.input, args.output)
    
    if success and args.test:
        logger.info("\n=== 분리된 모델 로드 테스트 ===")
        model_path = os.path.join(args.output, 'model.pth')
        weight_path = os.path.join(args.output, 'weight.pth')
        
        model_structure, state_dict = load_separated_model(model_path, weight_path)
        
        if model_structure is not None and state_dict is not None:
            logger.info("테스트 성공: 분리된 모델을 정상적으로 로드했습니다.")
        else:
            logger.info("테스트 실패: 분리된 모델 로드에 실패했습니다.")

if __name__ == "__main__":
    main() 