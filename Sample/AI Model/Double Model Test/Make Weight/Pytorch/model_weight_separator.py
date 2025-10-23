import torch
import os
import argparse
import logging
from collections import OrderedDict

# Logging 설정
logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def verify_convnext_large(state_dict):
    """
    state_dict가 ConvNeXt Large 384 모델의 것인지 확인합니다.
    """
    # ConvNeXt Large의 주요 특징 확인
    expected_layers = {
        'downsample_layers.0.0.weight',  # First downsampling layer
        'stages.0.0.layer_scale',        # First stage
        'stages.3.2.layer_scale',        # Last stage (3rd stage, 2nd block)
        'norm.weight',                   # Final normalization
        'head.fc.weight'                 # Classification head
    }
    
    # 주요 레이어들이 있는지 확인
    missing_layers = expected_layers - set(state_dict.keys())
    if missing_layers:
        logger.warning(f"ConvNeXt Large 모델의 주요 레이어가 누락되었습니다: {missing_layers}")
        return False
        
    # 입력 크기 확인 (384x384)
    if 'downsample_layers.0.0.weight' in state_dict:
        patch_size = state_dict['downsample_layers.0.0.weight'].shape[-1]
        if patch_size != 4:  # ConvNeXt는 4x4 패치 사용
            logger.warning(f"패치 크기가 일치하지 않습니다: {patch_size}x{patch_size} (예상: 4x4)")
            return False
    
    return True

def clean_state_dict(state_dict):
    """
    state_dict를 정리하고 필요한 경우 키 이름을 수정합니다.
    """
    new_state_dict = OrderedDict()
    
    for key, value in state_dict.items():
        # module. 접두사 제거 (DataParallel 사용 시 생기는 접두사)
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value
    
    return new_state_dict

def get_model_info(state_dict):
    """
    ConvNeXt Large 모델의 상세 정보를 추출합니다.
    """
    info = {
        'name': 'ConvNeXt Large 384',
        'total_params': sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)),
        'layers': len(state_dict)
    }
    
    # 스테이지 수 계산
    stages = set()
    for key in state_dict.keys():
        if key.startswith('stages.'):
            stage_num = int(key.split('.')[1])
            stages.add(stage_num)
    info['num_stages'] = len(stages)
    
    # 출력 클래스 수 계산
    if 'head.fc.weight' in state_dict:
        info['num_classes'] = state_dict['head.fc.weight'].shape[0]
    
    return info

def extract_weights(input_path, output_dir):
    """
    ConvNeXt Large 384 모델에서 가중치를 추출합니다.
    """
    logger.info(f"ConvNeXt Large 384 가중치 추출 시작: {input_path}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 모델 로드
        loaded_obj = torch.load(input_path, map_location='cpu')
        logger.info(f"파일 로드 완료: {type(loaded_obj)}")
        
        # state_dict 형태로 변환
        if isinstance(loaded_obj, dict):
            state_dict = loaded_obj
        else:
            state_dict = loaded_obj.state_dict()
        
        # 2. ConvNeXt Large 384 모델 검증
        if not verify_convnext_large(state_dict):
            logger.error("이 파일은 ConvNeXt Large 384 모델이 아닙니다!")
            return False
        
        # 3. 가중치 정리
        cleaned_state_dict = clean_state_dict(state_dict)
        
        # 4. 모델 정보 수집
        model_info = get_model_info(cleaned_state_dict)
        
        # 5. 가중치 저장
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        weight_path = os.path.join(output_dir, f'{base_name}_weights.pth')
        torch.save(cleaned_state_dict, weight_path)
        
        # 6. 정보 출력
        logger.info(f"\n=== ConvNeXt Large 384 가중치 추출 완료 ===")
        logger.info(f"모델: {model_info['name']}")
        logger.info(f"총 파라미터 수: {model_info['total_params']:,}")
        logger.info(f"총 레이어 수: {model_info['layers']}")
        logger.info(f"스테이지 수: {model_info['num_stages']}")
        if 'num_classes' in model_info:
            logger.info(f"출력 클래스 수: {model_info['num_classes']}")
        
        # 텐서 통계
        tensor_count = sum(1 for v in cleaned_state_dict.values() if isinstance(v, torch.Tensor))
        other_count = len(cleaned_state_dict) - tensor_count
        logger.info(f"\n레이어 구성:")
        logger.info(f"- 텐서 개수: {tensor_count}")
        if other_count > 0:
            logger.info(f"- 기타 값 개수: {other_count}")
            
        logger.info(f"\n생성된 파일: {weight_path}")
        logger.info(f"파일 크기: {os.path.getsize(weight_path) / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"가중치 추출 실패: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='ConvNeXt Large 384 모델 가중치 추출 도구')
    parser.add_argument('--input', '-i', required=True, help='입력 모델 파일 경로 (.pth)')
    parser.add_argument('--output', '-o', required=True, help='출력 디렉토리 경로')
    
    args = parser.parse_args()
    
    # 입력 파일 검증
    if not os.path.exists(args.input):
        logger.error(f"입력 파일을 찾을 수 없습니다: {args.input}")
        return
    
    # 가중치 추출 실행
    success = extract_weights(args.input, args.output)
    
    if success:
        logger.info("가중치 추출이 성공적으로 완료되었습니다.")
    else:
        logger.error("가중치 추출에 실패했습니다.")

if __name__ == "__main__":
    main() 