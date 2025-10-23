import torch
import os
import argparse
import logging

# Logging 설정
logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def extract_weights(input_path, output_dir):
    """
    PyTorch 모델에서 가중치만 추출합니다.
    """
    logger.info(f"가중치 추출 시작: {input_path}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 모델 로드
        model = torch.load(input_path, map_location='cpu', weights_only=False)
        logger.info(f"모델 로드 완료: {type(model)}")
        
        # 2. 가중치 저장
        weight_path = os.path.join(output_dir, 'weights.pth')
        state_dict = model.state_dict()
        torch.save(state_dict, weight_path)
        logger.info(f"가중치 저장 완료: {weight_path}")
        
        # 3. 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"=== 가중치 추출 완료 ===")
        logger.info(f"모델 클래스: {model.__class__.__name__}")
        logger.info(f"총 파라미터 수: {total_params:,}")
        logger.info(f"생성된 파일: {weight_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"가중치 추출 실패: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='PyTorch 모델 가중치 추출 도구')
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