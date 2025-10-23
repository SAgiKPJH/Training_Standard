import tensorflow as tf
import os
import argparse
import logging

# Logging 설정
logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def save_model_as_folder(input_path, output_dir):
    """
    TensorFlow 모델을 폴더 형식으로 저장합니다.
    """
    logger.info(f"모델 변환 시작: {input_path}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 모델 로드
        model = tf.keras.models.load_model(input_path)
        logger.info(f"모델 로드 완료: {type(model)}")
        model.save('saved_model_dir')
        
    except Exception as e:
        logger.error(f"모델 변환 실패: {e}")
        return False

def load_model_from_folder(model_folder_path):
    """
    폴더에서 모델을 로드합니다.
    """
    try:
        model = tf.keras.models.load_model(model_folder_path)
        logger.info(f"폴더에서 모델 로드 완료")
        return model
        
    except Exception as e:
        logger.error(f"폴더에서 모델 로드 실패: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='TensorFlow 모델을 폴더 형식으로 변환')
    parser.add_argument('--input', '-i', required=True, help='입력 모델 파일 경로 (.h5)')
    parser.add_argument('--output', '-o', required=True, help='출력 디렉토리 경로')
    parser.add_argument('--test', '-t', action='store_true', help='변환 후 로드 테스트')
    
    args = parser.parse_args()
    
    # 입력 파일 검증
    if not os.path.exists(args.input):
        logger.error(f"입력 파일을 찾을 수 없습니다: {args.input}")
        return
    
    # 모델 변환 실행
    success = save_model_as_folder(args.input, args.output)
    
    if success:
        logger.info("변환이 성공적으로 완료되었습니다.")
    else:
        logger.error("변환에 실패했습니다.")

if __name__ == "__main__":
    main() 