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
        # 1. 모델 로드 시도 (여러 방법 시도)
        model = None
        
        # 첫 번째 시도: 일반적인 로드
        try:
            model = tf.keras.models.load_model(input_path)
            logger.info(f"모델 로드 완료: {type(model)}")
        except Exception as e:
            logger.info(f"일반 로드 실패, 대안 방법 시도: {e}")
            
            # 두 번째 시도: compile=False로 로드
            try:
                model = tf.keras.models.load_model(input_path, compile=False)
                logger.info(f"모델 로드 완료 (compile=False): {type(model)}")
            except Exception as e2:
                logger.info(f"compile=False 로드도 실패: {e2}")
                
                # 세 번째 시도: custom_objects 사용
                try:
                    # DepthwiseConv2D의 groups 파라미터 문제 해결
                    custom_objects = {
                        'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D
                    }
                    model = tf.keras.models.load_model(input_path, custom_objects=custom_objects, compile=False)
                    logger.info(f"모델 로드 완료 (custom_objects): {type(model)}")
                except Exception as e3:
                    logger.error(f"모든 로드 방법 실패: {e3}")
                    return False
        
        if model is None:
            logger.error("모델 로드에 실패했습니다.")
            return False
        
        # 2. 모델을 폴더로 저장
        model_folder_path = os.path.join(output_dir, 'model')
        
        try:
            model.save(model_folder_path)
            logger.info(f"모델 폴더 저장 완료: {model_folder_path}/")
        except Exception as e:
            logger.info(f"일반 저장 실패, 대안 방법 시도: {e}")
            
            # 저장 옵션 조정
            try:
                model.save(model_folder_path, save_format='tf')
                logger.info(f"모델 폴더 저장 완료 (tf 형식): {model_folder_path}/")
            except Exception as e2:
                logger.error(f"저장 실패: {e2}")
                return False
        
        # 3. 정보 출력
        total_params = model.count_params()
        layer_count = len(model.layers)
        logger.info(f"=== 변환 완료 ===")
        logger.info(f"총 파라미터 수: {total_params:,}")
        logger.info(f"레이어 수: {layer_count}")
        logger.info(f"생성된 폴더: {model_folder_path}/")
        
        return True
        
    except Exception as e:
        logger.error(f"모델 변환 실패: {e}")
        return False

def load_model_from_folder(model_folder_path):
    """
    폴더에서 모델을 로드합니다.
    """
    try:
        # 여러 방법으로 로드 시도
        model = None
        
        try:
            model = tf.keras.models.load_model(model_folder_path)
            logger.info(f"폴더에서 모델 로드 완료")
        except Exception as e:
            logger.info(f"일반 로드 실패, compile=False 시도: {e}")
            model = tf.keras.models.load_model(model_folder_path, compile=False)
            logger.info(f"폴더에서 모델 로드 완료 (compile=False)")
        
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
    
    # TensorFlow 버전 정보
    logger.info(f"TensorFlow 버전: {tf.__version__}")
    
    # 모델 변환 실행
    success = save_model_as_folder(args.input, args.output)
    
    if success:
        logger.info("변환이 성공적으로 완료되었습니다.")
        
        if args.test:
            logger.info("\n=== 폴더 모델 로드 테스트 ===")
            model_folder_path = os.path.join(args.output, 'model')
            
            model = load_model_from_folder(model_folder_path)
            
            if model is not None:
                logger.info("테스트 성공: 폴더 모델을 정상적으로 로드했습니다.")
                logger.info(f"모델 파라미터 수: {model.count_params():,}")
            else:
                logger.error("테스트 실패: 폴더 모델 로드에 실패했습니다.")
    else:
        logger.error("변환에 실패했습니다.")

if __name__ == "__main__":
    main() 