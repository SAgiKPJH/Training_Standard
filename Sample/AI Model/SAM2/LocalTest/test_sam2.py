import os
import pickle
import cv2
import numpy as np
import logging
from model_handler import ModelHandler

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def create_test_image(width=512, height=512):
    """테스트용 이미지 생성"""
    # 간단한 도형들이 있는 이미지 생성
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 배경색
    image[:] = (100, 100, 100)
    
    # 원 그리기
    cv2.circle(image, (width//4, height//4), 50, (255, 0, 0), -1)
    
    # 사각형 그리기
    cv2.rectangle(image, (width//2, height//2), (width//2 + 100, height//2 + 100), (0, 255, 0), -1)
    
    # 삼각형 그리기
    pts = np.array([[width*3//4, height//4], [width*3//4 - 50, height//4 + 100], [width*3//4 + 50, height//4 + 100]], np.int32)
    cv2.fillPoly(image, [pts], (0, 0, 255))
    
    return image

def test_sam2_handler():
    """SAM2 핸들러 테스트"""
    try:
        logger.info("SAM2 모델 핸들러 테스트를 시작합니다...")
        
        # 테스트 이미지 생성
        test_image = create_test_image(512, 512)
        logger.info(f"테스트 이미지 생성 완료: {test_image.shape}")
        
        # 이미지를 pickle로 직렬화
        data = pickle.dumps(test_image)
        
        # 모델 핸들러 초기화
        logger.info("모델 핸들러 초기화 중...")
        handler = ModelHandler(None, None)
        
        # 테스트 실행
        logger.info("모델 테스트 실행 중...")
        handler.test()
        
        # Segmentation 수행
        logger.info("Segmentation 수행 중...")
        result, context = handler(data=data, context={})
        
        # 결과 파싱
        result_dict = pickle.loads(result)
        
        # 결과 출력
        logger.info("=== 테스트 결과 ===")
        logger.info(f"처리 시간: {result_dict['processing_time']:.4f}초")
        logger.info(f"원본 이미지 크기: {result_dict['original_shape']}")
        logger.info(f"결과 이미지 크기: {result_dict['segmented_shape']}")
        
        # 결과 이미지 저장
        output_path = os.path.join(os.path.dirname(__file__), "test_output.jpg")
        cv2.imwrite(output_path, result_dict['segmented_image'])
        logger.info(f"결과 이미지 저장: {output_path}")
        
        # 원본 이미지도 저장
        original_path = os.path.join(os.path.dirname(__file__), "test_input.jpg")
        cv2.imwrite(original_path, test_image)
        logger.info(f"입력 이미지 저장: {original_path}")
        
        logger.info("테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        return False

def test_with_real_image(image_path):
    """실제 이미지로 테스트"""
    try:
        if not os.path.exists(image_path):
            logger.error(f"이미지 파일이 존재하지 않습니다: {image_path}")
            return False
        
        logger.info(f"실제 이미지로 테스트: {image_path}")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            logger.error("이미지 로드 실패")
            return False
        
        logger.info(f"이미지 로드 완료: {image.shape}")
        
        # 이미지를 pickle로 직렬화
        data = pickle.dumps(image)
        
        # 모델 핸들러 초기화
        handler = ModelHandler(None, None)
        
        # Segmentation 수행
        result, context = handler(data=data, context={})
        result_dict = pickle.loads(result)
        
        # 결과 출력
        logger.info("=== 실제 이미지 테스트 결과 ===")
        logger.info(f"처리 시간: {result_dict['processing_time']:.4f}초")
        logger.info(f"원본 이미지 크기: {result_dict['original_shape']}")
        logger.info(f"결과 이미지 크기: {result_dict['segmented_shape']}")
        
        # 결과 이미지 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(os.path.dirname(__file__), f"{base_name}_segmented.jpg")
        cv2.imwrite(output_path, result_dict['segmented_image'])
        logger.info(f"결과 이미지 저장: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"실제 이미지 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    # 기본 테스트 실행
    success = test_sam2_handler()
    
    if success:
        logger.info("기본 테스트 성공!")
        
        # 실제 이미지가 있다면 추가 테스트
        # test_with_real_image("path/to/your/image.jpg")
    else:
        logger.error("기본 테스트 실패!")







