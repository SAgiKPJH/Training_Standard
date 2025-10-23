import os
import pickle
import json
import numpy as np
import torch
import cv2
import time
import logging
from typing import Optional, Tuple, List
import requests
from PIL import Image
import io

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

## GPU Memory 제한
memory_limit = 8 * 1024  # SAM2는 더 많은 메모리가 필요할 수 있음

def print_gpu_memory_info(stage=""):
    """GPU 메모리 사용량 출력"""
    try:
        if torch.cuda.is_available():
            for gpu_idx in range(min(2, torch.cuda.device_count())):
                try:
                    allocated = torch.cuda.memory_allocated(gpu_idx) / (1024**2)
                    reserved = torch.cuda.memory_reserved(gpu_idx) / (1024**2)
                    max_allocated = torch.cuda.max_memory_allocated(gpu_idx) / (1024**2)
                    
                    total = memory_limit
                    free = total - allocated
                    
                    logger.info(f"[GPU{gpu_idx} MEMORY {stage}] 사용: {allocated:.1f}MB, 여유: {free:.1f}MB, 전체: {total:.1f}MB, 피크: {max_allocated:.1f}MB")
                except Exception as gpu_e:
                    logger.info(f"[GPU{gpu_idx} MEMORY {stage}] GPU{gpu_idx} 메모리 정보 가져오기 실패: {gpu_e}")
        else:
            logger.info(f"[GPU MEMORY {stage}] GPU 없음")
    except Exception as e:
        logger.info(f"[GPU MEMORY {stage}] 전체 메모리 정보 가져오기 실패: {e}")

class SAM2Handler:
    """SAM2 모델 핸들러 - 자동 segmentation 수행"""
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"SAM2.1 모델이 {self.device}에서 실행됩니다. (모델: sam2.1_hiera_l)")
        self.model = None
        self.mask_generator = None
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        # SAM2 모델 초기화
        self._load_sam2_model()
        
    def _load_sam2_model(self):
        """SAM2 모델 로드"""
        try:
            print_gpu_memory_info("SAM2 모델 로딩 전")
            
            # SAM2 모델 import 및 로드
            import sam2
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            # SAM2.1 설정 파일 경로 (무조건 large 모델 사용)
            from hydra import initialize, compose
            from hydra.core.global_hydra import GlobalHydra
            GlobalHydra.instance().clear()
            original_cwd = os.getcwd() # 현재 작업 디렉토리를 설정 파일이 있는 디렉토리로 변경
            os.chdir(self.base_path)   #
            initialize(config_path=".", job_name="sam2_inference", version_base=None)
            config_file = "sam2.1_hiera_l.yaml"
            
            # 설정 파일 존재 확인
            if not os.path.exists(config_file):
                logger.warning(f"SAM2 설정 파일이 없습니다: {config_file}")
                logger.info("SAM2 설정 파일을 다운로드하거나 경로를 확인해주세요.")
                raise FileNotFoundError(f"SAM2 설정 파일을 찾을 수 없습니다: {config_file}")
            
            # SAM2.1 모델 체크포인트 경로 (무조건 large 모델 사용)
            checkpoint_path = os.path.join(self.base_path, "sam2.1_hiera_large.pt")
            
            if not os.path.exists(checkpoint_path):
                logger.warning(f"SAM2 체크포인트가 없습니다: {checkpoint_path}")
                logger.info("SAM2 체크포인트를 다운로드하거나 경로를 확인해주세요.")
                logger.info("체크포인트 없이 모델을 로드합니다.")
                # 체크포인트 없이 모델 로드 시도
                self.model = build_sam2(config_file=config_file, device=self.device)
                self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
                logger.info("SAM2 모델 로드 완료 (체크포인트 없음)")
                print_gpu_memory_info("SAM2 모델 로딩 완료 후")
                return
            
            # SAM2 모델 로드
            self.model = build_sam2(config_file=config_file, ckpt_path=checkpoint_path, device=self.device)
            self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
            
            logger.info("SAM2 모델 로드 완료")
            print_gpu_memory_info("SAM2 모델 로딩 완료 후")
            
        except ImportError as e:
            logger.error(f"SAM2 라이브러리 import 실패: {e}")
            logger.info("sam2 패키지를 설치해주세요: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        except Exception as e:
            logger.error(f"SAM2 모델 로드 실패: {e}")
            logger.error(f"에러 타입: {type(e).__name__}")
            logger.error(f"에러 상세: {str(e)}")
    

    
    def segment_image(self, image: np.ndarray) -> dict:
        """이미지 segmentation 수행 - 자동 마스크 생성"""
        if self.mask_generator is None:
            logger.error("SAM2 모델이 로드되지 않았습니다.")
            logger.error("모델 파일을 확인하거나 다운로드해주세요.")
            return {
                'colored_mask': image.copy(),     # 원본 이미지 반환
                'mask': np.zeros(image.shape[:2], dtype=bool),
                'scores': [0.0],
                'num_masks': 0
            }
        
        try:
            # SAM2AutomaticMaskGenerator를 사용한 자동 마스크 생성
            masks = self.mask_generator.generate(image)
            
            if not masks:
                logger.warning("생성된 마스크가 없습니다.")
                return {
                    'colored_mask': image,
                    'mask': np.zeros(image.shape[:2], dtype=bool),
                    'scores': [0.0],
                    'num_masks': 0
                }
            
            # 모든 마스크를 색상화한 이미지 생성
            colored_mask = self._create_colored_mask(masks, image)
            
            return {
                'colored_mask': colored_mask,     # 모든 마스크가 포함된 이미지
                'mask': np.any([m['segmentation'] for m in masks], axis=0),  # 모든 마스크 결합
                'scores': [mask.get('predicted_iou', 0.0) for mask in masks],
                'num_masks': len(masks),
                'all_masks': masks
            }
            
        except Exception as e:
            logger.error(f"Segmentation 실패: {e}")
            return {
                'colored_mask': image,
                'mask': np.zeros(image.shape[:2], dtype=bool),
                'scores': [0.0],
                'num_masks': 0
            }
    
    def _create_colored_mask(self, masks: list, image: np.ndarray) -> np.ndarray:
        """모든 마스크를 색상화하여 이미지와 합성"""
        # 다양한 색상 팔레트 정의
        colors = [
            [255, 0, 0],    # 빨강
            [0, 255, 0],    # 초록
            [0, 0, 255],    # 파랑
            [255, 255, 0],  # 노랑
            [255, 0, 255],  # 마젠타
            [0, 255, 255],  # 시안
            [255, 165, 0],  # 주황
            [128, 0, 128],  # 보라
            [255, 192, 203], # 분홍
            [0, 128, 0],    # 진한 초록
        ]
        
        # 원본 이미지 복사
        result = image.copy().astype(np.float32)
        
        # 모든 마스크 처리
        for i, mask_info in enumerate(masks):
            mask = mask_info['segmentation'].astype(bool)
            color = np.array(colors[i % len(colors)])
            alpha = 0.4  # 마스크 투명도
            result[mask] = (1 - alpha) * result[mask] + alpha * color
        
        return result.astype(np.uint8)

class ModelHandler:
    def __init__(self, data, context):
        print_gpu_memory_info("모델 로딩 전")
        
        # SAM2 핸들러 초기화 (무조건 large 모델 사용)
        self.sam2_handler = SAM2Handler()
        
        print_gpu_memory_info("모델 로딩 완료 후")

    def test(self):
        """모델 테스트 - 워밍업"""
        print_gpu_memory_info("모델 테스트 전")
        
        # 더미 이미지로 테스트
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = self.sam2_handler.segment_image(dummy_image)
        
        logger.info(f"테스트 완료 - 결과 이미지 크기: {result['colored_mask'].shape}")
        logger.info(f"생성된 마스크 개수: {result.get('num_masks', 0)}")
        print_gpu_memory_info("모델 테스트 완료 후")

    def __call__(self, data, context):
        print_gpu_memory_info("모델 추론 전")
        
        # 입력 데이터 로드
        data = pickle.loads(data)
        images = np.array(data)
        
        # 입력 이미지 검증
        if len(images.shape) != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got shape: {images.shape}")
        
        # BGR to RGB 변환
        if images.shape[2] == 3:
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        
        # 이미지 크기 조정 (SAM2는 다양한 크기 지원)
        max_size = 1024
        h, w = images.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            images = cv2.resize(images, (new_w, new_h))
            logger.info(f"이미지 크기 조정: {h}x{w} -> {new_h}x{new_w}")
        
        result = {}
        
        # Segmentation 수행
        predict_start_time = time.time()
        
        segmentation_result = self.sam2_handler.segment_image(images)
        
        predict_end_time = time.time()
        predict_duration = predict_end_time - predict_start_time
        logger.info(f"[PREDICT TIME] Segmentation 시간: {predict_duration:.4f}초")
        
        print_gpu_memory_info("모델 추론 완료 후")
        
        # RGB to BGR 변환 (OpenCV 형식으로)
        colored_mask = cv2.cvtColor(segmentation_result['colored_mask'], cv2.COLOR_RGB2BGR)
        
        # 결과 패킹 (TensorFlow 모델과 동일한 형식)
        result = {
            'colored_mask': colored_mask.tolist(),  # JSON 직렬화 호환
            'mask': segmentation_result['mask'].tolist(),  # JSON 직렬화 호환
            'scores': segmentation_result.get('scores', []),
            'num_masks': segmentation_result.get('num_masks', 0),
            'processing_time': predict_duration,
            'original_shape': list(images.shape),
            'segmented_shape': list(colored_mask.shape)
        }
        
        result = pickle.dumps(result)
        
        print_gpu_memory_info("완료 후")
        
        return result, context

if __name__ == "__main__":
    # 테스트용 이미지 생성
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    image_path = "D:\\FTP\\notrhDoor.png"  # 여기에 실제 이미지 경로 입력
    test_image = cv2.imread(image_path)
    data = pickle.dumps(test_image)

    # 모델 핸들러 테스트 (무조건 sam2.1_hiera_large 모델 사용)
    handler = ModelHandler(None, None)
    # handler.test()
    
    result, context = handler(data=data, context={})
    result_dict = pickle.loads(result)
    
    logger.info(f"테스트 완료 - 처리 시간: {result_dict['processing_time']:.4f}초")
    logger.info(f"원본 크기: {result_dict['original_shape']}")
    logger.info(f"결과 크기: {result_dict['segmented_shape']}")
    
    # 이미지 표시 (numpy 배열로 변환)
    original_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    colored_mask = np.array(result_dict['colored_mask'], dtype=np.uint8)
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
    
    # matplotlib을 사용하여 이미지 표시
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(colored_mask)
    plt.title(f'All Objects ({result_dict["num_masks"]} masks)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 결과 이미지 저장 (numpy 배열로 변환)
    cv2.imwrite("original_image.jpg", test_image)
    colored_mask_bgr = np.array(result_dict['colored_mask'], dtype=np.uint8)
    cv2.imwrite("colored_mask.jpg", colored_mask_bgr)
    
    logger.info("이미지가 표시되었습니다. 창을 닫으면 프로그램이 종료됩니다.")
    logger.info("결과 이미지가 저장되었습니다: original_image.jpg, colored_mask.jpg")