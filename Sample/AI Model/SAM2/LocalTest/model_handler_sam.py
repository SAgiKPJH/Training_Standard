import os
import pickle
import json
import numpy as np
import torch
import cv2
import time
import logging
from typing import Optional, Tuple, List

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

## GPU Memory 제한
memory_limit = 6 * 1024

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

class SAMHandler:
    """SAM 모델 핸들러 - 자동 segmentation 수행"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sam_predictor = None
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        # SAM 모델 초기화
        self._load_sam_model()
        
    def _load_sam_model(self):
        """SAM 모델 로드"""
        try:
            print_gpu_memory_info("SAM 모델 로딩 전")
            
            # SAM 모델 import 및 로드
            from segment_anything import sam_model_registry, SamPredictor
            
            # SAM 모델 체크포인트 경로
            checkpoint_path = os.path.join(self.base_path, "sam_vit_h_4b8939.pth")
            
            if not os.path.exists(checkpoint_path):
                logger.warning(f"SAM 체크포인트가 없습니다: {checkpoint_path}")
                logger.info("SAM 체크포인트를 다운로드하거나 경로를 확인해주세요.")
                return
            
            # SAM 모델 로드
            self.model = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
            self.model.to(device=self.device)
            self.sam_predictor = SamPredictor(self.model)
            
            logger.info("SAM 모델 로드 완료")
            print_gpu_memory_info("SAM 모델 로딩 완료 후")
            
        except ImportError as e:
            logger.error(f"SAM 라이브러리 import 실패: {e}")
            logger.info("segment-anything 패키지를 설치해주세요: pip install segment-anything")
        except Exception as e:
            logger.error(f"SAM 모델 로드 실패: {e}")
    
    def _get_auto_points(self, image: np.ndarray, num_points: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """이미지에서 자동으로 포인트 생성 (중앙 기준)"""
        h, w = image.shape[:2]
        
        # 중앙 포인트
        center_x, center_y = w // 2, h // 2
        
        # 중앙 주변에 포인트들 생성
        points = []
        labels = []
        
        # 중앙 포인트
        points.append([center_x, center_y])
        labels.append(1)  # foreground
        
        # 중앙 주변에 추가 포인트들
        radius = min(w, h) // 4
        for i in range(num_points - 1):
            angle = 2 * np.pi * i / (num_points - 1)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            # 이미지 범위 내로 제한
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            
            points.append([x, y])
            labels.append(1)  # foreground
        
        return np.array(points), np.array(labels)
    
    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 segmentation 수행"""
        if self.sam_predictor is None:
            logger.error("SAM 모델이 로드되지 않았습니다.")
            return np.zeros_like(image)
        
        try:
            # SAM predictor에 이미지 설정
            self.sam_predictor.set_image(image)
            
            # 자동 포인트 생성
            points, labels = self._get_auto_points(image)
            
            # Segmentation 수행
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
            
            # 가장 높은 점수의 마스크 선택
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # 마스크를 이미지 형태로 변환
            segmented_image = image.copy()
            segmented_image[~best_mask] = 0  # 배경을 검은색으로
            
            return segmented_image
            
        except Exception as e:
            logger.error(f"Segmentation 실패: {e}")
            return image

class ModelHandler:
    def __init__(self, data, context):
        print_gpu_memory_info("모델 로딩 전")
        
        # SAM 핸들러 초기화
        self.sam_handler = SAMHandler()
        
        print_gpu_memory_info("모델 로딩 완료 후")

    def test(self):
        """모델 테스트 - 워밍업"""
        print_gpu_memory_info("모델 테스트 전")
        
        # 더미 이미지로 테스트
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = self.sam_handler.segment_image(dummy_image)
        
        logger.info(f"테스트 완료 - 결과 이미지 크기: {result.shape}")
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
        
        # 이미지 크기 조정 (SAM은 다양한 크기 지원)
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
        
        segmented_image = self.sam_handler.segment_image(images)
        
        predict_end_time = time.time()
        predict_duration = predict_end_time - predict_start_time
        logger.info(f"[PREDICT TIME] Segmentation 시간: {predict_duration:.4f}초")
        
        print_gpu_memory_info("모델 추론 완료 후")
        
        # RGB to BGR 변환 (OpenCV 형식으로)
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        
        # 결과 저장
        result['segmented_image'] = segmented_image
        result['processing_time'] = predict_duration
        result['original_shape'] = images.shape
        result['segmented_shape'] = segmented_image.shape
        
        result = pickle.dumps(result)
        
        print_gpu_memory_info("완료 후")
        
        return result, context

if __name__ == "__main__":
    # 테스트용 이미지 생성
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    data = pickle.dumps(test_image)
    
    # 모델 핸들러 테스트
    handler = ModelHandler(None, None)
    handler.test()
    
    result, context = handler(data=data, context={})
    result_dict = pickle.loads(result)
    
    logger.info(f"테스트 완료 - 처리 시간: {result_dict['processing_time']:.4f}초")
    logger.info(f"원본 크기: {result_dict['original_shape']}")
    logger.info(f"결과 크기: {result_dict['segmented_shape']}")



