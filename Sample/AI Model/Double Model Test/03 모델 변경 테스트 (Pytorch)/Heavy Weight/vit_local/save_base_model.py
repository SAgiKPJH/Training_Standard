import torch
from timm import create_model
import logging
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def save_base_model(model_name='vit_huge_patch14_224_in21k', num_classes=10, save_dir='base_models'):
    """ViT-Huge base 모델을 저장하는 함수
    
    Args:
        model_name (str): timm ViT-Huge 모델 이름
        num_classes (int): 클래스 수
        save_dir (str): 저장할 디렉토리
    """
    try:
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # ViT-Huge base 모델 생성
        logger.info(f"Creating ViT-Huge base model: {model_name}")
        model = create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )
        
        # 모델 저장
        save_path = os.path.join(save_dir, f"{model_name}_{num_classes}_base.model")
        torch.save(model, save_path)
        logger.info(f"ViT-Huge base model saved to: {save_path}")
        
        # 모델 아키텍처 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"Error saving ViT-Huge base model: {e}")
        raise

if __name__ == "__main__":
    # ViT-Huge base 모델 저장
    save_base_model()
