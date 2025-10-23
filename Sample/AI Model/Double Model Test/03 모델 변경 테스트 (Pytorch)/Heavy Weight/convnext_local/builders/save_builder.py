import os
import json
import logging

logger = logging.getLogger(__name__)

class LocalSaveBuilder:
    def __init__(self, hyperparameter_builder):
        self.hyperparameter_builder = hyperparameter_builder
        self.save_dir = 'saved_models'
        os.makedirs(self.save_dir, exist_ok=True)

    def initialize(self):
        """저장 설정 초기화"""
        # 설정 저장
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.hyperparameter_builder.config, f, indent=4)
        logger.info(f"Saved configuration to {config_path}")
        return self

    def save_model(self, model, epoch):
        """모델 저장"""
        save_path = os.path.join(self.save_dir, f'convnext_model_epoch_{epoch+1}.pth')
        model.save(save_path)
        logger.info(f'Model saved to {save_path}')
        return self

    def build(self):
        return self 