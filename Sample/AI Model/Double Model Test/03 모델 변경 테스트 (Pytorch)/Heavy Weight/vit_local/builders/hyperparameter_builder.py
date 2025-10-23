from torchvision import transforms

class HyperparameterBuilder:
    def __init__(self, config=None):
        self.config = config or self.get_default_config()

    def get_default_config(self):
        return {
            'hyperparameter': {
                'epoch': 20,
                'save_epoch': 1,
                'batch_size': 4,  # ViT-Huge는 메모리 사용량이 크므로 배치 크기 감소
                'lr': 5e-5,  # ViT-Huge에 적합한 더 낮은 학습률
                'optimizer_name': 'AdamW',  # ViT에 적합한 옵티마이저
                'input_size': 224,  # ViT 기본 입력 크기
                'normalize_mean': [0.485, 0.456, 0.406],  # ImageNet 평균
                'normalize_stdev': [0.229, 0.224, 0.225],  # ImageNet 표준편차
                'using_gpu': True,
                'using_amp': True,
                'train_ratio': 0.8,
                'validation_save_random': False,
                'num_samples': 1000,
                'num_classes': 10
            }
        }

    def initialize(self):
        """하이퍼파라미터 초기화"""
        return self

    def get_transform(self):
        """데이터 변환 함수 반환"""
        return transforms.Compose([
            transforms.Normalize(
                mean=self.config['hyperparameter']['normalize_mean'],
                std=self.config['hyperparameter']['normalize_stdev']
            )
        ])

    def get_num_classes(self):
        return self.config['hyperparameter']['num_classes']

    def get_batch_size(self):
        return self.config['hyperparameter']['batch_size']

    def get_num_epochs(self):
        return self.config['hyperparameter']['epoch']

    def get_learning_rate(self):
        return self.config['hyperparameter']['lr']

    def get_image_size(self):
        return self.config['hyperparameter']['input_size']

    def get_num_samples(self):
        return self.config['hyperparameter']['num_samples']

    def get_train_ratio(self):
        return self.config['hyperparameter']['train_ratio']

    def get_use_gpu(self):
        return self.config['hyperparameter']['using_gpu']

    def get_use_amp(self):
        return self.config['hyperparameter']['using_amp']

    def get_save_epoch(self):
        return self.config['hyperparameter']['save_epoch']

    def get_optimizer_name(self):
        return self.config['hyperparameter']['optimizer_name']

    def build(self):
        return self
