import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class DummyDataset(Dataset):
    def __init__(self, num_samples, num_classes, image_size, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform
        
        # 더미 데이터 생성
        self.data = torch.randn(num_samples, 3, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DummyDatasetBuilder:
    def __init__(self, hyperparameter_builder):
        self.hyperparameter_builder = hyperparameter_builder
        self.train_loader = None
        self.val_loader = None

    def initialize(self):
        """데이터셋 초기화"""
        return self

    def build_dataset(self):
        """데이터셋 생성"""
        num_samples = self.hyperparameter_builder.get_num_samples()
        num_classes = self.hyperparameter_builder.get_num_classes()
        image_size = self.hyperparameter_builder.get_image_size()
        transform = self.hyperparameter_builder.get_transform()
        
        dataset = DummyDataset(num_samples, num_classes, image_size, transform)
        
        # 학습/검증 데이터 분할
        train_size = int(self.hyperparameter_builder.get_train_ratio() * num_samples)
        val_size = num_samples - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # 데이터로더 생성
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.hyperparameter_builder.get_batch_size(),
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Created dataset with {num_samples} samples, {num_classes} classes")
        logger.info(f"Train size: {train_size}, Validation size: {val_size}")
        
        return self

    def get_train_loader(self):
        return self.train_loader

    def get_validation_loader(self):
        return self.val_loader

    def build(self):
        return self.build_dataset() 