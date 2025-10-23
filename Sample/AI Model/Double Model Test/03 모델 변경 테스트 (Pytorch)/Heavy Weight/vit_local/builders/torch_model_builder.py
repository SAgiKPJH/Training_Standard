import torch
import torch.nn as nn
import torch.optim as optim
from timm import create_model
import logging
import time
from torch.cuda.amp import autocast, GradScaler

class TorchModelBuilder:
    def __init__(self, hyperparameter_builder, logger):
        self.hyperparameter_builder = hyperparameter_builder
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() and hyperparameter_builder.get_use_gpu() else "cpu")
        self.scaler = GradScaler() if hyperparameter_builder.get_use_amp() else None
        self.__logger = logger

    def initialize(self):
        """모델 초기화"""
        return self

    def init_model(self):
        """ViT 모델 생성"""
        self.model = create_model(
            'vit_huge_patch14_224_in21k',  # ViT-Huge 모델 사용
            pretrained=True,
            num_classes=self.hyperparameter_builder.get_num_classes()
        )
        
        # 사전 학습된 가중치 로드 시도
        try:
            weights_path = 'base_models/vit_huge_patch14_224_in21k_10_base.model'
            if torch.cuda.is_available():
                model_state = torch.load(weights_path)
            else:
                model_state = torch.load(weights_path, map_location='cpu')
            
            if isinstance(model_state, dict):
                self.model.load_state_dict(model_state, strict=False)
            else:
                self.model = model_state
            self.__logger.info(f"Loaded pretrained weights from {weights_path}")
        except Exception as e:
            self.__logger.warning(f"Could not load pretrained weights: {e}")
        
        self.model.to(self.device)
        return self

    def init_optimizer(self):
        """옵티마이저 초기화"""
        optimizer_name = self.hyperparameter_builder.get_optimizer_name()
        lr = self.hyperparameter_builder.get_learning_rate()
        
        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name.lower() == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return self

    def init_criterion(self):
        """손실 함수 초기화"""
        self.criterion = nn.CrossEntropyLoss()
        return self

    def train_epoch(self, train_loader, epoch):
        """한 에폭 학습"""
        self.model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            if self.hyperparameter_builder.get_use_amp():
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                self.__logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        return train_loss / len(train_loader)

    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_loader.dataset)
        
        return val_loss, accuracy

    def save_model(self, epoch):
        """모델 저장"""
        save_path = f'saved_models/vit_huge_model_epoch_{epoch+1}.pth'
        torch.save(self.model, save_path)
        self.__logger.info(f'Model saved to {save_path}')

    def train(self, train_loader, val_loader):
        """전체 학습 과정"""
        self.__logger.info(f"Starting training on {self.device}")
        
        for epoch in range(self.hyperparameter_builder.get_num_epochs()):
            epoch_start_time = time.time()
            
            # 학습
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 검증
            val_loss, accuracy = self.validate(val_loader)
            
            # 결과 출력
            epoch_time = time.time() - epoch_start_time
            self.__logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s')
            self.__logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                       f'Accuracy: {accuracy:.2f}%')
            
            # 모델 저장
            if (epoch + 1) % self.hyperparameter_builder.get_save_epoch() == 0:
                self.save_model(epoch)

    def build(self):
        return self \
            .init_model() \
            .init_optimizer() \
            .init_criterion()
