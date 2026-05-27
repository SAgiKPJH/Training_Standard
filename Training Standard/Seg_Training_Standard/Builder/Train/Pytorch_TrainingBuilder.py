import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .TrainHook import TrainHook


class Pytorch_TrainingBuilder:
    """Segmentation 학습 빌더 (PyTorch)."""

    def __init__(self, logger=None):
        self.__logger = logger

        self.__epoch_total = None
        self.__device = 'cpu'
        self.__using_amp = False
        self.__loss_eps = 0.0
        self.__weight_decay = 1e-4

        self.__optimizer = None
        self.__criterion = None
        self.__model = None

        self.__iteration_start_time = None
        self.__epoch_start_time = None

    def initialize(self, epoch_total: int, using_amp, device: str = 'cpu', loss_eps: float = 0.0, weight_decay: float = 1e-4):
        self.__epoch_total = epoch_total
        self.__device = device
        self.__using_amp = using_amp
        self.__loss_eps = float(loss_eps)
        self.__weight_decay = float(weight_decay)
        try:
            self.__scaler = torch.amp.GradScaler(self.__device, enabled=using_amp)
        except (TypeError, AttributeError):
            self.__scaler = torch.cuda.amp.GradScaler(enabled=using_amp)
        return self

    def init_model(self, model):
        self.__model = model
        return self

    def init_optimizer(self, optimizer_name, lr):
        if optimizer_name.lower() == "adam":
            self.__optimizer = optim.Adam(self.__model.parameters(), lr=lr, weight_decay=self.__weight_decay)
        elif optimizer_name.lower() == "sgd":
            self.__optimizer = optim.SGD(self.__model.parameters(), lr=lr, momentum=0.9, weight_decay=self.__weight_decay)
        elif optimizer_name.lower() == "adagrad":
            self.__optimizer = optim.Adagrad(self.__model.parameters(), lr=lr, weight_decay=self.__weight_decay)
        else:
            raise Exception("Invalid Optimizer Option")
        return self

    def init_criterion(self, criterion_name):
        name = criterion_name.lower()
        if name == "crossentropyloss":
            self.__criterion = nn.CrossEntropyLoss(ignore_index=-1)
        elif name == "diceloss":
            self.__criterion = _DiceLoss()
        else:
            self.__criterion = nn.CrossEntropyLoss(ignore_index=-1)
        return self

    def train(self, train_data_loader, validation_data_loader, hook: TrainHook = None):
        if hook:
            hook.training_start()

        total_iteration = len(train_data_loader)
        if total_iteration == 0:
            raise RuntimeError("No training data: train_data_loader is empty.")

        self.__iteration_start_time = time.time()
        for epoch in range(1, self.__epoch_total + 1):
            self.__epoch_start_time = time.time()
            if self.__logger:
                self.__logger.info(f"Epoch : {epoch:4d}/{self.__epoch_total:4d}")

            train_epoch_loss = 0.0
            for n_epoch, batch in enumerate(train_data_loader, 1):
                loss = self._train_step(batch)
                train_epoch_loss += loss.item()

                if n_epoch % max(1, total_iteration // 10) == 0 or n_epoch == total_iteration:
                    iteration_elapsed_time = time.time() - self.__iteration_start_time
                    self.__logger.info(
                        f"Epoch : {epoch:4d}, Iter : {n_epoch:4d}/{total_iteration:4d}, "
                        f"Loss : {loss.item():4.4f}, Time : {iteration_elapsed_time:4.2f}s"
                    )
                    self.__iteration_start_time = time.time()

            valid_epoch_loss = 0.0
            valid_count = 0
            if validation_data_loader:
                with torch.no_grad():
                    self.__model.eval()
                    for valid_batch in validation_data_loader:
                        valid_loss = self._valid_step(valid_batch)
                        valid_epoch_loss += valid_loss.item()
                        valid_count += 1
                    self.__model.train()

            if hook:
                hook.on_epoch_end(
                    total_epoch=self.__epoch_total,
                    epoch=epoch,
                    train_loss=train_epoch_loss / n_epoch,
                    validation_loss=valid_epoch_loss / valid_count if valid_count > 0 else None,
                    epoch_elapsed_time=time.time() - self.__epoch_start_time,
                    model=self.__model
                )

        if hook:
            hook.training_end()
        return self

    def _train_step(self, batch):
        images = batch[0].to(self.__device, dtype=torch.float32)
        masks = batch[1].to(self.__device, dtype=torch.long)

        with torch.amp.autocast(device_type=self.__device, enabled=self.__using_amp):
            outputs = self.__model(images)
            loss = self.__criterion(outputs, masks)
            if self.__loss_eps > 0:
                loss = loss + self.__loss_eps

        self.__optimizer.zero_grad()
        self.__scaler.scale(loss).backward()
        self.__scaler.step(self.__optimizer)
        self.__scaler.update()
        return loss

    def _valid_step(self, batch):
        images = batch[0].to(self.__device, dtype=torch.float32)
        masks = batch[1].to(self.__device, dtype=torch.long)
        with torch.amp.autocast(device_type=self.__device, enabled=self.__using_amp):
            outputs = self.__model(images)
            loss = self.__criterion(outputs, masks)
        return loss

    def builder(self):
        return self


class _DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        valid = (target != self.ignore_index)
        target_clamped = target.clamp(min=0)
        one_hot = torch.nn.functional.one_hot(target_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()

        intersection = (probs * one_hot * valid_mask).sum(dim=(2, 3))
        union = (probs * valid_mask).sum(dim=(2, 3)) + (one_hot * valid_mask).sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
