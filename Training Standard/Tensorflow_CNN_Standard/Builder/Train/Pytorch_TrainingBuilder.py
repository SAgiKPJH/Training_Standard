import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .TrainHook import TrainHook

class Pytorch_TrainingBuilder:
    def __init__(self, logger = None):
        self.__logger = logger

        self.__epoch_total = None
        self.__device = 'cpu'
        self.__using_amp = False
        self.__loss_eps = 0.0

        self.__optimizer = None
        self.__criterion = None
        self.__model = None

        self.__iteration_start_time = None
        self.__epoch_start_time = None

    def initialize(self, epoch_total:int, using_amp, device:str='cpu', loss_eps:float=0.0):
        self.__epoch_total = epoch_total
        self.__device = device
        self.__using_amp = using_amp
        self.__loss_eps = float(loss_eps)
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
            self.__optimizer = optim.Adam(self.__model.parameters(), lr=lr)
        elif optimizer_name.lower() == "sgd":
            self.__optimizer = optim.SGD(self.__model.parameters(), lr=lr)
        elif optimizer_name.lower() == "adagrad":
            self.__optimizer = optim.Adagrad(self.__model.parameters(), lr=lr)
        else:
            raise Exception("Invalid Optimizer Option")
        return self
    
    def init_criterion(self, criterion_name):
        if criterion_name.lower() == "crossentropyloss":
            criterion = nn.CrossEntropyLoss()
        elif criterion_name.lower() == "bcewithlogitsloss":
            criterion = nn.BCEWithLogitsLoss()
        elif criterion_name.lower() == "focalloss":
            criterion = nn.BCEWithLogitsLoss()
        elif criterion_name.lower() == "mse":
            criterion = nn.MSELoss()

        self.__criterion = criterion
        return self

    def train(self, train_data_loader, validation_data_loader, hook:TrainHook = None):
        if hook: hook.training_start()

        total_iteration = len(train_data_loader)
        if total_iteration == 0:
            raise RuntimeError("No training data: train_data_loader is empty.")

        self.__iteration_start_time = time.time()
        for epoch in range(1, self.__epoch_total + 1):
            self.__epoch_start_time = time.time()
            if self.__logger: self.__logger.info(f"Epoch : {epoch:4d}/{self.__epoch_total:4d}")
            
            train_epoch_loss = 0.0

            for n_epoch, batch in enumerate(train_data_loader, 1):
                train_inputs, loss = self._train_step(batch)

                train_epoch_loss += loss.item()
                if n_epoch % max(1, total_iteration // 10) == 0 or n_epoch == total_iteration:
                    iteration_elapsed_time = time.time() - self.__iteration_start_time
                    self.__logger.info(f"Epoch : {epoch:4d}, Iterations : {n_epoch:4d}/{total_iteration:4d}, Loss : {loss : 4.4f}, Time : {iteration_elapsed_time : 4.4f}")
                    self.__iteration_start_time = time.time()
            
            valid_epoch_loss = 0.0
            if validation_data_loader:
                with torch.no_grad():
                    self.__model.eval()
                    for _, valid_batch in enumerate(validation_data_loader, 1):
                        
                        valid_inputs, valid_loss = self._valid_step(valid_batch)

                        valid_epoch_loss += valid_loss.item()
                    self.__model.train()

            if hook: hook.on_epoch_end(
                total_epoch= self.__epoch_total,
                epoch= epoch,
                train_loss= train_epoch_loss / n_epoch,
                validation_loss= valid_epoch_loss / len(validation_data_loader) if validation_data_loader else None,
                epoch_elapsed_time= time.time() - self.__epoch_start_time,
                model= self.__model
            )

        if hook: hook.training_end()
        return self
    
    def _train_step(self, batch):
        inputs = batch[0].to(self.__device)
        labels = batch[1].to(self.__device)
        with torch.amp.autocast(device_type=self.__device, enabled=self.__using_amp):
            outputs = self.__model(inputs)
            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs
            loss = self.__criterion(output, labels)
            # loss eps: NaN/underflow 방지를 위해 작은 값 추가
            if self.__loss_eps > 0:
                loss = loss + self.__loss_eps
        self.__optimizer.zero_grad()
        self.__scaler.scale(loss).backward()
        self.__scaler.step(self.__optimizer)
        self.__scaler.update()
        return inputs, loss

    def _valid_step(self, batch):
        inputs = batch[0].to(self.__device)
        labels = batch[1].to(self.__device)

        with torch.amp.autocast(device_type=self.__device, enabled=self.__using_amp):
            outputs = self.__model(inputs)
            if isinstance(outputs, tuple):
                valid_outputs = outputs[0]
            else:
                valid_outputs = outputs
            loss = self.__criterion(valid_outputs, labels)
        return inputs, loss

    def builder(self):
        return self