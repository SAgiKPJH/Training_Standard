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
        
        self.__optimizer = None
        self.__criterion = None
        self.__model = None
        
        self.__iteration_start_time = None
        self.__epoch_start_time = None

    def initialize(self, epoch_total:int, using_amp, device:str='cpu'):
        self.__epoch_total = epoch_total
        self.__device = device
        self.__using_amp = using_amp
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
                predict_list = np.array([])
                label_list = np.array([])
                with torch.no_grad():
                    self.__model.eval()
                    for _, valid_batch in enumerate(validation_data_loader, 1):

                        valid_inputs = valid_batch[0].to(self.__device)
                        valid_labels = valid_batch[1].to(self.__device)

                        with torch.cuda.amp.autocast(enabled=self.__using_amp):
                            valid_outputs = self.__model(valid_inputs)
                            valid_loss = self.__criterion(valid_outputs, valid_labels)

                        valid_epoch_loss += valid_loss.item()
                        predict_list = np.concatenate([predict_list, valid_outputs.argmax(dim=1).cpu().numpy()], 0)
                        label_list = np.concatenate([label_list, valid_labels.cpu().numpy()], 0)
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
        with torch.cuda.amp.autocast(enabled=self.__using_amp):
            output, *_ = self.__model(inputs)
            loss = self.__criterion(output, labels)
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()
        return inputs, loss

    def builder(self):
        return self