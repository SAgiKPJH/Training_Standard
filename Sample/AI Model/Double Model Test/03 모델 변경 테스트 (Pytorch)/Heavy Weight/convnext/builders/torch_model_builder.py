import time
import logging
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

class Torch_Model_Builder:
    def __init__(self, logger):
        self.__model = None
        self.__optimizer = None
        self.__criterion = None
        self.__iteration_start_time = None
        self.__epoch_start_time = None
        self.__epoch_total = None
        self.__save_epoch = None
        self.__logger = logger

    def initialize(self, epoch_total, save_epoch):
        self.__epoch_total = epoch_total
        self.__save_epoch = save_epoch
        return self
    
    def init_model(self, num_classes, device):
        model = torchvision.models.inception_v3(num_classes=num_classes, init_weights=False)
        model.to(device)
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

    def train(self, train_data_loader, valid_data_loader, device, using_amp, save_builder):
        total_iteration = len(train_data_loader)
        if total_iteration == 0:
            raise RuntimeError("No training data: train_data_loader is empty.")

        self.__iteration_start_time = time.time()
        for epoch in range(1, self.__epoch_total + 1):
            self.__epoch_start_time = time.time()
            self.__logger.info(f"Epoch : {epoch:4d}/{self.__epoch_total:4d}")
            
            train_epoch_loss = 0.0

            for n_epoch, batch in enumerate(train_data_loader, 1):
                train_inputs, loss = self.train_step(batch, device, using_amp)

                train_epoch_loss += loss.item()
                if n_epoch % max(1, total_iteration // 10) == 0 or n_epoch == total_iteration:
                    iteration_elapsed_time = time.time() - self.__iteration_start_time
                    self.__logger.info(f"Epoch : {epoch:4d}, Iterations : {n_epoch:4d}/{total_iteration:4d}, Loss : {loss : 4.4f}, Time : {iteration_elapsed_time : 4.4f}")
                    self.__iteration_start_time = time.time()

            save_builder.append_train_loss(train_epoch_loss / n_epoch)
            save_builder.set_hierarchy_root(f"epoch_{epoch}")

            valid_epoch_loss = 0.0
            if valid_data_loader:
                predict_list = np.array([])
                label_list = np.array([])
                with torch.no_grad():
                    self.__model.eval()
                    for _, valid_batch in enumerate(valid_data_loader, 1):

                        valid_inputs = valid_batch[0].to(device)
                        valid_labels = valid_batch[1].to(device)

                        with torch.cuda.amp.autocast(enabled=using_amp):
                            valid_outputs = self.__model(valid_inputs)
                            valid_loss = self.__criterion(valid_outputs, valid_labels)

                        valid_epoch_loss += valid_loss.item()
                        predict_list = np.concatenate([predict_list, valid_outputs.argmax(dim=1).cpu().numpy()], 0)
                        label_list = np.concatenate([label_list, valid_labels.cpu().numpy()], 0)
                    self.__model.train()
                
                save_builder.append_valid_loss(valid_epoch_loss / n_epoch)
                save_builder.save_validateion(epoch, label_list, predict_list)
            
            epoch_valid_loss_mean = valid_epoch_loss/len(valid_data_loader) if valid_data_loader else 0
            save_builder.save_training(self.__model, epoch, self.__save_epoch, train_inputs)

            save_builder.save_train_valid_csv()

            train_loss = train_epoch_loss/total_iteration
            self.monitoring(epoch, epoch_valid_loss_mean, train_loss)

        return self
    
    def train_step(self, batch, device, using_amp):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        with torch.cuda.amp.autocast(enabled=using_amp):
            output, _ = self.__model(inputs)
            loss = self.__criterion(output, labels)
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()
        return inputs, loss

    def monitoring(self, epoch, epoch_valid_loss_mean, train_loss):
        epoch_elapsed_time = time.time() - self.__epoch_start_time

        remaining_epochs = self.__epoch_total - epoch
        estimated_time_per_epoch = epoch_elapsed_time if epoch > 1 else 0
        estimated_remaining_time = remaining_epochs * estimated_time_per_epoch
        self.__logger.info("MonitoringData:"
            f"Epoch:[{epoch:4d}/{self.__epoch_total:4d}], "
            f"Train Loss: {train_loss:4.4f}, "
            f"Valid Loss : {epoch_valid_loss_mean:4.4f}, "
            f"Time: {epoch_elapsed_time:4.2f}s, "
            f"Estimated Remaining Time: {estimated_remaining_time / 60:.2f} minutes")
        
    def build(self):
        return self.__model 