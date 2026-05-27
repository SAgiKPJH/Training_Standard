import time
import numpy as np
import tensorflow as tf
from .TrainHook import TrainHook


class Tensorflow_TrainingBuilder:
    """Segmentation 학습 빌더 (TensorFlow)."""

    def __init__(self, logger=None):
        self.__logger = logger

        self.__epoch_total = None
        self.__device = '/cpu:0'
        self.__using_amp = False
        self.__loss_eps = 0.0
        self.__weight_decay = 1e-4

        self.__optimizer = None
        self.__criterion = None
        self.__model = None

        self.__iteration_start_time = None
        self.__epoch_start_time = None

    def initialize(self, epoch_total: int, using_amp, device: str = '/cpu:0', loss_eps: float = 0.0, weight_decay: float = 1e-4):
        self.__epoch_total = epoch_total
        self.__device = device
        self.__using_amp = using_amp
        self.__loss_eps = float(loss_eps)
        self.__weight_decay = float(weight_decay)
        if using_amp:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        return self

    def init_model(self, model):
        self.__model = model
        return self

    def init_optimizer(self, optimizer_name, lr):
        name = optimizer_name.lower()
        if name == "adam":
            self.__optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif name == "sgd":
            self.__optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif name == "adagrad":
            self.__optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        else:
            raise Exception("Invalid Optimizer Option")
        return self

    def init_criterion(self, criterion_name):
        # SparseCategoricalCrossentropy: target은 int32 class indices, logits=from_logits=True
        self.__criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        return self

    def train(self, train_data_loader, validation_data_loader, hook: TrainHook = None):
        if hook:
            hook.training_start()

        total_iteration = sum(1 for _ in train_data_loader)
        if total_iteration == 0:
            raise RuntimeError("No training data: train_data_loader is empty.")

        self.__iteration_start_time = time.time()
        for epoch in range(1, self.__epoch_total + 1):
            self.__epoch_start_time = time.time()
            if self.__logger:
                self.__logger.info(f"Epoch : {epoch:4d}/{self.__epoch_total:4d}")

            train_epoch_loss = 0.0
            n_epoch = 0
            for inputs, labels in train_data_loader:
                n_epoch += 1
                loss = self._train_step(inputs, labels)
                train_epoch_loss += loss.numpy()
                if n_epoch % max(1, total_iteration // 10) == 0 or n_epoch == total_iteration:
                    iteration_elapsed_time = time.time() - self.__iteration_start_time
                    self.__logger.info(
                        f"Epoch : {epoch:4d}, Iter : {n_epoch:4d}/{total_iteration:4d}, "
                        f"Loss : {loss.numpy():4.4f}, Time : {iteration_elapsed_time:4.2f}s"
                    )
                    self.__iteration_start_time = time.time()

            valid_epoch_loss = 0.0
            valid_count = 0
            if validation_data_loader:
                for valid_inputs, valid_labels in validation_data_loader:
                    valid_loss = self._valid_step(valid_inputs, valid_labels)
                    valid_epoch_loss += valid_loss.numpy()
                    valid_count += 1

            if hook:
                hook.on_epoch_end(
                    total_epoch=self.__epoch_total,
                    epoch=epoch,
                    train_loss=train_epoch_loss / max(1, n_epoch),
                    validation_loss=valid_epoch_loss / valid_count if valid_count > 0 else None,
                    epoch_elapsed_time=time.time() - self.__epoch_start_time,
                    model=self.__model
                )

        if hook:
            hook.training_end()
        return self

    @tf.function
    def _train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            logits = self.__model(inputs, training=True)
            # labels: (B, H, W) int32, logits: (B, H, W, C)
            valid = tf.cast(labels >= 0, tf.float32)
            labels_clamped = tf.maximum(labels, 0)
            per_pixel = self.__criterion(labels_clamped, logits)
            loss = tf.reduce_sum(per_pixel * valid) / (tf.reduce_sum(valid) + 1e-6)
            if self.__loss_eps > 0:
                loss = loss + self.__loss_eps
        gradients = tape.gradient(loss, self.__model.trainable_variables)
        self.__optimizer.apply_gradients(zip(gradients, self.__model.trainable_variables))
        return loss

    @tf.function
    def _valid_step(self, inputs, labels):
        logits = self.__model(inputs, training=False)
        valid = tf.cast(labels >= 0, tf.float32)
        labels_clamped = tf.maximum(labels, 0)
        per_pixel = self.__criterion(labels_clamped, logits)
        loss = tf.reduce_sum(per_pixel * valid) / (tf.reduce_sum(valid) + 1e-6)
        return loss

    def builder(self):
        return self
