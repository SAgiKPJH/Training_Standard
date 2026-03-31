import time
import logging
import numpy as np
import tensorflow as tf
from .TrainHook import TrainHook

logging.basicConfig(
    format="[DEBUG][%(levelname)s] %(message)s",
    level=logging.DEBUG
)

def _tensor_stats(name: str, t: tf.Tensor) -> str:
    t_f = tf.cast(t, tf.float32)
    has_nan = tf.reduce_any(tf.math.is_nan(t_f)).numpy()
    has_inf = tf.reduce_any(tf.math.is_inf(t_f)).numpy()
    return (
        f"{name} | shape={list(t.shape)} dtype={t.dtype.name} "
        f"min={tf.reduce_min(t_f).numpy():.6f} max={tf.reduce_max(t_f).numpy():.6f} "
        f"mean={tf.reduce_mean(t_f).numpy():.6f} std={tf.math.reduce_std(t_f).numpy():.6f} "
        f"nan={has_nan} inf={has_inf}"
    )


class Tensorflow_TrainingBuilder_Debug:
    """
    NaN 손실 디버깅용 TrainingBuilder (TensorFlow).
    각 step마다 입력/출력/gradient 통계를 로그로 출력합니다.

    log_every_n_iterations : int
        몇 iteration 마다 상세 로그를 출력할지 (기본 1 = 매 iteration)
    stop_on_nan : bool
        True 이면 NaN loss 감지 시 즉시 RuntimeError 발생
    """

    def __init__(self, logger=None, log_every_n_iterations: int = 1,
                 log_layer_hooks: bool = True, stop_on_nan: bool = True):
        self.__external_logger = logger
        self.__log_every_n = log_every_n_iterations
        self.__stop_on_nan = stop_on_nan

        self.__epoch_total = None
        self.__device = '/cpu:0'
        self.__using_amp = False

        self.__optimizer = None
        self.__criterion = None
        self.__model = None

        self.__iteration_start_time = None
        self.__epoch_start_time = None

    # ------------------------------------------------------------------
    # Internal logging helper
    # ------------------------------------------------------------------
    def _log(self, msg: str, level: str = "info"):
        if self.__external_logger:
            getattr(self.__external_logger, level)(msg)
        else:
            getattr(logging, level)(msg)

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------
    def initialize(self, epoch_total: int, using_amp, device: str = '/cpu:0'):
        self.__epoch_total = epoch_total
        self.__device = device
        self.__using_amp = using_amp
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
            self.__optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif name == "adagrad":
            self.__optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
        else:
            raise Exception("Invalid Optimizer Option")
        return self

    def init_criterion(self, criterion_name):
        name = criterion_name.lower()
        if name == "crossentropyloss":
            self.__criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif name == "bcewithlogitsloss":
            self.__criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif name == "focalloss":
            self.__criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif name == "mse":
            self.__criterion = tf.keras.losses.MeanSquaredError()
        else:
            raise Exception("Invalid Criterion Option")
        return self

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self, train_data_loader, validation_data_loader, hook: TrainHook = None):
        if hook:
            hook.training_start()

        total_iteration = len(train_data_loader)
        if total_iteration == 0:
            raise RuntimeError("No training data: train_data_loader is empty.")

        self.__iteration_start_time = time.time()

        for epoch in range(1, self.__epoch_total + 1):
            self.__epoch_start_time = time.time()
            self._log(f"========== Epoch {epoch}/{self.__epoch_total} ==========", level="info")

            train_epoch_loss = 0.0

            for n_iter, (inputs, labels) in enumerate(train_data_loader, 1):
                do_log = (n_iter % max(1, self.__log_every_n) == 0) or (n_iter == total_iteration)

                if do_log:
                    self._log(f"[TRAIN] Epoch {epoch}, Iter {n_iter}/{total_iteration} - step start", level="debug")

                loss, gradients = self._train_step(inputs, labels, do_log)

                loss_val = loss.numpy()
                train_epoch_loss += loss_val

                is_nan = np.isnan(loss_val) or np.isinf(loss_val)
                if is_nan:
                    self._log(
                        f"[NaN DETECTED] Epoch {epoch}, Iter {n_iter} - loss={loss_val}",
                        level="error"
                    )
                    if self.__stop_on_nan:
                        raise RuntimeError(
                            f"NaN/Inf loss detected at epoch={epoch}, iter={n_iter}. "
                            "Check the logs above for the problematic layer/tensor."
                        )

                if do_log:
                    elapsed = time.time() - self.__iteration_start_time
                    self._log(
                        f"[TRAIN] Epoch {epoch:4d}, Iter {n_iter:4d}/{total_iteration:4d}, "
                        f"Loss={loss_val:.6f}, Elapsed={elapsed:.4f}s",
                        level="info"
                    )
                    self.__iteration_start_time = time.time()

            # Validation
            valid_epoch_loss = 0.0
            valid_count = 0
            if validation_data_loader:
                for valid_inputs, valid_labels in validation_data_loader:
                    valid_loss = self._valid_step(valid_inputs, valid_labels)
                    valid_epoch_loss += valid_loss.numpy()
                    valid_count += 1

            avg_train = train_epoch_loss / n_iter
            avg_valid = (valid_epoch_loss / valid_count
                         if validation_data_loader and valid_count > 0 else None)

            self._log(
                f"[EPOCH END] Epoch {epoch}/{self.__epoch_total} "
                f"| train_loss={avg_train:.6f} "
                f"| valid_loss={avg_valid:.6f if avg_valid is not None else 'N/A'}",
                level="info"
            )

            if hook:
                hook.on_epoch_end(
                    total_epoch=self.__epoch_total,
                    epoch=epoch,
                    train_loss=avg_train,
                    validation_loss=avg_valid,
                    epoch_elapsed_time=time.time() - self.__epoch_start_time,
                    model=self.__model
                )

        if hook:
            hook.training_end()
        return self

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------
    def _train_step(self, inputs, labels, do_log: bool):
        if do_log:
            self._log(_tensor_stats("  [INPUT ]", inputs), level="debug")
            self._log(_tensor_stats("  [LABEL ]", tf.cast(labels, tf.float32)), level="debug")

        with tf.GradientTape() as tape:
            outputs = self.__model(inputs, training=True)

            if do_log:
                self._log(_tensor_stats("  [OUTPUT]", outputs), level="debug")

            loss = self.__criterion(labels, outputs)

        if do_log:
            self._log(
                f"  [LOSS  ] value={loss.numpy():.6f} "
                f"nan={tf.math.is_nan(loss).numpy()} "
                f"inf={tf.math.is_inf(loss).numpy()}",
                level="debug"
            )

        gradients = tape.gradient(loss, self.__model.trainable_variables)

        # Gradient diagnostics
        if do_log:
            self._check_gradients(gradients)

        self.__optimizer.apply_gradients(zip(gradients, self.__model.trainable_variables))
        return loss, gradients

    def _valid_step(self, inputs, labels):
        outputs = self.__model(inputs, training=False)
        loss = self.__criterion(labels, outputs)
        return loss

    # ------------------------------------------------------------------
    # Gradient diagnostics
    # ------------------------------------------------------------------
    def _check_gradients(self, gradients):
        nan_layers = []
        for var, grad in zip(self.__model.trainable_variables, gradients):
            if grad is None:
                continue
            g = tf.cast(grad, tf.float32)
            has_nan = tf.reduce_any(tf.math.is_nan(g)).numpy()
            has_inf = tf.reduce_any(tf.math.is_inf(g)).numpy()
            if has_nan or has_inf:
                nan_layers.append(var.name)
                self._log(
                    f"  [GRAD NaN/Inf] {var.name} | {_tensor_stats('grad', grad)}",
                    level="warning"
                )
            else:
                self._log(
                    f"  [GRAD ] {var.name} | {_tensor_stats('grad', grad)}",
                    level="debug"
                )
        if nan_layers:
            self._log(
                f"  [GRAD SUMMARY] NaN/Inf detected in: {nan_layers}",
                level="error"
            )

    def builder(self):
        return self
