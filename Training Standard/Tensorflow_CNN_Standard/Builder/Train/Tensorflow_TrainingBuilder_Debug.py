import time
import logging
import numpy as np
from collections import deque
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
    각 step마다 입력/출력/gradient/loss 상세 통계를 로그로 출력합니다.

    log_every_n_iterations : int
        몇 iteration 마다 상세 로그를 출력할지 (기본 1 = 매 iteration)
    stop_on_nan : bool
        True 이면 NaN loss 감지 시 즉시 RuntimeError 발생
    loss_moving_avg_window : int
        loss 이동평균 계산 윈도우 크기 (기본 20)
    """

    def __init__(self, logger=None, log_every_n_iterations: int = 1,
                 log_layer_hooks: bool = True, stop_on_nan: bool = True,
                 loss_moving_avg_window: int = 20):
        self.__external_logger = logger
        self.__log_every_n = log_every_n_iterations
        self.__stop_on_nan = stop_on_nan

        self.__epoch_total = None
        self.__device = '/cpu:0'
        self.__using_amp = False

        self.__optimizer = None
        self.__criterion = None
        self.__criterion_unreduced = None  # per-sample loss 계산용
        self.__model = None

        self.__iteration_start_time = None
        self.__epoch_start_time = None

        # Loss 추세 추적
        self.__loss_history = deque(maxlen=loss_moving_avg_window)
        self.__prev_loss = None
        self.__loss_moving_avg_window = loss_moving_avg_window
        self.__global_step = 0

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
            self.__criterion_unreduced = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        elif name == "bcewithlogitsloss":
            self.__criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.__criterion_unreduced = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
        elif name == "focalloss":
            self.__criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.__criterion_unreduced = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
        elif name == "mse":
            self.__criterion = tf.keras.losses.MeanSquaredError()
            self.__criterion_unreduced = tf.keras.losses.MeanSquaredError(reduction='none')
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
                do_log = epoch >= 20 and ((n_iter % max(1, self.__log_every_n) == 0) or (n_iter == total_iteration))

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
        self.__global_step += 1

        if do_log:
            self._log(_tensor_stats("  [INPUT ]", inputs), level="debug")
            self._log(_tensor_stats("  [LABEL ]", tf.cast(labels, tf.float32)), level="debug")
            self._log_label_distribution(labels)

        # 파라미터 스냅샷 (weight update 크기 측정용)
        param_snapshot = None
        if do_log:
            param_snapshot = {var.name: tf.identity(var) for var in self.__model.trainable_variables}

        with tf.GradientTape() as tape:
            outputs = self.__model(inputs, training=True)

            if do_log:
                self._log(_tensor_stats("  [OUTPUT]", outputs), level="debug")
                self._log_logit_distribution(outputs, labels)

            loss = self.__criterion(labels, outputs)

        if do_log:
            # 기본 loss 정보
            self._log(
                f"  [LOSS  ] value={loss.numpy():.6f} "
                f"nan={tf.math.is_nan(loss).numpy()} "
                f"inf={tf.math.is_inf(loss).numpy()}",
                level="debug"
            )
            # 상세 loss 분석
            self._log_loss_detail(outputs, labels, loss)
            # loss 추세
            self._log_loss_trend(loss.numpy())

        gradients = tape.gradient(loss, self.__model.trainable_variables)

        # Gradient diagnostics
        if do_log:
            self._check_gradients(gradients)

        self.__optimizer.apply_gradients(zip(gradients, self.__model.trainable_variables))

        # Weight update 크기 측정
        if do_log and param_snapshot is not None:
            self._log_weight_update(param_snapshot)

        return loss, gradients

    def _valid_step(self, inputs, labels):
        outputs = self.__model(inputs, training=False)
        loss = self.__criterion(labels, outputs)
        return loss

    # ------------------------------------------------------------------
    # Loss detail diagnostics
    # ------------------------------------------------------------------
    def _log_loss_detail(self, output: tf.Tensor, labels: tf.Tensor, loss: tf.Tensor):
        """Per-sample loss, softmax 확률 분포, prediction-target 비교 등 상세 분석"""
        batch_size = output.shape[0]
        if batch_size is None:
            return

        # --- Per-sample loss ---
        per_sample = self.__criterion_unreduced(labels, output)
        if len(per_sample.shape) > 1:
            per_sample = tf.reduce_mean(per_sample, axis=list(range(1, len(per_sample.shape))))

        ps_np = per_sample.numpy()
        ps_min = float(np.min(ps_np))
        ps_max = float(np.max(ps_np))
        ps_mean = float(np.mean(ps_np))
        ps_std = float(np.std(ps_np)) if batch_size > 1 else 0.0
        ps_median = float(np.median(ps_np))

        self._log(
            f"  [LOSS DETAIL] per_sample | "
            f"min={ps_min:.6f} max={ps_max:.6f} mean={ps_mean:.6f} "
            f"std={ps_std:.6f} median={ps_median:.6f}",
            level="debug"
        )

        # 가장 큰 loss를 가진 샘플 Top-3
        if batch_size >= 3:
            top_k = min(3, batch_size)
            top_vals, top_idx = tf.math.top_k(per_sample, k=top_k)
            top_info = ", ".join(
                [f"sample[{idx}]={val:.6f}" for val, idx in zip(top_vals.numpy(), top_idx.numpy())]
            )
            self._log(f"  [LOSS DETAIL] worst_samples | {top_info}", level="debug")

        # --- Softmax 확률 분석 (classification) ---
        if len(output.shape) == 2 and output.shape[1] > 1:
            logits = tf.cast(output, tf.float32)
            probs = tf.nn.softmax(logits, axis=1)
            num_classes = output.shape[1]

            # 정답 클래스의 확률
            labels_np = labels.numpy() if hasattr(labels, 'numpy') else labels
            if len(tf.shape(labels)) == 1 and labels.dtype in (tf.int32, tf.int64):
                batch_indices = tf.range(batch_size, dtype=labels.dtype)
                indices = tf.stack([batch_indices, tf.cast(labels, labels.dtype)], axis=1)
                target_probs = tf.gather_nd(probs, indices)

                self._log(
                    f"  [LOSS DETAIL] target_class_prob | "
                    f"min={tf.reduce_min(target_probs).numpy():.6f} "
                    f"max={tf.reduce_max(target_probs).numpy():.6f} "
                    f"mean={tf.reduce_mean(target_probs).numpy():.6f}",
                    level="debug"
                )

                # Top-1 정확도
                preds = tf.argmax(logits, axis=1, output_type=labels.dtype)
                correct = tf.cast(tf.equal(preds, labels), tf.float32)
                top1_acc = tf.reduce_mean(correct).numpy() * 100
                n_correct = int(tf.reduce_sum(correct).numpy())

                self._log(
                    f"  [LOSS DETAIL] batch_top1_acc={top1_acc:.1f}% "
                    f"({n_correct}/{batch_size})",
                    level="debug"
                )

                # 오분류 샘플 상세
                wrong_mask = tf.not_equal(preds, labels)
                wrong_idx = tf.where(wrong_mask)
                n_wrong = len(wrong_idx)
                if 0 < n_wrong <= 10:
                    preds_np = preds.numpy()
                    labels_np = labels.numpy()
                    probs_np = probs.numpy()
                    ps_np_local = per_sample.numpy()
                    for wi in wrong_idx.numpy():
                        w = int(wi[0])
                        pred_c = int(preds_np[w])
                        true_c = int(labels_np[w])
                        pred_p = float(probs_np[w, pred_c])
                        true_p = float(probs_np[w, true_c])
                        self._log(
                            f"  [LOSS DETAIL] misclassified sample[{w}] "
                            f"true={true_c}(p={true_p:.4f}) pred={pred_c}(p={pred_p:.4f}) "
                            f"loss={ps_np_local[w]:.6f}",
                            level="debug"
                        )

            # 확률 분포 전체 통계 (엔트로피, 최대확률)
            max_probs = tf.reduce_max(probs, axis=1)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-12), axis=1)
            self._log(
                f"  [LOSS DETAIL] max_prob | "
                f"min={tf.reduce_min(max_probs).numpy():.4f} "
                f"max={tf.reduce_max(max_probs).numpy():.4f} "
                f"mean={tf.reduce_mean(max_probs).numpy():.4f}",
                level="debug"
            )
            self._log(
                f"  [LOSS DETAIL] entropy  | "
                f"min={tf.reduce_min(entropy).numpy():.4f} "
                f"max={tf.reduce_max(entropy).numpy():.4f} "
                f"mean={tf.reduce_mean(entropy).numpy():.4f} "
                f"(uniform={np.log(num_classes):.4f})",
                level="debug"
            )

            # Logit 크기 분석 (overflow 위험 감지)
            logit_abs = tf.abs(logits)
            logit_abs_max = tf.reduce_max(logit_abs).numpy()
            logit_abs_mean = tf.reduce_mean(logit_abs).numpy()
            if logit_abs_max > 50:
                self._log(
                    f"  [LOSS DETAIL] WARNING: logit_abs_max={logit_abs_max:.2f} "
                    f"(>50, softmax overflow risk)",
                    level="warning"
                )
            else:
                self._log(
                    f"  [LOSS DETAIL] logit_abs | max={logit_abs_max:.4f} mean={logit_abs_mean:.4f}",
                    level="debug"
                )

    def _log_loss_trend(self, loss_val: float):
        """Loss 이동평균 및 변화 추세"""
        self.__loss_history.append(loss_val)

        # 이전 step 대비 변화량
        if self.__prev_loss is not None:
            delta = loss_val - self.__prev_loss
            delta_pct = (delta / (abs(self.__prev_loss) + 1e-12)) * 100
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            self._log(
                f"  [LOSS TREND] step_delta={delta:+.6f} ({delta_pct:+.2f}%) {direction}",
                level="debug"
            )

        # 이동평균
        if len(self.__loss_history) >= 2:
            ma = np.mean(list(self.__loss_history))
            ma_std = np.std(list(self.__loss_history))
            deviation = (loss_val - ma) / (ma_std + 1e-12)
            self._log(
                f"  [LOSS TREND] moving_avg({len(self.__loss_history)})={ma:.6f} "
                f"std={ma_std:.6f} current_deviation={deviation:+.2f}σ",
                level="debug"
            )

            # 급격한 loss 스파이크 감지 (3σ 이상)
            if abs(deviation) > 3.0:
                self._log(
                    f"  [LOSS TREND] SPIKE DETECTED: loss={loss_val:.6f} "
                    f"deviates {abs(deviation):.1f}σ from moving average",
                    level="warning"
                )

        self.__prev_loss = loss_val

    def _log_label_distribution(self, labels: tf.Tensor):
        """배치 내 라벨 분포"""
        if len(tf.shape(labels)) == 1 and labels.dtype in (tf.int32, tf.int64):
            unique, _, counts = tf.unique_with_counts(labels)
            dist = {int(u): int(c) for u, c in zip(unique.numpy(), counts.numpy())}
            self._log(
                f"  [LABEL DIST] classes={len(dist)} distribution={dist}",
                level="debug"
            )

    def _log_logit_distribution(self, output: tf.Tensor, labels: tf.Tensor):
        """클래스별 logit 통계"""
        if len(output.shape) == 2 and output.shape[1] is not None and output.shape[1] > 1:
            logits = tf.cast(output, tf.float32)
            num_classes = output.shape[1]

            class_means = tf.reduce_mean(logits, axis=0)
            class_stds = tf.math.reduce_std(logits, axis=0) if output.shape[0] > 1 else tf.zeros(num_classes)

            if num_classes <= 20:
                for c in range(num_classes):
                    self._log(
                        f"  [LOGIT DIST] class[{c}] mean={class_means[c].numpy():.4f} "
                        f"std={class_stds[c].numpy():.4f}",
                        level="debug"
                    )
            else:
                self._log(
                    f"  [LOGIT DIST] class_mean | "
                    f"min={tf.reduce_min(class_means).numpy():.4f} "
                    f"max={tf.reduce_max(class_means).numpy():.4f} "
                    f"range={tf.reduce_max(class_means).numpy() - tf.reduce_min(class_means).numpy():.4f}",
                    level="debug"
                )

    # ------------------------------------------------------------------
    # Gradient diagnostics
    # ------------------------------------------------------------------
    def _check_gradients(self, gradients):
        nan_layers = []
        total_grad_norm = 0.0
        for var, grad in zip(self.__model.trainable_variables, gradients):
            if grad is None:
                continue
            g = tf.cast(grad, tf.float32)
            has_nan = tf.reduce_any(tf.math.is_nan(g)).numpy()
            has_inf = tf.reduce_any(tf.math.is_inf(g)).numpy()
            grad_norm = tf.norm(g).numpy()
            total_grad_norm += grad_norm ** 2

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

        total_grad_norm = total_grad_norm ** 0.5
        self._log(
            f"  [GRAD TOTAL] global_norm={total_grad_norm:.6f}",
            level="debug"
        )
        if total_grad_norm > 100:
            self._log(
                f"  [GRAD TOTAL] WARNING: gradient explosion risk (norm={total_grad_norm:.2f})",
                level="warning"
            )
        if nan_layers:
            self._log(
                f"  [GRAD SUMMARY] NaN/Inf detected in: {nan_layers}",
                level="error"
            )

    # ------------------------------------------------------------------
    # Weight update diagnostics
    # ------------------------------------------------------------------
    def _log_weight_update(self, param_snapshot: dict):
        """Optimizer step 전후 파라미터 변화량 측정"""
        total_update_norm = 0.0
        total_param_norm = 0.0

        for var in self.__model.trainable_variables:
            if var.name not in param_snapshot:
                continue
            old = param_snapshot[var.name]
            delta = tf.cast(var - old, tf.float32)
            update_norm = tf.norm(delta).numpy()
            param_norm = tf.norm(tf.cast(var, tf.float32)).numpy()
            total_update_norm += update_norm ** 2
            total_param_norm += param_norm ** 2

            ratio = update_norm / (param_norm + 1e-12)

            self._log(
                f"  [WEIGHT UPDATE] {var.name} | "
                f"update_norm={update_norm:.8f} param_norm={param_norm:.4f} "
                f"ratio={ratio:.8f}",
                level="debug"
            )

        total_update_norm = total_update_norm ** 0.5
        total_param_norm = total_param_norm ** 0.5
        total_ratio = total_update_norm / (total_param_norm + 1e-12)

        self._log(
            f"  [WEIGHT UPDATE TOTAL] update_norm={total_update_norm:.8f} "
            f"param_norm={total_param_norm:.4f} ratio={total_ratio:.8f}",
            level="debug"
        )

        if total_ratio > 0.1:
            self._log(
                f"  [WEIGHT UPDATE] WARNING: large update ratio={total_ratio:.6f} "
                f"(learning rate may be too high)",
                level="warning"
            )
        elif total_ratio < 1e-8:
            self._log(
                f"  [WEIGHT UPDATE] WARNING: negligible update ratio={total_ratio:.10f} "
                f"(learning rate may be too low or vanishing gradients)",
                level="warning"
            )

    def builder(self):
        return self
