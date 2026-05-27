import time
import logging
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .TrainHook import TrainHook

logging.basicConfig(
    format="[DEBUG][%(levelname)s] %(message)s",
    level=logging.DEBUG
)

def _tensor_stats(name: str, t: torch.Tensor) -> str:
    t_f = t.detach().float()
    has_nan = torch.isnan(t_f).any().item()
    has_inf = torch.isinf(t_f).any().item()
    return (
        f"{name} | shape={list(t.shape)} dtype={t.dtype} "
        f"min={t_f.min().item():.6f} max={t_f.max().item():.6f} "
        f"mean={t_f.mean().item():.6f} std={t_f.std().item():.6f} "
        f"nan={has_nan} inf={has_inf}"
    )


class Pytorch_TrainingBuilder_Debug:
    """
    NaN 손실 디버깅용 TrainingBuilder.
    각 step마다 입력·출력·gradient·loss 상세 통계를 로그로 출력합니다.

    log_every_n_iterations : int
        몇 iteration 마다 상세 로그를 출력할지 (기본 1 = 매 iteration)
    log_layer_hooks : bool
        True 이면 모든 레이어의 forward 출력 통계를 로그로 출력
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
        self.__log_layer_hooks = log_layer_hooks
        self.__stop_on_nan = stop_on_nan

        self.__epoch_total = None
        self.__device = 'cpu'
        self.__using_amp = False

        self.__optimizer = None
        self.__criterion = None
        self.__criterion_unreduced = None  # per-sample loss 계산용
        self.__model = None

        self.__iteration_start_time = None
        self.__epoch_start_time = None

        self.__hook_handles = []

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
    # Layer-level forward hook
    # ------------------------------------------------------------------
    def _register_layer_hooks(self):
        """모든 named module에 forward hook을 등록합니다."""
        def make_hook(layer_name):
            def hook(module, input, output):
                # output이 tuple일 경우 첫 번째 tensor만 검사
                out = output[0] if isinstance(output, tuple) else output
                if not isinstance(out, torch.Tensor):
                    return
                has_nan = torch.isnan(out.detach().float()).any().item()
                has_inf = torch.isinf(out.detach().float()).any().item()
                if has_nan or has_inf:
                    self._log(
                        f"[LAYER NaN/Inf] {layer_name} -> {_tensor_stats('output', out)}",
                        level="warning"
                    )
                else:
                    self._log(
                        f"[LAYER] {layer_name} -> {_tensor_stats('output', out)}",
                        level="debug"
                    )
            return hook

        for name, module in self.__model.named_modules():
            if name == '':
                continue  # 최상위 모듈 skip
            handle = module.register_forward_hook(make_hook(name))
            self.__hook_handles.append(handle)

    def _remove_layer_hooks(self):
        for h in self.__hook_handles:
            h.remove()
        self.__hook_handles.clear()

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------
    def initialize(self, epoch_total: int, using_amp, device: str = 'cpu', loss_eps: float = 0.0):
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
            criterion_unreduced = nn.CrossEntropyLoss(reduction='none')
        elif criterion_name.lower() == "bcewithlogitsloss":
            criterion = nn.BCEWithLogitsLoss()
            criterion_unreduced = nn.BCEWithLogitsLoss(reduction='none')
        elif criterion_name.lower() == "focalloss":
            criterion = nn.BCEWithLogitsLoss()
            criterion_unreduced = nn.BCEWithLogitsLoss(reduction='none')
        elif criterion_name.lower() == "mse":
            criterion = nn.MSELoss()
            criterion_unreduced = nn.MSELoss(reduction='none')
        else:
            raise Exception("Invalid Criterion Option")
        self.__criterion = criterion
        self.__criterion_unreduced = criterion_unreduced
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

        if self.__log_layer_hooks:
            self._log("[DEBUG] Registering layer forward hooks...", level="info")
            self._register_layer_hooks()

        self.__iteration_start_time = time.time()

        for epoch in range(1, self.__epoch_total + 1):
            self.__epoch_start_time = time.time()
            self._log(f"========== Epoch {epoch}/{self.__epoch_total} ==========", level="info")

            train_epoch_loss = 0.0

            for n_iter, batch in enumerate(train_data_loader, 1):
                do_log = epoch >= 20 and ((n_iter % max(1, self.__log_every_n) == 0) or (n_iter == total_iteration))

                if do_log:
                    self._log(f"[TRAIN] Epoch {epoch}, Iter {n_iter}/{total_iteration} - step start", level="debug")

                inputs, loss = self._train_step(batch, n_iter, do_log)

                loss_val = loss.item()
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
            if validation_data_loader:
                with torch.no_grad():
                    self.__model.eval()
                    for n_valid, valid_batch in enumerate(validation_data_loader, 1):
                        _, valid_loss = self._valid_step(valid_batch)
                        valid_epoch_loss += valid_loss.item()
                    self.__model.train()

            avg_train = train_epoch_loss / n_iter
            avg_valid = (valid_epoch_loss / len(validation_data_loader)
                         if validation_data_loader else None)

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

        if self.__log_layer_hooks:
            self._remove_layer_hooks()

        if hook:
            hook.training_end()
        return self

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------
    def _train_step(self, batch, n_iter: int, do_log: bool):
        self.__global_step += 1
        inputs = batch[0].to(self.__device)
        labels = batch[1].to(self.__device)

        if do_log:
            self._log(_tensor_stats("  [INPUT ]", inputs), level="debug")
            self._log(_tensor_stats("  [LABEL ]", labels.float()), level="debug")
            self._log_label_distribution(labels)

        # 파라미터 스냅샷 (weight update 크기 측정용)
        param_snapshot = None
        if do_log:
            param_snapshot = {name: p.data.clone() for name, p in self.__model.named_parameters() if p.requires_grad}

        with torch.amp.autocast(device_type=self.__device, enabled=self.__using_amp):
            outputs = self.__model(inputs)
            output = outputs[0] if isinstance(outputs, tuple) else outputs

            if do_log:
                self._log(_tensor_stats("  [OUTPUT]", output), level="debug")
                self._log_logit_distribution(output, labels)

            loss = self.__criterion(output, labels)
            if self.__loss_eps > 0:
                loss = loss + self.__loss_eps

        if do_log:
            # 기본 loss 정보
            self._log(
                f"  [LOSS  ] value={loss.item():.6f} "
                f"nan={torch.isnan(loss).item()} inf={torch.isinf(loss).item()}",
                level="debug"
            )
            # 상세 loss 분석
            self._log_loss_detail(output, labels, loss)
            # loss 추세
            self._log_loss_trend(loss.item())

        self.__optimizer.zero_grad()
        self.__scaler.scale(loss).backward()

        # Gradient 검사
        if do_log:
            self._check_gradients()

        self.__scaler.step(self.__optimizer)
        self.__scaler.update()

        # Weight update 크기 측정
        if do_log and param_snapshot is not None:
            self._log_weight_update(param_snapshot)

        return inputs, loss

    def _valid_step(self, batch):
        inputs = batch[0].to(self.__device)
        labels = batch[1].to(self.__device)

        with torch.amp.autocast(device_type=self.__device, enabled=self.__using_amp):
            outputs = self.__model(inputs)
            valid_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = self.__criterion(valid_outputs, labels)
        return inputs, loss

    # ------------------------------------------------------------------
    # Loss detail diagnostics
    # ------------------------------------------------------------------
    def _log_loss_detail(self, output: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        """Per-sample loss, softmax 확률 분포, prediction-target 비교 등 상세 분석"""
        batch_size = output.shape[0]

        # --- Per-sample loss ---
        with torch.no_grad():
            per_sample = self.__criterion_unreduced(output.detach().float(), labels)
            if per_sample.dim() > 1:
                per_sample = per_sample.mean(dim=tuple(range(1, per_sample.dim())))

            ps_min = per_sample.min().item()
            ps_max = per_sample.max().item()
            ps_mean = per_sample.mean().item()
            ps_std = per_sample.std().item() if batch_size > 1 else 0.0
            ps_median = per_sample.median().item()

            self._log(
                f"  [LOSS DETAIL] per_sample | "
                f"min={ps_min:.6f} max={ps_max:.6f} mean={ps_mean:.6f} "
                f"std={ps_std:.6f} median={ps_median:.6f}",
                level="debug"
            )

            # 가장 큰 loss를 가진 샘플 Top-3
            if batch_size >= 3:
                top_vals, top_idx = per_sample.topk(min(3, batch_size))
                top_info = ", ".join(
                    [f"sample[{idx.item()}]={val.item():.6f}" for val, idx in zip(top_vals, top_idx)]
                )
                self._log(f"  [LOSS DETAIL] worst_samples | {top_info}", level="debug")

            # --- Softmax 확률 분석 (classification) ---
            if output.dim() == 2 and output.shape[1] > 1:
                probs = F.softmax(output.detach().float(), dim=1)
                num_classes = output.shape[1]

                # 정답 클래스의 확률
                if labels.dim() == 1 and labels.dtype in (torch.long, torch.int):
                    target_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                    self._log(
                        f"  [LOSS DETAIL] target_class_prob | "
                        f"min={target_probs.min().item():.6f} max={target_probs.max().item():.6f} "
                        f"mean={target_probs.mean().item():.6f}",
                        level="debug"
                    )

                    # Top-1 정확도
                    preds = output.detach().argmax(dim=1)
                    correct = (preds == labels).float()
                    top1_acc = correct.mean().item() * 100
                    self._log(
                        f"  [LOSS DETAIL] batch_top1_acc={top1_acc:.1f}% "
                        f"({int(correct.sum().item())}/{batch_size})",
                        level="debug"
                    )

                    # 오분류 샘플 상세
                    wrong_mask = (preds != labels)
                    n_wrong = wrong_mask.sum().item()
                    if n_wrong > 0 and n_wrong <= 10:
                        wrong_idx = wrong_mask.nonzero(as_tuple=True)[0]
                        for wi in wrong_idx:
                            w = wi.item()
                            pred_c = preds[w].item()
                            true_c = labels[w].item()
                            pred_p = probs[w, pred_c].item()
                            true_p = probs[w, true_c].item()
                            self._log(
                                f"  [LOSS DETAIL] misclassified sample[{w}] "
                                f"true={true_c}(p={true_p:.4f}) pred={pred_c}(p={pred_p:.4f}) "
                                f"loss={per_sample[w].item():.6f}",
                                level="debug"
                            )

                # 확률 분포 전체 통계 (엔트로피, 최대확률)
                max_probs = probs.max(dim=1).values
                entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)
                self._log(
                    f"  [LOSS DETAIL] max_prob | "
                    f"min={max_probs.min().item():.4f} max={max_probs.max().item():.4f} "
                    f"mean={max_probs.mean().item():.4f}",
                    level="debug"
                )
                self._log(
                    f"  [LOSS DETAIL] entropy  | "
                    f"min={entropy.min().item():.4f} max={entropy.max().item():.4f} "
                    f"mean={entropy.mean().item():.4f} "
                    f"(uniform={np.log(num_classes):.4f})",
                    level="debug"
                )

                # Logit 크기 분석 (overflow 위험 감지)
                logit_abs_max = output.detach().float().abs().max().item()
                logit_abs_mean = output.detach().float().abs().mean().item()
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

    def _log_label_distribution(self, labels: torch.Tensor):
        """배치 내 라벨 분포"""
        if labels.dim() == 1 and labels.dtype in (torch.long, torch.int):
            unique, counts = labels.unique(return_counts=True)
            dist = {int(u.item()): int(c.item()) for u, c in zip(unique, counts)}
            self._log(
                f"  [LABEL DIST] classes={len(dist)} distribution={dist}",
                level="debug"
            )

    def _log_logit_distribution(self, output: torch.Tensor, labels: torch.Tensor):
        """클래스별 logit 통계"""
        if output.dim() == 2 and output.shape[1] > 1:
            logits = output.detach().float()
            num_classes = logits.shape[1]

            # 전체 class 평균/std
            class_means = logits.mean(dim=0)
            class_stds = logits.std(dim=0) if logits.shape[0] > 1 else torch.zeros(num_classes)

            if num_classes <= 20:
                for c in range(num_classes):
                    self._log(
                        f"  [LOGIT DIST] class[{c}] mean={class_means[c].item():.4f} "
                        f"std={class_stds[c].item():.4f}",
                        level="debug"
                    )
            else:
                self._log(
                    f"  [LOGIT DIST] class_mean | "
                    f"min={class_means.min().item():.4f} max={class_means.max().item():.4f} "
                    f"range={class_means.max().item() - class_means.min().item():.4f}",
                    level="debug"
                )

    # ------------------------------------------------------------------
    # Gradient diagnostics
    # ------------------------------------------------------------------
    def _check_gradients(self):
        nan_layers = []
        total_grad_norm = 0.0
        for name, param in self.__model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.detach().float()
            has_nan = torch.isnan(g).any().item()
            has_inf = torch.isinf(g).any().item()
            grad_norm = g.norm().item()
            total_grad_norm += grad_norm ** 2

            if has_nan or has_inf:
                nan_layers.append(name)
                self._log(
                    f"  [GRAD NaN/Inf] {name} | {_tensor_stats('grad', param.grad)}",
                    level="warning"
                )
            else:
                self._log(
                    f"  [GRAD ] {name} | {_tensor_stats('grad', param.grad)}",
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
        layer_updates = []

        for name, param in self.__model.named_parameters():
            if not param.requires_grad or name not in param_snapshot:
                continue
            old = param_snapshot[name]
            delta = (param.data - old).float()
            update_norm = delta.norm().item()
            param_norm = param.data.float().norm().item()
            total_update_norm += update_norm ** 2
            total_param_norm += param_norm ** 2

            ratio = update_norm / (param_norm + 1e-12)
            layer_updates.append((name, update_norm, param_norm, ratio))

            self._log(
                f"  [WEIGHT UPDATE] {name} | "
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
