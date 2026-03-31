import time
import logging
import numpy as np
import torch
import torch.nn as nn
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
    각 step마다 입력·출력·gradient 통계를 로그로 출력합니다.

    log_every_n_iterations : int
        몇 iteration 마다 상세 로그를 출력할지 (기본 1 = 매 iteration)
    log_layer_hooks : bool
        True 이면 모든 레이어의 forward 출력 통계를 로그로 출력
    stop_on_nan : bool
        True 이면 NaN loss 감지 시 즉시 RuntimeError 발생
    """

    def __init__(self, logger=None, log_every_n_iterations: int = 1,
                 log_layer_hooks: bool = True, stop_on_nan: bool = True):
        self.__external_logger = logger
        self.__log_every_n = log_every_n_iterations
        self.__log_layer_hooks = log_layer_hooks
        self.__stop_on_nan = stop_on_nan

        self.__epoch_total = None
        self.__device = 'cpu'
        self.__using_amp = False

        self.__optimizer = None
        self.__criterion = None
        self.__model = None

        self.__iteration_start_time = None
        self.__epoch_start_time = None

        self.__hook_handles = []

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
    def initialize(self, epoch_total: int, using_amp, device: str = 'cpu'):
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
        else:
            raise Exception("Invalid Criterion Option")
        self.__criterion = criterion
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
                do_log = (n_iter % max(1, self.__log_every_n) == 0) or (n_iter == total_iteration)

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
        inputs = batch[0].to(self.__device)
        labels = batch[1].to(self.__device)

        if do_log:
            self._log(_tensor_stats("  [INPUT ]", inputs), level="debug")
            self._log(_tensor_stats("  [LABEL ]", labels.float()), level="debug")

        with torch.cuda.amp.autocast(enabled=self.__using_amp):
            outputs = self.__model(inputs)
            output = outputs[0] if isinstance(outputs, tuple) else outputs

            if do_log:
                self._log(_tensor_stats("  [OUTPUT]", output), level="debug")

            loss = self.__criterion(output, labels)

        if do_log:
            self._log(f"  [LOSS  ] value={loss.item():.6f} nan={torch.isnan(loss).item()} inf={torch.isinf(loss).item()}", level="debug")

        self.__optimizer.zero_grad()
        loss.backward()

        # Gradient 검사
        if do_log:
            self._check_gradients()

        self.__optimizer.step()
        return inputs, loss

    def _valid_step(self, batch):
        inputs = batch[0].to(self.__device)
        labels = batch[1].to(self.__device)

        with torch.cuda.amp.autocast(enabled=self.__using_amp):
            outputs = self.__model(inputs)
            valid_outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = self.__criterion(valid_outputs, labels)
        return inputs, loss

    # ------------------------------------------------------------------
    # Gradient diagnostics
    # ------------------------------------------------------------------
    def _check_gradients(self):
        nan_layers = []
        for name, param in self.__model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.detach().float()
            has_nan = torch.isnan(g).any().item()
            has_inf = torch.isinf(g).any().item()
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
        if nan_layers:
            self._log(
                f"  [GRAD SUMMARY] NaN/Inf detected in: {nan_layers}",
                level="error"
            )

    def builder(self):
        return self
