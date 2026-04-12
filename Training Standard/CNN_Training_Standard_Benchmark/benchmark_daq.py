"""
CNN Training Standard 벤치마크 (DAQ)
benchmark_config.json의 실험 설정을 DAQ 서버에서 순차 실행하고 결과를 비교합니다.
"""

##$--
parameters = '''{
    "version" : "CNN_Training_Standard_Benchmark_v1.0.3",
    "hyperparameter":{
        "framework" : "pytorch",
        "epoch" : 10,
        "save_epoch" : 0,
        "batch_size" : 16,
        "lr" : 1e-3,
        "optimizer_name" : "Adam",
        "criterion" : "CrossEntropyLoss",
        "input_size" : 224,
        "normalize_mean" : 0.5,
        "normalize_stdev" : 0.5,
        "using_gpu" : true,
        "using_amp" : false,
        "train_ratio" : 0.8,
        "validation_save_random" : false,
        "debug" : false,
        "daq_old_path" : false
    },
    "authentication": {
        "operation_service_address": "",
        "access_token" : ""
    },
    "result":{
        "id":"",
        "volume_id":"default"
    },
    "gt_dataset":{
        "gt_dataset_id" : ""
    },
    "chunk_size" : 100000
}'''
##$--

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CNN_Training_Standard'))

import json
import time
import logging

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

from Builder import Json_HyperparameterBuilder
from Builder import get_framework_builders, get_daq_framework_builders
from Builder import TrainHook
from Builder.Model.Pytorch_Classification_Models import _TORCHVISION_MODELS, _CUSTOM_MODELS, _resolve_model

INPUT_SIZES = {
    "inceptionv3": 299, "inceptionv4": 299,
    "efficientnet_b1": 240, "efficientnet_b2": 260, "efficientnet_b3": 300,
    "efficientnet_b4": 380, "efficientnet_b5": 456, "efficientnet_b6": 528, "efficientnet_b7": 600,
    "efficientnet_v2_s": 384, "efficientnet_v2_m": 480, "efficientnet_v2_l": 480,
    "swin_v2_t": 256, "swin_v2_s": 256, "swin_v2_b": 256,
    "vit_h_14": 518,
}

# benchmark_config.json 로드
_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_config.json')
if os.path.exists(_config_path):
    with open(_config_path, 'r', encoding='utf-8') as f:
        _benchmark_config = json.load(f)
else:
    _benchmark_config = None


class BenchmarkHook(TrainHook):
    def __init__(self):
        self.history = []
    def training_start(self): pass
    def training_end(self): pass
    def on_epoch_end(self, total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time, model):
        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "val_acc": (100 - validation_loss) if validation_loss is not None else None,
            "time": epoch_elapsed_time,
        })


def run_single_experiment(exp_params, operation_builder, classcode_builder, framework):
    """단일 실험 DAQ 실행"""
    network_name = exp_params['network_name']
    input_size = exp_params.get('input_size', INPUT_SIZES.get(network_name, 224))
    device = 'cuda' if framework == 'pytorch' else '/gpu:0'

    Model, _, TrainingBuilder, _ = get_framework_builders(framework)
    DAQDatasetBuilder = get_daq_framework_builders(framework)

    dataset_builder = DAQDatasetBuilder(logger=logger).init_url_info(
        operation_channel=operation_builder.get_operation_channel(),
        access_token=operation_builder.get_access_token()
    ).init_dataset_gts(
        gt_dataset_id=operation_builder.get_gt_dataset_id()
    ).init_transform(
        input_size=input_size,
        normalize_mean=exp_params.get('normalize_mean', 0.5),
        normalize_stdev=exp_params.get('normalize_stdev', 0.5)
    ).build()

    dataset_builder.create_train_dataset(
        train_ratio=exp_params.get('train_ratio', 0.8),
        batch_size=exp_params['batch_size'],
        validation_save_random=False,
        class_code_info=classcode_builder.get_class_code_info()
    )

    if dataset_builder.success() is False:
        raise RuntimeError("Dataset Build Failed")

    try:
        model_builder = Model().init_device(device)
        if framework == "tensorflow":
            model = model_builder.init_model(
                num_classes=classcode_builder.get_class_count(),
                network_name=network_name,
                input_size=input_size
            ).get_model()
        else:
            model = model_builder.init_model(
                num_classes=classcode_builder.get_class_count(),
                network_name=network_name
            ).get_model()

        training_builder = TrainingBuilder(None
            ).initialize(
                epoch_total=exp_params['epoch'],
                device=device,
                using_amp=exp_params.get('using_amp', False)
            ).init_model(model=model
            ).init_optimizer(
                optimizer_name=exp_params['optimizer_name'],
                lr=exp_params['lr']
            ).init_criterion(
                criterion_name=exp_params.get('criterion', 'CrossEntropyLoss')
            ).builder()

        hook = BenchmarkHook()
        training_builder.train(
            train_data_loader=dataset_builder.get_train_data_loader(),
            validation_data_loader=dataset_builder.get_validation_data_loader(),
            hook=hook
        )
        return hook.history
    finally:
        dataset_builder.temp_folder_delete()


def RecipeRun(**kwargs):
    from Builder import Operation_Builder
    from Builder import DAQ_Classification_ClassCodeBuilder

    framework = kwargs['hyperparameter'].get('framework', 'pytorch')

    operation_builder = Operation_Builder(**kwargs).initialize().build()
    classcode_builder = DAQ_Classification_ClassCodeBuilder().init_url_info(
        operation_channel=operation_builder.get_operation_channel(),
        access_token=operation_builder.get_access_token()
    ).init_label_data(
        gt_dataset_id=operation_builder.get_gt_dataset_id()
    ).build()

    # config 로드
    if _benchmark_config:
        common = _benchmark_config.get('common', {})
        experiments = _benchmark_config.get('experiments', [])
    else:
        # config 없으면 parameter 기반 단일 실험
        hp = kwargs['hyperparameter']
        common = {}
        experiments = [{"name": hp.get('network_name', 'default'), **hp}]

    logger.info(f"{'='*60}")
    logger.info(f"CNN Training Standard - DAQ Benchmark")
    logger.info(f"Framework: {framework}")
    logger.info(f"Experiments: {len(experiments)}")
    logger.info(f"{'='*60}\n")

    all_results = {}
    passed = 0
    failed = 0

    for i, exp in enumerate(experiments, 1):
        exp_name = exp.get('name', f"exp_{i}")
        params = {**common, **exp}

        logger.info(f"[{i}/{len(experiments)}] {exp_name}")
        logger.info(f"  network: {params['network_name']}, optimizer: {params['optimizer_name']}, "
                     f"lr: {params['lr']}, batch: {params['batch_size']}, epoch: {params['epoch']}")

        start = time.time()
        try:
            history = run_single_experiment(params, operation_builder, classcode_builder, framework)
            elapsed = time.time() - start
            last = history[-1]
            logger.info(f"  final: train_loss={last['train_loss']:.4f}, val_loss={last['validation_loss']:.4f if last['validation_loss'] else 'N/A'}")
            logger.info(f"  PASS ({elapsed:.1f}s)\n")
            all_results[exp_name] = history
            passed += 1
        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e).split('\n')[0][:100]
            logger.info(f"  FAIL ({elapsed:.1f}s): {error_msg}\n")
            all_results[exp_name] = []
            failed += 1

    # 결과 비교 출력
    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARK RESULTS ({passed} passed, {failed} failed)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Experiment':<35} {'Val Loss':>10} {'Val Acc':>10} {'Time':>8}")
    logger.info(f"{'-'*35} {'-'*10} {'-'*10} {'-'*8}")

    ranked = []
    for exp_name, history in all_results.items():
        if not history:
            logger.info(f"{exp_name:<35} {'FAIL':>10} {'':>10} {'':>8}")
            continue
        val_losses = [h['validation_loss'] for h in history if h['validation_loss'] is not None]
        val_accs = [h['val_acc'] for h in history if h['val_acc'] is not None]
        total_time = sum(h['time'] for h in history)
        best_loss = min(val_losses) if val_losses else None
        best_acc = max(val_accs) if val_accs else None
        ranked.append((exp_name, best_loss, best_acc, total_time))

    ranked.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))
    for rank, (name, loss, acc, t) in enumerate(ranked, 1):
        marker = " ★" if rank == 1 else ""
        logger.info(f"{name:<35} {loss:>10.4f} {acc:>9.2f}% {t:>7.1f}s{marker}")

    logger.info(f"\n{'='*60}")


if __name__ == "__main__":
    kwargs = json.loads(parameters)
    kwargs['authentication']['operation_service_address'] = ""
    kwargs['authentication']['access_token'] = ""
    kwargs['gt_dataset']['gt_dataset_id'] = r""
    import random
    kwargs['result']['id'] = f"benchmark_{random.randint(1000,9999)}"

    RecipeRun(**kwargs)
