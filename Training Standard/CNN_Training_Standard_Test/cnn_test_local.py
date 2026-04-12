"""
CNN Training Standard 전체 네트워크 테스트
등록된 모든 모델에 대해 1 epoch 학습이 정상 동작하는지 확인합니다.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CNN_Training_Standard'))

import json
import time
import logging
import traceback

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'create_dataset', 'dataset')
output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

from Builder import Json_HyperparameterBuilder, Local_Classification_ClassCodeBuilder
from Builder import get_framework_builders
from Builder import TrainHook, Local_SaveBuilder

# 테스트할 모델 목록 자동 수집
from Builder.Model.Pytorch_Classification_Models import _TORCHVISION_MODELS, _CUSTOM_MODELS, _resolve_model

PYTORCH_MODELS = [name for name in list(_CUSTOM_MODELS.keys()) + list(_TORCHVISION_MODELS.keys()) if _resolve_model(name) is not None]

# 모델별 input_size
INPUT_SIZES = {
    "inceptionv3": 299, "inceptionv4": 299,
    "efficientnet_b1": 240, "efficientnet_b2": 260, "efficientnet_b3": 300,
    "efficientnet_b4": 380, "efficientnet_b5": 456, "efficientnet_b6": 528, "efficientnet_b7": 600,
    "efficientnet_v2_s": 384, "efficientnet_v2_m": 480, "efficientnet_v2_l": 480,
    "swin_v2_t": 256, "swin_v2_s": 256, "swin_v2_b": 256,
    "vit_h_14": 518,
}


class MinimalSaveHook(TrainHook):
    """테스트용 최소 SaveHook - 저장 없이 로그만"""
    def training_start(self): pass
    def on_epoch_end(self, total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time, model):
        logger.info(f"    train_loss: {train_loss:.4f}, val_loss: {validation_loss:.4f if validation_loss else 'N/A'}, time: {epoch_elapsed_time:.1f}s")
    def training_end(self): pass


def test_model(network_name, framework="pytorch"):
    """단일 모델 테스트: 모델 생성 → 1 epoch 학습"""
    input_size = INPUT_SIZES.get(network_name, 224)

    params = {
        "framework": framework,
        "network_name": network_name,
        "epoch": 1,
        "save_epoch": 0,
        "batch_size": 2,
        "lr": 1e-3,
        "optimizer_name": "Adam",
        "criterion": "CrossEntropyLoss",
        "input_size": input_size,
        "normalize_mean": 0.5,
        "normalize_stdev": 0.5,
        "using_gpu": False,
        "using_amp": False,
        "train_ratio": 0.8,
        "validation_save_random": False,
    }

    hp = Json_HyperparameterBuilder(json.dumps(params)).build()
    classcode = Local_Classification_ClassCodeBuilder().init_label_data(dataset_path=dataset).build()
    Model, LocalDatasetBuilder, TrainingBuilder, _ = get_framework_builders(framework)

    dataset_builder = LocalDatasetBuilder(logger=None).init_dataset_path(dataset_path=dataset
        ).init_transform(input_size=input_size, normalize_mean=0.5, normalize_stdev=0.5
        ).create_train_dataset(
            train_ratio=0.8, batch_size=2,
            validation_save_random=False,
            class_code_info=classcode.get_class_code_info()
        )

    model_builder = Model().init_device('cpu')
    model = model_builder.init_model(
        num_classes=classcode.get_class_count(),
        network_name=network_name
    ).get_model()

    training_builder = TrainingBuilder(None
        ).initialize(epoch_total=1, device='cpu', using_amp=False
        ).init_model(model=model
        ).init_optimizer(optimizer_name='Adam', lr=1e-3
        ).init_criterion(criterion_name='CrossEntropyLoss'
        ).builder()

    training_builder.train(
        train_data_loader=dataset_builder.get_train_data_loader(),
        validation_data_loader=dataset_builder.get_validation_data_loader(),
        hook=MinimalSaveHook()
    )


def run_all_tests():
    logger.info(f"{'='*60}")
    logger.info(f"CNN Training Standard - Network Test")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Available PyTorch models: {len(PYTORCH_MODELS)}")
    logger.info(f"{'='*60}\n")

    results = {"pass": [], "fail": [], "skip": []}

    for i, name in enumerate(PYTORCH_MODELS, 1):
        logger.info(f"[{i}/{len(PYTORCH_MODELS)}] Testing: {name}")
        start = time.time()
        try:
            test_model(name)
            elapsed = time.time() - start
            logger.info(f"  ✓ PASS ({elapsed:.1f}s)\n")
            results["pass"].append(name)
        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e).split('\n')[0][:100]
            logger.info(f"  ✗ FAIL ({elapsed:.1f}s): {error_msg}\n")
            results["fail"].append((name, error_msg))

    # 결과 요약
    logger.info(f"{'='*60}")
    logger.info(f"RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  PASS: {len(results['pass'])}")
    logger.info(f"  FAIL: {len(results['fail'])}")

    if results["fail"]:
        logger.info(f"\nFailed models:")
        for name, err in results["fail"]:
            logger.info(f"  ✗ {name}: {err}")

    if results["pass"]:
        logger.info(f"\nPassed models:")
        for name in results["pass"]:
            logger.info(f"  ✓ {name}")

    logger.info(f"\n{'='*60}")
    total = len(results['pass']) + len(results['fail'])
    logger.info(f"Total: {len(results['pass'])}/{total} passed")


if __name__ == "__main__":
    run_all_tests()
