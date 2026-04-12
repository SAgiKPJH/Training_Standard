"""
CNN Training Standard 전체 네트워크 DAQ 테스트
DAQ 서버 환경에서 등록된 모든 모델에 대해 1 epoch 학습 테스트.
"""

##$--
parameters = '''{
    "version" : "CNN_Training_Standard_Test_v1.0.3",
    "hyperparameter":{
        "framework" : "pytorch",
        "network_name" : "efficientnet_b0",
        "epoch" : 1,
        "save_epoch" : 0,
        "batch_size" : 2,
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
import json
import time
import logging

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Builder 참조
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CNN_Training_Standard'))

from Builder import Json_HyperparameterBuilder
from Builder import get_framework_builders, get_daq_framework_builders
from Builder import TrainHook
from Builder.Model.Pytorch_Classification_Models import _TORCHVISION_MODELS, _CUSTOM_MODELS, _resolve_model

# 사용 가능한 PyTorch 모델 자동 수집
PYTORCH_MODELS = [n for n in list(_CUSTOM_MODELS.keys()) + list(_TORCHVISION_MODELS.keys()) if _resolve_model(n) is not None]

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
    def training_start(self): pass
    def on_epoch_end(self, total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time, model):
        logger.info(f"    train_loss: {train_loss:.4f}, val_loss: {validation_loss:.4f if validation_loss else 'N/A'}, time: {epoch_elapsed_time:.1f}s")
    def training_end(self): pass


def test_single_model(network_name, operation_builder, classcode_builder, framework="pytorch"):
    """단일 모델 DAQ 테스트"""
    input_size = INPUT_SIZES.get(network_name, 224)

    Model, _, TrainingBuilder, _ = get_framework_builders(framework)
    DAQDatasetBuilder = get_daq_framework_builders(framework)

    dataset_builder = DAQDatasetBuilder(logger=logger).init_url_info(
        operation_channel=operation_builder.get_operation_channel(),
        access_token=operation_builder.get_access_token()
    ).init_dataset_gts(
        gt_dataset_id=operation_builder.get_gt_dataset_id()
    ).init_transform(
        input_size=input_size, normalize_mean=0.5, normalize_stdev=0.5
    ).build()

    dataset_builder.create_train_dataset(
        train_ratio=0.8, batch_size=2,
        validation_save_random=False,
        class_code_info=classcode_builder.get_class_code_info()
    )

    if dataset_builder.success() is False:
        raise RuntimeError("Dataset Build Failed")

    try:
        model_builder = Model().init_device('cuda' if framework == 'pytorch' else '/gpu:0')
        model = model_builder.init_model(
            num_classes=classcode_builder.get_class_count(),
            network_name=network_name
        ).get_model()

        training_builder = TrainingBuilder(None
            ).initialize(epoch_total=1, device='cuda' if framework == 'pytorch' else '/gpu:0', using_amp=False
            ).init_model(model=model
            ).init_optimizer(optimizer_name='Adam', lr=1e-3
            ).init_criterion(criterion_name='CrossEntropyLoss'
            ).builder()

        training_builder.train(
            train_data_loader=dataset_builder.get_train_data_loader(),
            validation_data_loader=dataset_builder.get_validation_data_loader(),
            hook=MinimalSaveHook()
        )
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

    models = PYTORCH_MODELS
    logger.info(f"{'='*60}")
    logger.info(f"CNN Training Standard - DAQ Network Test")
    logger.info(f"Framework: {framework}")
    logger.info(f"Available models: {len(models)}")
    logger.info(f"{'='*60}\n")

    results = {"pass": [], "fail": []}

    for i, name in enumerate(models, 1):
        logger.info(f"[{i}/{len(models)}] Testing: {name}")
        start = time.time()
        try:
            test_single_model(name, operation_builder, classcode_builder, framework)
            elapsed = time.time() - start
            logger.info(f"  PASS ({elapsed:.1f}s)\n")
            results["pass"].append(name)
        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e).split('\n')[0][:100]
            logger.info(f"  FAIL ({elapsed:.1f}s): {error_msg}\n")
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
            logger.info(f"  X {name}: {err}")

    total = len(results['pass']) + len(results['fail'])
    logger.info(f"\nTotal: {len(results['pass'])}/{total} passed")


if __name__ == "__main__":
    kwargs = json.loads(parameters)
    kwargs['authentication']['operation_service_address'] = ""
    kwargs['authentication']['access_token'] = ""
    kwargs['gt_dataset']['gt_dataset_id'] = r""
    import random
    kwargs['result']['id'] = f"network_test_{random.randint(1000,9999)}"

    RecipeRun(**kwargs)
