"""
CNN Training Standard 전체 네트워크 로컬 테스트
network_list.json에 등록된 모든 네트워크에 대해 1 epoch 학습 테스트.
Dataset은 input_size별로 한 번만 생성하여 공유합니다.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CNN_Training_Standard'))

import json
import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'create_dataset', 'dataset')

from Builder import Local_Classification_ClassCodeBuilder
from Builder import get_framework_builders
from Builder import TrainHook

# network_list.json에서 모델 목록 로드
_list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'network_list.json')
with open(_list_path, 'r', encoding='utf-8') as f:
    _network_list = json.load(f)

PYTORCH_MODELS = [(m['name'], m['input_size']) for m in _network_list['pytorch']]
TENSORFLOW_MODELS = [(m['name'], m['input_size']) for m in _network_list['tensorflow']]


class MinimalSaveHook(TrainHook):
    def training_start(self): pass
    def on_epoch_end(self, total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time, model):
        val_str = f"{validation_loss:.4f}" if validation_loss is not None else "N/A"
        logger.info(f"    train_loss: {train_loss:.4f}, val_loss: {val_str}, time: {epoch_elapsed_time:.1f}s")
    def training_end(self): pass


def create_dataset_cached(framework, input_size, classcode, cache):
    """input_size별 dataset 캐시. 동일 input_size면 재사용."""
    cache_key = f"{framework}_{input_size}"
    if cache_key in cache:
        return cache[cache_key]

    logger.info(f"    Creating dataset (input_size={input_size})...")
    _, LocalDatasetBuilder, _, _ = get_framework_builders(framework)

    dataset_builder = LocalDatasetBuilder(logger=logger).init_dataset_path(dataset_path=dataset
        ).init_transform(input_size=input_size, normalize_mean=0.5, normalize_stdev=0.5
        ).create_train_dataset(
            train_ratio=0.8, batch_size=2,
            validation_save_random=False,
            class_code_info=classcode.get_class_code_info()
        )

    cache[cache_key] = dataset_builder
    return dataset_builder


def test_model(network_name, input_size, framework, classcode, dataset_cache):
    Model, _, TrainingBuilder, _ = get_framework_builders(framework)

    dataset_builder = create_dataset_cached(framework, input_size, classcode, dataset_cache)

    model_builder = Model().init_device('cpu')
    if framework == "tensorflow":
        model = model_builder.init_model(
            num_classes=classcode.get_class_count(),
            network_name=network_name,
            input_size=input_size
        ).get_model()
    else:
        model = model_builder.init_model(
            num_classes=classcode.get_class_count(),
            network_name=network_name
        ).get_model()

    training_builder = TrainingBuilder(logger
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


def run_framework_tests(framework, models, classcode, dataset_cache):
    results = {"pass": [], "fail": []}

    for i, (name, input_size) in enumerate(models, 1):
        logger.info(f"  [{i}/{len(models)}] {name} (input: {input_size})")
        start = time.time()
        try:
            test_model(name, input_size, framework, classcode, dataset_cache)
            elapsed = time.time() - start
            logger.info(f"    PASS ({elapsed:.1f}s)")
            results["pass"].append(name)
        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e).split('\n')[0][:100]
            logger.info(f"    FAIL ({elapsed:.1f}s): {error_msg}")
            results["fail"].append((name, error_msg))

    return results


def run_all_tests():
    classcode = Local_Classification_ClassCodeBuilder().init_label_data(dataset_path=dataset).build()
    dataset_cache = {}

    logger.info(f"{'='*60}")
    logger.info(f"CNN Training Standard - Local Network Test (All)")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"PyTorch models: {len(PYTORCH_MODELS)}")
    logger.info(f"TensorFlow models: {len(TENSORFLOW_MODELS)}")
    logger.info(f"{'='*60}\n")

    # PyTorch 테스트
    logger.info(f"--- PyTorch ({len(PYTORCH_MODELS)} models) ---")
    pt_results = run_framework_tests("pytorch", PYTORCH_MODELS, classcode, dataset_cache)

    # TensorFlow 테스트
    logger.info(f"\n--- TensorFlow ({len(TENSORFLOW_MODELS)} models) ---")
    tf_results = run_framework_tests("tensorflow", TENSORFLOW_MODELS, classcode, dataset_cache)

    # 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS")
    logger.info(f"{'='*60}")

    logger.info(f"\n[PyTorch] PASS: {len(pt_results['pass'])}, FAIL: {len(pt_results['fail'])}")
    if pt_results["fail"]:
        for name, err in pt_results["fail"]:
            logger.info(f"  X {name}: {err}")

    logger.info(f"\n[TensorFlow] PASS: {len(tf_results['pass'])}, FAIL: {len(tf_results['fail'])}")
    if tf_results["fail"]:
        for name, err in tf_results["fail"]:
            logger.info(f"  X {name}: {err}")

    total_pass = len(pt_results['pass']) + len(tf_results['pass'])
    total_fail = len(pt_results['fail']) + len(tf_results['fail'])
    total = total_pass + total_fail
    logger.info(f"\nTotal: {total_pass}/{total} passed")


if __name__ == "__main__":
    run_all_tests()
