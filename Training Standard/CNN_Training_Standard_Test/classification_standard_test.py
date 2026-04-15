"""
CNN Training Standard 전체 네트워크 DAQ 테스트
network_list.json에 등록된 모든 네트워크에 대해 학습 테스트.
Dataset은 input_size별로 한 번만 생성하여 공유합니다.
네트워크마다 결과 CSV를 업데이트합니다.
완료 후 실패 목록을 network_list_fail.json으로 저장합니다.
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CNN_Training_Standard'))

import json
import time
import logging

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

from Builder import get_framework_builders, get_daq_framework_builders
from Builder import TrainHook

# network_list.json에서 모델 목록 로드
_list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'network_list.json')
with open(_list_path, 'r', encoding='utf-8') as f:
    _network_list = json.load(f)

PYTORCH_MODELS = [(m['name'], m['input_size']) for m in _network_list['pytorch']]
TENSORFLOW_MODELS = [(m['name'], m['input_size']) for m in _network_list['tensorflow']]


class MetricsSaveHook(TrainHook):
    def __init__(self):
        self.history = []
    def training_start(self): pass
    def training_end(self): pass
    def on_epoch_end(self, total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time, model):
        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "time": epoch_elapsed_time,
        })
        val_str = f"{validation_loss:.4f}" if validation_loss is not None else "N/A"
        logger.info(f"    epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_str}, time={epoch_elapsed_time:.1f}s")


def create_dataset_cached(framework, input_size, operation_builder, classcode_builder, cache):
    cache_key = f"{framework}_{input_size}"
    if cache_key in cache:
        return cache[cache_key]

    logger.info(f"    Creating dataset (input_size={input_size})...")
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

    cache[cache_key] = dataset_builder
    return dataset_builder


def test_single_model(network_name, input_size, operation_builder, classcode_builder, framework, dataset_cache, epoch=1):
    device = 'cuda' if framework == 'pytorch' else '/gpu:0'
    Model, _, TrainingBuilder, _ = get_framework_builders(framework)

    dataset_builder = create_dataset_cached(framework, input_size, operation_builder, classcode_builder, dataset_cache)

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

    training_builder = TrainingBuilder(logger
        ).initialize(epoch_total=epoch, device=device, using_amp=False
        ).init_model(model=model
        ).init_optimizer(optimizer_name='Adam', lr=1e-3
        ).init_criterion(criterion_name='CrossEntropyLoss'
        ).builder()

    hook = MetricsSaveHook()
    training_builder.train(
        train_data_loader=dataset_builder.get_train_data_loader(),
        validation_data_loader=dataset_builder.get_validation_data_loader(),
        hook=hook
    )
    return hook.history


def save_csv_incremental(all_history, operation_builder):
    """현재까지의 결과를 CSV로 저장 (네트워크마다 호출)"""
    import mpp

    summary_csv = [["network", "status", "error", "final_train_loss", "final_val_loss", "best_val_loss", "total_time_sec"]]
    for network_name, data in all_history.items():
        if data["status"] == "FAIL":
            summary_csv.append([network_name, "FAIL", data.get("error", ""), "", "", "", f"{data['time']:.1f}"])
        else:
            history = data["history"]
            last = history[-1]
            val_losses = [h['validation_loss'] for h in history if h['validation_loss'] is not None]
            summary_csv.append([
                network_name, "PASS", "",
                f"{last['train_loss']:.6f}",
                f"{last['validation_loss']:.6f}" if last['validation_loss'] is not None else "",
                f"{min(val_losses):.6f}" if val_losses else "",
                f"{data['time']:.1f}"
            ])

    bucket_url = operation_builder.get_bucket_url()
    channel = operation_builder.get_operation_channel()
    access_token = operation_builder.get_access_token()
    chunk_size = operation_builder.get_chunk_size()

    mpp.intel64.save_csv(summary_csv, f"{bucket_url}/test_results.csv", channel=channel, access_token=access_token, chunk_size=chunk_size)


def save_fail_list(all_history):
    """실패 목록을 network_list_fail.json으로 저장"""
    fail_list = {"pytorch": [], "tensorflow": []}

    for network_name, data in all_history.items():
        if data["status"] != "FAIL":
            continue
        fw = data.get("framework", "pytorch")
        input_size = data.get("input_size", 224)
        error = data.get("error", "")
        fail_list[fw].append({"name": network_name, "input_size": input_size, "error": error})

    fail_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'network_list_fail.json')
    with open(fail_path, 'w', encoding='utf-8') as f:
        json.dump(fail_list, f, indent=2, ensure_ascii=False)
    logger.info(f"Fail list saved: network_list_fail.json ({len(fail_list['pytorch'])} pytorch, {len(fail_list['tensorflow'])} tensorflow)")


def run_framework_tests(framework, models, operation_builder, classcode_builder, dataset_cache, all_history, epoch=1):
    results = {"pass": [], "fail": []}

    for i, (name, input_size) in enumerate(models, 1):
        logger.info(f"  [{i}/{len(models)}] {name} (input: {input_size}, epoch: {epoch})")
        start = time.time()
        try:
            history = test_single_model(name, input_size, operation_builder, classcode_builder, framework, dataset_cache, epoch)
            elapsed = time.time() - start
            logger.info(f"    PASS ({elapsed:.1f}s)")
            results["pass"].append(name)
            all_history[name] = {"status": "PASS", "framework": framework, "input_size": input_size, "time": elapsed, "history": history}
        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e).split('\n')[0][:100]
            logger.info(f"    FAIL ({elapsed:.1f}s): {error_msg}")
            results["fail"].append((name, error_msg))
            all_history[name] = {"status": "FAIL", "framework": framework, "input_size": input_size, "time": elapsed, "error": error_msg, "history": []}

        # 네트워크마다 CSV 업데이트
        try:
            save_csv_incremental(all_history, operation_builder)
        except Exception:
            pass

    return results


def RecipeRun(**kwargs):
    from Builder import Operation_Builder
    from Builder import DAQ_Classification_ClassCodeBuilder

    operation_builder = Operation_Builder(**kwargs).initialize().build()
    classcode_builder = DAQ_Classification_ClassCodeBuilder().init_url_info(
        operation_channel=operation_builder.get_operation_channel(),
        access_token=operation_builder.get_access_token()
    ).init_label_data(
        gt_dataset_id=operation_builder.get_gt_dataset_id()
    ).build()

    epoch = int(kwargs['hyperparameter'].get('epoch', 1))

    logger.info(f"{'='*60}")
    logger.info(f"CNN Training Standard - DAQ Network Test (All)")
    logger.info(f"Epoch: {epoch}")
    logger.info(f"PyTorch models: {len(PYTORCH_MODELS)}")
    logger.info(f"TensorFlow models: {len(TENSORFLOW_MODELS)}")
    logger.info(f"{'='*60}\n")

    dataset_cache = {}
    all_history = {}

    # PyTorch 테스트
    logger.info(f"--- PyTorch ({len(PYTORCH_MODELS)} models) ---")
    pt_results = run_framework_tests("pytorch", PYTORCH_MODELS, operation_builder, classcode_builder, dataset_cache, all_history, epoch)

    # TensorFlow 테스트
    logger.info(f"\n--- TensorFlow ({len(TENSORFLOW_MODELS)} models) ---")
    tf_results = run_framework_tests("tensorflow", TENSORFLOW_MODELS, operation_builder, classcode_builder, dataset_cache, all_history, epoch)

    # temp 폴더 정리
    for ds in dataset_cache.values():
        try:
            ds.temp_folder_delete()
        except Exception:
            pass

    # 실패 목록 저장
    save_fail_list(all_history)

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
    kwargs = json.loads(parameters)
    kwargs['authentication']['operation_service_address'] = ""
    kwargs['authentication']['access_token'] = ""
    kwargs['gt_dataset']['gt_dataset_id'] = r""
    import random
    kwargs['result']['id'] = f"network_test_{random.randint(1000,9999)}"

    RecipeRun(**kwargs)
