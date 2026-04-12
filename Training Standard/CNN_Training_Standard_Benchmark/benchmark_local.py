"""
CNN Training Standard 벤치마크 (로컬)
benchmark_config.json의 실험 설정을 순차 실행하고 결과를 비교합니다.

Usage:
    python benchmark_local.py
    python benchmark_local.py --config my_config.json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CNN_Training_Standard'))

import json
import time
import argparse
import logging
import csv

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

from Builder import Json_HyperparameterBuilder, Local_Classification_ClassCodeBuilder
from Builder import get_framework_builders
from Builder import TrainHook


class BenchmarkHook(TrainHook):
    """벤치마크용 Hook - epoch별 loss 기록"""
    def __init__(self):
        self.history = []

    def training_start(self): pass
    def training_end(self): pass

    def on_epoch_end(self, total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time, model):
        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_acc": 100 - train_loss,
            "val_acc": (100 - validation_loss) if validation_loss is not None else None,
            "time": epoch_elapsed_time,
        })


def run_experiment(exp_params, dataset_path, framework):
    """단일 실험 실행. BenchmarkHook.history 반환."""
    input_size = exp_params['input_size']
    network_name = exp_params['network_name']

    classcode = Local_Classification_ClassCodeBuilder().init_label_data(dataset_path=dataset_path).build()
    Model, LocalDatasetBuilder, TrainingBuilder, _ = get_framework_builders(framework)

    dataset_builder = LocalDatasetBuilder(logger=None).init_dataset_path(dataset_path=dataset_path
        ).init_transform(
            input_size=input_size,
            normalize_mean=exp_params.get('normalize_mean', 0.5),
            normalize_stdev=exp_params.get('normalize_stdev', 0.5)
        ).create_train_dataset(
            train_ratio=exp_params.get('train_ratio', 0.8),
            batch_size=exp_params['batch_size'],
            validation_save_random=False,
            class_code_info=classcode.get_class_code_info()
        )

    model_builder = Model().init_device(exp_params.get('device', 'cpu'))
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

    training_builder = TrainingBuilder(None
        ).initialize(
            epoch_total=exp_params['epoch'],
            device=exp_params.get('device', 'cpu'),
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


def save_results_csv(all_results, output_path):
    """전체 결과를 CSV로 저장"""
    filepath = os.path.join(output_path, "benchmark_results.csv")
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "epoch", "train_loss", "val_loss", "train_acc", "val_acc", "time_sec"])
        for exp_name, history in all_results.items():
            for h in history:
                writer.writerow([
                    exp_name, h['epoch'],
                    f"{h['train_loss']:.6f}",
                    f"{h['validation_loss']:.6f}" if h['validation_loss'] is not None else "",
                    f"{h['train_acc']:.2f}",
                    f"{h['val_acc']:.2f}" if h['val_acc'] is not None else "",
                    f"{h['time']:.2f}",
                ])
    logger.info(f"Results saved: {filepath}")


def save_summary(all_results, output_path):
    """최종 요약 비교 테이블 저장"""
    filepath = os.path.join(output_path, "benchmark_summary.csv")
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "network", "optimizer", "lr", "epochs",
                         "final_train_loss", "final_val_loss", "best_val_loss",
                         "final_train_acc", "final_val_acc", "best_val_acc",
                         "total_time_sec"])
        for exp_name, history in all_results.items():
            if not history:
                continue
            last = history[-1]
            val_losses = [h['validation_loss'] for h in history if h['validation_loss'] is not None]
            val_accs = [h['val_acc'] for h in history if h['val_acc'] is not None]
            total_time = sum(h['time'] for h in history)

            parts = exp_name.split('_')
            writer.writerow([
                exp_name,
                history[0].get('network', exp_name),
                "", "",
                len(history),
                f"{last['train_loss']:.6f}",
                f"{last['validation_loss']:.6f}" if last['validation_loss'] is not None else "",
                f"{min(val_losses):.6f}" if val_losses else "",
                f"{last['train_acc']:.2f}",
                f"{last['val_acc']:.2f}" if last['val_acc'] is not None else "",
                f"{max(val_accs):.2f}" if val_accs else "",
                f"{total_time:.1f}",
            ])
    logger.info(f"Summary saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="CNN Training Standard Benchmark")
    parser.add_argument("--config", "-c", type=str, default="benchmark_config.json", help="벤치마크 설정 파일")
    parser.add_argument("--dataset", "-d", type=str, default=None, help="데이터셋 경로")
    parser.add_argument("--output", "-o", type=str, default=None, help="결과 저장 경로")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    dataset_path = args.dataset or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'create_dataset', 'dataset')
    output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_path, exist_ok=True)

    common = config.get('common', {})
    experiments = config.get('experiments', [])
    framework = common.get('framework', 'pytorch')

    logger.info(f"{'='*60}")
    logger.info(f"CNN Training Standard - Benchmark")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Framework: {framework}")
    logger.info(f"Experiments: {len(experiments)}")
    logger.info(f"Output: {output_path}")
    logger.info(f"{'='*60}\n")

    all_results = {}
    passed = 0
    failed = 0

    for i, exp in enumerate(experiments, 1):
        exp_name = exp.get('name', f"exp_{i}")
        # common과 experiment 파라미터 병합
        params = {**common, **exp}
        params['device'] = 'cuda' if params.get('using_gpu', False) else 'cpu'

        logger.info(f"[{i}/{len(experiments)}] {exp_name}")
        logger.info(f"  network: {params['network_name']}, optimizer: {params['optimizer_name']}, "
                     f"lr: {params['lr']}, batch: {params['batch_size']}, epoch: {params['epoch']}")

        start = time.time()
        try:
            history = run_experiment(params, dataset_path, framework)
            elapsed = time.time() - start
            last = history[-1]
            logger.info(f"  final_loss: train={last['train_loss']:.4f}, val={last['validation_loss']:.4f if last['validation_loss'] else 'N/A'}")
            logger.info(f"  PASS ({elapsed:.1f}s)\n")
            all_results[exp_name] = history
            passed += 1
        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e).split('\n')[0][:100]
            logger.info(f"  FAIL ({elapsed:.1f}s): {error_msg}\n")
            all_results[exp_name] = []
            failed += 1

    # 결과 저장
    save_results_csv(all_results, output_path)
    save_summary(all_results, output_path)

    # 최종 비교 출력
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

    # best_val_loss 기준 정렬
    ranked.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))
    for rank, (name, loss, acc, t) in enumerate(ranked, 1):
        marker = " ★" if rank == 1 else ""
        logger.info(f"{name:<35} {loss:>10.4f} {acc:>9.2f}% {t:>7.1f}s{marker}")

    logger.info(f"\n{'='*60}")


if __name__ == "__main__":
    main()
