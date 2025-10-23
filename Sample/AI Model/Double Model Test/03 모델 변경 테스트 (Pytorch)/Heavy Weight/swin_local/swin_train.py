import json
import logging
from builders import (
    HyperparameterBuilder,
    DummyDatasetBuilder,
    TorchModelBuilder,
    LocalSaveBuilder
)

##!--{"Name":"hyperparameter","Type":"epoch","Key":"epoch","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"batch_size","Key":"batch_size","Value":"","Category":""}
##!--{"Name":"save","Type":"save_dir","Key":"save_dir","Value":"","Category":""}

##$--
parameters = '''{
    "hyperparameter": {
        "epoch": 3,
        "save_epoch": 1,
        "batch_size": 4,
        "lr": 5e-5,
        "optimizer_name": "AdamW",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_stdev": [0.229, 0.224, 0.225],
        "using_gpu": true,
        "using_amp": true,
        "train_ratio": 0.8,
        "validation_save_random": false,
        "num_samples": 1000,
        "num_classes": 10
    },
    "save": {
        "save_dir": "saved_models"
    }
}'''
##$--

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # 기본 설정으로 학습 실행
    kwargs = json.loads(parameters)
    try:
        # 1. 하이퍼파라미터 설정
        logger.info("Initializing hyperparameters...")
        hyperparameter_builder = HyperparameterBuilder(kwargs)
        hyperparameter_builder.initialize().build()

        # 2. 데이터셋 생성
        logger.info("Creating dataset...")
        dataset_builder = DummyDatasetBuilder(hyperparameter_builder)
        dataset_builder.initialize().build()

        # 3. 모델 생성 및 학습
        logger.info("Setting up Swin-Large model...")
        model_builder = TorchModelBuilder(hyperparameter_builder, logger)
        model = model_builder.initialize().build()

        # 4. 저장 설정
        logger.info("Setting up model saver...")
        save_builder = LocalSaveBuilder(hyperparameter_builder)
        save_builder.initialize().build()

        # 5. 학습 실행
        logger.info("Starting Swin-Large training...")
        model_builder.train(
            dataset_builder.get_train_loader(),
            dataset_builder.get_validation_loader()
        )

        logger.info("Swin-Large training completed successfully!")

    except Exception as e:
        logger.error(f"Swin-Large training failed: {e}")
        raise
