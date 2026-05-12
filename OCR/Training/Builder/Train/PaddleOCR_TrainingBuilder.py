import logging
import os
import shutil
import subprocess
import sys


class PaddleOCR_TrainingBuilder:
    """
    PaddleOCR rec 학습 빌더.
    YAML 생성 → paddle.distributed.launch 호출.
    TrainHook.on_epoch_end 는 호출되지 않음 (PaddleOCR 내부 루프).
    training_start / training_end 는 정상 호출됨.
    """

    def __init__(self, logger=None):
        self.__logger    = logger or logging.getLogger()
        self.__hp        = None
        self.__base_dir  = None

    def initialize(self, hp_builder, base_dir: str) -> 'PaddleOCR_TrainingBuilder':
        self.__hp       = hp_builder
        self.__base_dir = base_dir
        return self

    def builder(self) -> 'PaddleOCR_TrainingBuilder':
        return self

    def train(self, dataset_builder, hook=None):
        from ..Config.PaddleOCR_ConfigBuilder import save_yaml_config

        config   = self.__build_config(dataset_builder)
        temp_dir = os.path.join(self.__base_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        yml_path = os.path.join(temp_dir, "train_config.yml")

        try:
            if hook:
                hook.training_start()

            save_yaml_config(config, self.__base_dir, yml_path)
            self.__run(yml_path)

            if hook:
                hook.training_end()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ── 내부 ─────────────────────────────────────────────────────────────────

    def __build_config(self, dataset_builder) -> dict:
        hp = self.__hp

        def rel(abs_path):
            return os.path.relpath(abs_path, self.__base_dir)

        return {
            'pretrained_model':    hp.get_pretrained_model(),
            'resume_path':         hp.get_resume_path(),
            'rec_char_dict_path':  rel(dataset_builder.get_char_dict_path()),
            'save_dir':            hp.get_save_dir(),
            'train_data_dir':      rel(dataset_builder.get_train_data_dir()),
            'train_label_file':    rel(dataset_builder.get_train_label_file()),
            'val_data_dir':        rel(dataset_builder.get_val_data_dir()),
            'val_label_file':      rel(dataset_builder.get_val_label_file()),
            'image_shape':         hp.get_image_shape(),
            'max_text_length':     hp.get_max_text_length(),
            'epoch_num':           hp.get_epoch_num(),
            'save_epoch_step':     hp.get_save_epoch_step(),
            'batch_size':          hp.get_batch_size(),
            'learning_rate':       hp.get_learning_rate(),
            'use_space_char':      hp.get_use_space_char(),
            'use_guided_training': hp.get_use_guided_training(),
            'device':              hp.get_device(),
            'gpu_id':              hp.get_gpu_id(),
        }

    def __run(self, yml_path: str):
        tools_train = self.__find_tools_train()

        if self.__hp.get_device() == 'gpu':
            cmd = [
                sys.executable, "-m", "paddle.distributed.launch",
                "--gpus", self.__hp.get_gpu_id(),
                tools_train, "-c", yml_path,
            ]
        else:
            cmd = [sys.executable, tools_train, "-c", yml_path]

        self.__logger.info(f"CMD: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=self.__base_dir)

    def __find_tools_train(self) -> str:
        # 1. hyperparameter paddleocr_home
        home = self.__hp.get_paddleocr_home()
        # 2. env var fallback
        if not home:
            home = os.environ.get('PADDLEOCR_HOME', '')
        # 3. auto-detect: ./PaddleOCR or ../PaddleOCR relative to base_dir
        if not home:
            for rel in ('PaddleOCR', os.path.join('..', 'PaddleOCR')):
                candidate = os.path.normpath(os.path.join(self.__base_dir, rel))
                if os.path.isdir(candidate):
                    home = candidate
                    break

        if not home:
            raise FileNotFoundError(
                "PaddleOCR source not found.\n"
                "Clone: git clone https://github.com/PaddlePaddle/PaddleOCR.git -b release/2.7\n"
                "Then set 'paddleocr_home' in hyperparameters or PADDLEOCR_HOME env var."
            )

        tools_train = os.path.join(home, 'tools', 'train.py')
        if not os.path.exists(tools_train):
            raise FileNotFoundError(f"tools/train.py not found in: {home}")
        return tools_train
