import os
import json
import matplotlib.pyplot as plt
import numpy as np
import io
import cv2
import itertools
import torch
import pickle
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, Optional

"""
Builder class for saving training and validation metrics, model checkpoints, and related files to local filesystem.
Usage Example:
    saver = Local_SaveBuilder()
        .init_inference_info(label_info = {...}, etc={'input_size': [224,224,3]})
        .init_save_url("./output/my-model/")
    saver.append_metrics({
             'Train Loss': 0.0008,
             'Train Acc': 99.5,
             'epoch': 2,
             'best_model': False
         })
         .save_epoch_file(epoch=2, file=my_file, file_full_path="file.txt")
         .save_epoch_model(epoch=2, model=my_model, file_full_path="model.pth")
    saver.save_summary_and_close(metric_full_path="metrics.csv")
"""
class Local_SaveBuilder:
    def __init__(self):
        self.__metrics = dict()
        self.__save_url = None
        self.__inference_info = None
        self.__saved_files = {} # { "directory": [{"name": "file1", "type": "csv"}, ...] }

    def init_inference_info(self, label_info: Optional[Dict[str, Any]] = None, etc: Optional[Dict[str, Any]] = None):
        """
        Accepts `label_info` and `etc` as dictionaries and stores them as JSON strings under the
        keys 'label_info' and 'etc'.

        Example:
            builder.init_inference_info(label_info={...}, etc={'input_size': [224,224,3], ...})
        """
        # allow None -> treat as empty dict so fields remain blank in saved metadata
        if label_info is None and etc is None:
            return self
        if label_info is None:
            label_info = {}
        if etc is None:
            etc = {}

        if not isinstance(label_info, dict):
            raise TypeError("label_info must be a dict or None")
        if not isinstance(etc, dict):
            raise TypeError("etc must be a dict or None")

        # store both parts as JSON strings; keep keys consistent
        self.__inference_info = {
            'label_info': json.dumps(label_info),
            'inference_info': json.dumps(etc)
        }
        return self

    def init_save_url(self, save_path: str):
        """
        Sets the base path where all files will be saved locally.
        Example:
            builder.init_save_url("./output/my-model/")
        """
        if not isinstance(save_path, str):
            raise TypeError("save_path must be a string")

        self.__save_url = save_path

        # Create base directory if it doesn't exist
        os.makedirs(self.__save_url, exist_ok=True)

        return self

    def append_metrics(self, metrics: Dict[str, Any]):
        """메트릭 데이터를 추가합니다.

        Example:
        builder.append_metrics({
            'Train Loss': 0.0008,
            'Train Acc': 99.5,
            'epoch': 2,
            'best_model': False
        })

        사용자가 제공한 메트릭을 그대로 저장합니다. 각 메트릭 이름에 대해 값들의 리스트가 유지됩니다.
        """
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict")

        # 각 메트릭을 저장된 리스트에 추가
        for name, value in metrics.items():
            if name not in self.__metrics:
                self.__metrics[name] = []
            self.__metrics[name].append(value)

        return self
    
    def save_csv_metric(self, file_full_path: str="metrics.csv"):
        """현재까지 저장된 메트릭을 CSV 파일로 저장합니다.

        Example:
        builder.save_metric("metrics.csv")
        """
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")

        data = self._create_metrics_csv()
        save_path = os.path.join(self.__save_url, file_full_path)
        self._save_csv_to_local(data, save_path)
        self._register_saved_file(file_full_path, file_type='csv')
        return self

    def save_model(self, file_full_path:str, model):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if model is None:
            raise ValueError("model must not be None")

        save_path = os.path.join(self.__save_url, file_full_path)
        self._save_model_to_local(model, save_path)
        self._register_saved_file(file_full_path, file_type='model')
        return self

    def save_file(self, file, file_full_path:str):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if file is None:
            raise ValueError("file must not be None")

        save_path = os.path.join(self.__save_url, file_full_path)
        self._save_to_local(file, save_path)

        file_type = 'file'
        if file_full_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', 'tiff', 'tif')):
            file_type = 'image'
        self._register_saved_file(file_full_path, file_type=file_type)
        return self

    def save_csv(self, csv, file_full_path:str):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if csv is None:
            raise ValueError("csv must not be None")

        save_path = os.path.join(self.__save_url, file_full_path)
        self._save_csv_to_local(csv, save_path)
        self._register_saved_file(file_full_path, file_type='csv')
        return self

    def _save_to_local(self, data, save_path: str):
        """로컬 파일 시스템에 데이터를 저장합니다."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if isinstance(data, str):
            # 문자열 데이터는 텍스트 파일로 저장
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(data)
        elif isinstance(data, bytes):
            # 바이너리 데이터
            with open(save_path, 'wb') as f:
                f.write(data)
        elif isinstance(data, np.ndarray):
            # 이미지 데이터인 경우
            if save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                cv2.imwrite(save_path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
            else:
                np.save(save_path, data)
        else:
            # 기타 객체는 pickle로 저장
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

    def _save_model_to_local(self, model, save_path: str):
        """모델을 로컬 파일 시스템에 저장합니다."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        extension = os.path.splitext(save_path)[1].lower()

        if hasattr(model, 'state_dict'):
            # PyTorch
            self._save_pytorch_model(model, save_path, extension)
        elif hasattr(model, 'trainable_variables'):
            # TensorFlow/Keras
            self._save_tensorflow_model(model, save_path, extension)
        else:
            self._save_model_with_pickle(model, save_path)

    def _save_tensorflow_model(self, model, save_path: str, extension: str):
        """TensorFlow 모델을 저장합니다."""
        import tensorflow as tf
        if self.__inference_info is not None:
            inference_info_json = json.dumps(self.__inference_info, ensure_ascii=False)
            model.inference_info = tf.Variable(inference_info_json, trainable=False, dtype=tf.string, name="inference_info")

        if extension == '.h5':
            model.save(save_path, save_format='h5')
        elif extension == '.keras':
            model.save(save_path, save_format='keras')
        else:
            saved_model_path = save_path.replace('.pth', '')
            model.save(saved_model_path, save_format='tf')

        self._save_inference_info_as_json(save_path)

    def _save_pytorch_model(self, model, save_path: str, extension: str):
        """PyTorch 모델을 저장합니다."""
        if extension == '.pth':
            self._save_pytorch_model_as_torchscript(model, save_path)
        else:
            self._save_pytorch_model_as_state_dict(model, save_path)

    def _save_pytorch_model_as_torchscript(self, model, save_path: str):
        """PyTorch 모델을 TorchScript 형식으로 저장합니다. (script → trace → state_dict fallback)"""
        origin_mode = model.training
        if origin_mode:
            model.eval()

        model_script = None

        # 1) torch.jit.script 시도
        try:
            model_script = torch.jit.script(model)
        except Exception:
            pass

        # 2) script 실패 시 torch.jit.trace 시도
        if model_script is None:
            try:
                device = next(model.parameters()).device
                # inference_info에서 input_size 추출
                input_size = self._get_input_size_from_inference_info()
                dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
                model_script = torch.jit.trace(model, dummy_input)
            except Exception:
                pass

        # 3) 둘 다 실패 시 state_dict fallback
        if model_script is None:
            self._save_pytorch_model_as_state_dict(model, save_path)
            if origin_mode:
                model.train()
            return

        if origin_mode:
            model.train()

        extra_files = self._prepare_extra_files()
        model_buffer = model_script.save_to_buffer(_extra_files=extra_files)
        with open(save_path, 'wb') as f:
            f.write(model_buffer)

    def _save_pytorch_model_as_state_dict(self, model, save_path: str):
        """PyTorch 모델을 state_dict 형식으로 저장합니다."""
        torch.save(model.state_dict(), save_path)
        self._save_inference_info_as_json(save_path)

    def _save_model_with_pickle(self, model, save_path: str):
        """기타 모델을 pickle로 저장합니다."""
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

    def _get_input_size_from_inference_info(self) -> int:
        """inference_info에서 input_size를 추출합니다. 실패 시 기본값 224."""
        try:
            if self.__inference_info is not None:
                label_info = self.__inference_info.get('label_info', '{}')
                if isinstance(label_info, str):
                    label_info = json.loads(label_info)
                inner = label_info.get('inference_info', '{}')
                if isinstance(inner, str):
                    inner = json.loads(inner)
                return int(inner.get('input_size', 224))
        except Exception:
            pass
        return 224

    def _prepare_extra_files(self) -> dict:
        """inference_info를 TorchScript extra_files 형식으로 준비합니다."""
        extra_files = {}
        if self.__inference_info is not None:
            for key, value in self.__inference_info.items():
                extra_files[key] = value
        return extra_files

    def _save_inference_info_as_json(self, model_path: str):
        """inference_info를 별도 JSON 파일로 저장합니다."""
        if self.__inference_info is not None:
            info_path = model_path.rsplit('.', 1)[0] + '_info.json'
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(self.__inference_info, f, indent=2, ensure_ascii=False)

    def _save_csv_to_local(self, csv_data, save_path: str):
        """CSV 데이터를 로컬 파일 시스템에 저장합니다."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', encoding='utf-8', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(csv_data)

    def _register_saved_file(self, path: str, file_type: str = 'file') -> None:
        """저장된 파일을 내부 인덱스에 등록합니다.

        Args:
            path (str): 저장된 파일의 경로
            file_type (str): 파일 유형 ('file', 'csv', 'model', 'image' 등)
        """
        # 경로를 디렉토리와 파일이름으로 분리
        directory = os.path.dirname(path)
        filename = os.path.basename(path)

        # 디렉토리가 없으면 새 리스트 생성
        if directory not in self.__saved_files:
            self.__saved_files[directory] = []

        # 이미 존재하는 파일이면 업데이트, 없으면 추가
        file_info = {"name": filename, "type": file_type}

        # 같은 이름의 파일이 있는지 확인
        existing_file = next((f for f in self.__saved_files[directory]
                            if f["name"] == filename), None)

        if existing_file:
            existing_file.update(file_info)  # 기존 정보 업데이트
        else:
            self.__saved_files[directory].append(file_info)  # 새로운 파일 정보 추가

    def _create_metrics_csv(self):
        """Build CSV table from `self.__metrics`.

        This implementation is independent of 'Epoch' — the CSV header is the metric
        names and each row is one record (index-aligned across metrics). Missing
        values are written as empty strings.

        Returns the CSV as a list of rows (header + data rows).
        """

        metric_names = list(self.__metrics.keys())
        if not metric_names:
            return [[]]

        max_length = max((len(lst) for lst in self.__metrics.values()), default=0)

        header = metric_names
        result = [header]

        for idx in range(max_length):
            row = []
            for name in metric_names:
                lst = self.__metrics.get(name, [])
                value = lst[idx] if idx < len(lst) else None
                # Write empty string for missing values so CSV cells are blank
                row.append('' if value is None else value)
            result.append(row)

        return result

    def save_file_index(self, index_path: str = "file_index.csv") -> None:
        """저장된 파일 목록을 CSV 파일로 저장합니다."""
        # CSV 형태로 변환: [header, row1, row2, ...]
        # Header: directory, filename, type
        data = [["directory", "filename", "type"]]

        # self.__saved_files 구조: { "directory": [{"name": "file1", "type": "csv"}, ...] }
        for directory, files in self.__saved_files.items():
            for file_info in files:
                row = [
                    directory if directory else "",  # 빈 디렉토리는 빈 문자열로
                    file_info.get("name", ""),
                    file_info.get("type", "")
                ]
                data.append(row)

        self.save_csv(data, index_path)

    def build(self):
        return self
