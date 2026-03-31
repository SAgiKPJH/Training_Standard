import os
import json
import numpy as np
import cv2
import pickle
import tensorflow as tf
from typing import Dict, Any, Optional

class Local_SaveBuilder:
    def __init__(self):
        self.__metrics = dict()
        self.__save_url = None
        self.__inference_info = None
        self.__saved_files = {}

    def init_inference_info(self, label_info: Optional[Dict[str, Any]] = None, etc: Optional[Dict[str, Any]] = None):
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

        self.__inference_info = {
            'label_info': json.dumps(label_info),
            'inference_info': json.dumps(etc)
        }
        return self

    def init_save_url(self, save_path: str):
        if not isinstance(save_path, str):
            raise TypeError("save_path must be a string")
        self.__save_url = save_path
        os.makedirs(self.__save_url, exist_ok=True)
        return self

    def append_metrics(self, metrics: Dict[str, Any]):
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict")
        for name, value in metrics.items():
            if name not in self.__metrics:
                self.__metrics[name] = []
            self.__metrics[name].append(value)
        return self

    def save_csv_metric(self, file_full_path: str = "metrics.csv"):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        data = self._create_metrics_csv()
        save_path = os.path.join(self.__save_url, file_full_path)
        self._save_csv_to_local(data, save_path)
        self._register_saved_file(file_full_path, file_type='csv')
        return self

    def save_model(self, file_full_path: str, model):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if model is None:
            raise ValueError("model must not be None")

        save_path = os.path.join(self.__save_url, file_full_path)
        self._save_model_to_local(model, save_path)
        self._register_saved_file(file_full_path, file_type='model')
        return self

    def save_file(self, file, file_full_path: str):
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

    def save_csv(self, csv, file_full_path: str):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if csv is None:
            raise ValueError("csv must not be None")

        save_path = os.path.join(self.__save_url, file_full_path)
        self._save_csv_to_local(csv, save_path)
        self._register_saved_file(file_full_path, file_type='csv')
        return self

    def _save_to_local(self, data, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if isinstance(data, str):
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(data)
        elif isinstance(data, bytes):
            with open(save_path, 'wb') as f:
                f.write(data)
        elif isinstance(data, np.ndarray):
            if save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                cv2.imwrite(save_path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
            else:
                np.save(save_path, data)
        else:
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

    def _save_model_to_local(self, model, save_path: str):
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else save_path, exist_ok=True)
        extension = os.path.splitext(save_path)[1].lower()

        if isinstance(model, tf.keras.Model):
            self._save_tensorflow_model(model, save_path, extension)
        else:
            self._save_model_with_pickle(model, save_path)

    def _save_tensorflow_model(self, model, save_path: str, extension: str):
        if extension == '.h5':
            model.save(save_path, save_format='h5')
        elif extension == '.keras':
            model.save(save_path, save_format='keras')
        else:
            # SavedModel format (directory)
            saved_model_path = save_path.replace('.pth', '')
            model.save(saved_model_path, save_format='tf')

        self._save_inference_info_as_json(save_path)

    def _save_model_with_pickle(self, model, save_path: str):
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

    def _save_inference_info_as_json(self, model_path: str):
        if self.__inference_info is not None:
            info_path = model_path.rsplit('.', 1)[0] + '_info.json'
            os.makedirs(os.path.dirname(info_path) if os.path.dirname(info_path) else '.', exist_ok=True)
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(self.__inference_info, f, indent=2, ensure_ascii=False)

    def _save_csv_to_local(self, csv_data, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(csv_data)

    def _register_saved_file(self, path: str, file_type: str = 'file') -> None:
        directory = os.path.dirname(path)
        filename = os.path.basename(path)

        if directory not in self.__saved_files:
            self.__saved_files[directory] = []

        file_info = {"name": filename, "type": file_type}
        existing_file = next((f for f in self.__saved_files[directory]
                            if f["name"] == filename), None)

        if existing_file:
            existing_file.update(file_info)
        else:
            self.__saved_files[directory].append(file_info)

    def _create_metrics_csv(self):
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
                row.append('' if value is None else value)
            result.append(row)

        return result

    def save_file_index(self, index_path: str = "file_index.csv") -> None:
        data = [["directory", "filename", "type"]]
        for directory, files in self.__saved_files.items():
            for file_info in files:
                row = [
                    directory if directory else "",
                    file_info.get("name", ""),
                    file_info.get("type", "")
                ]
                data.append(row)
        self.save_csv(data, index_path)

    def build(self):
        return self
