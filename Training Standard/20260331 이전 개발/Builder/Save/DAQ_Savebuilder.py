import os
import json
import matplotlib.pyplot as plt
import numpy as np
import io
import cv2
import itertools
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, Optional
import mpp

"""
Builder class for saving training and validation metrics, model checkpoints, and related files.
Usage Example:
    saver = Save_Builder()
        .init_inference_info(label_info = {...}, etc={'input_size': [224,224,3]})
        .init_save_url("s3://my-bucket/my-model/")
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
class DAQ_SaveBuilder:
    def __init__(self):
        self.__metrics = dict()
        self.__save_url = None
        self.__inference_info = None
        self.__saved_files = {} # { "directory": [{"name": "file1", "type": "csv"}, ...] }
        
        self.__operation_channel = None
        self.__access_token = None
        self.__chunk_size = None
    
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
    
    def init_save_url(self, bucket_url: str, operation_channel, access_token: str, chunk_size: int):
        """
        Sets the base URL where all files will be saved.
        Example:
            builder.set_save_url("s3://my-bucket/my-model/")
        """
        if not isinstance(bucket_url, str):
            raise TypeError("bucket_url must be a string")
        # if not isinstance(operation_channel, str):
        #     raise TypeError("operation_channel must be a string")
        if not isinstance(access_token, str):
            raise TypeError("access_token must be a string")
        if not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")

        self.__operation_channel = operation_channel
        self.__access_token = access_token
        self.__chunk_size = chunk_size
        self.__save_url = bucket_url
        
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
        save_uri = f"{self.__save_url}/{file_full_path}"
        mpp.intel64.save_csv(data, save_uri, channel=self.__operation_channel, access_token=self.__access_token, chunk_size=self.__chunk_size)
        self._register_saved_file(file_full_path, file_type='csv')
        return self
    
    def save_model(self, file_full_path: str="model.pth", model=None):
        """모델을 지정된 경로에 저장합니다.

        Example:
        builder.save_model("model.pth", model=my_model)
        """
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if model is None:
            raise ValueError("model must not be None")

        save_uri = f"{self.__save_url}/{file_full_path}"
        mpp.daq.object_service.upload_model(model, uri=save_uri, inference_info=self.__inference_info, channel=self.__operation_channel, access_token=self.__access_token, chunk_size=self.__chunk_size)
        self._register_saved_file(file_full_path, file_type='model')
        return self
    
    def save_file(self, file, file_full_path:str):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if file is None:
            raise ValueError("file must not be None")
        
        save_uri = f"{self.__save_url}/{file_full_path}"
        mpp.intel64.save(file, save_uri, channel=self.__operation_channel, access_token=self.__access_token, chunk_size=self.__chunk_size)
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
        
        save_uri = f"{self.__save_url}/{file_full_path}"
        mpp.intel64.save_csv(csv, save_uri, channel=self.__operation_channel, access_token=self.__access_token, chunk_size=self.__chunk_size)
        self._register_saved_file(file_full_path, file_type='csv')
        return self
    
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

    def save_file_index(self, index_path: str = "file_index.csv") -> None:
        """저장된 파일 목록을 CSV 파일로 저장합니다."""
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