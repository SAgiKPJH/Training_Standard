import os
import json
from typing import Dict, Any, Optional
import mpp

"""
DAQ OLD 방식의 SaveBuilder.
기존 train_recipe.py의 저장 경로 패턴을 따릅니다:
  - 모델: {result_uri}/epoch_{N}/model/model.h5
  - CSV:  {result_uri}/epoch_{N}/train_loss_csv/train_loss_csv.csv
  - H5 포맷 + h5py로 inference_info를 extra_info 그룹에 저장
  - mpp.daq.object_service.upload_object 사용
"""
class DAQ_SaveBuilder_DAQ_OLD:
    def __init__(self):
        self.__metrics = dict()
        self.__save_url = None
        self.__inference_info = None
        self.__saved_files = {}

        self.__operation_channel = None
        self.__access_token = None
        self.__chunk_size = None

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

    def init_save_url(self, bucket_url: str, operation_channel, access_token: str, chunk_size: int):
        if not isinstance(bucket_url, str):
            raise TypeError("bucket_url must be a string")
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
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict")
        for name, value in metrics.items():
            if name not in self.__metrics:
                self.__metrics[name] = []
            self.__metrics[name].append(value)
        return self

    def save_csv_metric(self, epoch: int = None, file_full_path: str = None):
        """OLD 방식: epoch_{N}/train_loss_csv/train_loss_csv.csv 경로에 저장"""
        data = self._create_metrics_csv()

        if file_full_path is None:
            if epoch is not None:
                file_full_path = f"epoch_{epoch}/train_loss_csv/train_loss_csv.csv"
            else:
                file_full_path = "metrics.csv"

        save_uri = f"{self.__save_url}/{file_full_path}"
        mpp.intel64.save_csv(data, save_uri, channel=self.__operation_channel, access_token=self.__access_token, chunk_size=self.__chunk_size)
        self._register_saved_file(file_full_path, file_type='csv')
        return self

    def save_model(self, file_full_path: str = None, model=None, epoch: int = None):
        """OLD 방식 경로 패턴 + mpp.daq.object_service.upload_model로 저장"""
        if model is None:
            raise ValueError("model must not be None")

        if file_full_path is None:
            if epoch is not None:
                file_full_path = f"epoch_{epoch}/model/model.h5"
            else:
                file_full_path = "model.h5"

        # PyTorch 모델이면 확장자를 .pth로 변경 (.h5는 Keras 전용)
        if hasattr(model, 'state_dict') and file_full_path.endswith('.h5'):
            file_full_path = file_full_path[:-3] + '.pth'

        save_uri = f"{self.__save_url}/{file_full_path}"

        # TensorFlow/Keras 모델은 H5에 inference_info를 직접 내장한 후 업로드
        is_tf_model = hasattr(model, 'trainable_variables') and not hasattr(model, 'state_dict')
        if is_tf_model and file_full_path.endswith('.h5'):
            self._upload_tensorflow_model_h5(model, save_uri)
        else:
            mpp.daq.object_service.upload_model(model, uri=save_uri, inference_info=self.__inference_info, channel=self.__operation_channel, access_token=self.__access_token, chunk_size=self.__chunk_size)

        self._register_saved_file(file_full_path, file_type='model')
        return self

    def _upload_tensorflow_model_h5(self, model, save_uri: str):
        """TF 모델을 H5로 저장 → inference_info attrs 추가 → upload_object."""
        import io
        import h5py
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save(tmp_path, save_format='h5')

            if self.__inference_info is not None:
                inference_info_json = json.dumps(self.__inference_info, ensure_ascii=False)
                with h5py.File(tmp_path, 'a') as f:
                    f.attrs['inference_info'] = inference_info_json

            with open(tmp_path, 'rb') as f:
                stream = io.BufferedReader(io.BytesIO(f.read()))

            uri_path, fileinfo = os.path.split(save_uri)
            filename, ext = os.path.splitext(fileinfo)
            mpp.daq.object_service.upload_object(
                stream=stream,
                filename=filename,
                extension=ext,
                uri=uri_path,
                channel=self.__operation_channel,
                access_token=self.__access_token,
                chunk_size=self.__chunk_size
            )
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def save_file(self, file, file_full_path: str):
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

    def save_csv(self, csv, file_full_path: str):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if csv is None:
            raise ValueError("csv must not be None")

        save_uri = f"{self.__save_url}/{file_full_path}"
        mpp.intel64.save_csv(csv, save_uri, channel=self.__operation_channel, access_token=self.__access_token, chunk_size=self.__chunk_size)
        self._register_saved_file(file_full_path, file_type='csv')
        return self

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
