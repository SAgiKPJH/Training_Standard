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
    
    def init_save_url(self, bucket_url: str, operation_channel: str, access_token: str, chunk_size: int):
        """
        Sets the base URL where all files will be saved.
        Example:
            builder.set_save_url("s3://my-bucket/my-model/")
        """
        if not isinstance(bucket_url, str):
            raise TypeError("bucket_url must be a string")
        if not isinstance(operation_channel, str):
            raise TypeError("operation_channel must be a string")
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
   
    def save_epoch_file(self, epoch:int, file, file_full_path:str):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if file is None:
            raise ValueError("file must not be None")
        
        save_uri = f"{self.__save_url}/{epoch}/{file_full_path}"
        mpp.intel64.save(file, save_uri, channel=self.__operation_channel, access_token=self.__access_token, chunk_size=self.__chunk_size)
        file_type = 'file'
        if file_full_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', 'tiff', 'tif')):
            file_type = 'image'
        self._register_saved_file(file_full_path, file_type=file_type)
        return self
    
    def save_epoch_model(self, epoch:int, model, file_full_path:str):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if model is None:
            raise ValueError("model must not be None")
        
        save_uri = f"{self.__save_url}/{epoch}/{file_full_path}"
        mpp.daq.object_service.upload_model(model, uri=save_uri, inference_info=self.__inference_info, channel=self.__operation_channel, access_token=self.__access_token, chunk_size=self.__chunk_size)
        self._register_saved_file(file_full_path, file_type='model')
        return self
    
    def save_epoch_csv(self, epoch:int, csv, file_full_path:str):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if csv is None:
            raise ValueError("csv must not be None")
        
        save_uri = f"{self.__save_url}/{epoch}/{file_full_path}"
        mpp.intel64.save_csv(csv, save_uri, channel=self.__operation_channel, access_token=self.__access_token, chunk_size=self.__chunk_size)
        self._register_saved_file(file_full_path, file_type='csv')
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
    
    def score_list_graph_image(self, total_epoch, loss_list, y_max=None, title="", color='r'):
        y_section = 100
        if 0<=y_max<3: y_section = 0.1
        elif 3<=y_max<10 : y_section = 1
        elif 10<=y_max<50 : y_section = 5
        elif 50<=y_max<100 : y_section = 10
        elif 100<=y_max<500 : y_section = 50
        elif 500<=y_max<1000 : y_section = 100
        elif 1000<=y_max<5000 : y_section = 500
        elif 5000<=y_max<10000 : y_section = 1000
        elif 10000<=y_max : y_section = 5000

        x_section = 10
        if 1<=total_epoch<=10: x_section = 1
        elif 10<total_epoch<=50: x_section = 5
        elif 50<total_epoch<=100: x_section = 10
        elif 100<total_epoch<=500 : x_section = 50
        elif 500<total_epoch : x_section = 100

        axes = plt.axes()
        axes.set_xlim([1, total_epoch])
        axes.set_ylim([0, y_max])

        x_axis = list(range(0, total_epoch+1, x_section))
        x_axis[0] = 1
        plt.xticks(x_axis)
        plt.yticks(list(np.arange(0, y_max, y_section)))
        plt.plot(range(1, len(loss_list)+1), loss_list, color, label=title)

        plt.ylabel("loss")
        plt.xlabel("Epoch")
        plt.legend()

        loss_image_buffer = io.BytesIO()
        plt.savefig(loss_image_buffer, format='png')

        img_arr = np.frombuffer(loss_image_buffer.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.clf()
        plt.close()

        return img

    def confusion_matrix_image(self, true_list, pred_list, labels):
        matrix = confusion_matrix(true_list, pred_list, labels=[x for x in range(len(labels))])
        plt.figure(figsize=(9,9))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
        plt.title("Confusion Matrix")
        plt.colorbar()
        marks = np.arange(len(labels))
        nlabels = []
        for k in range(len(matrix)):
            nlabel = f'{labels[k]}'
            nlabels.append(nlabel)

        plt.xticks(marks, labels, rotation=45)
        plt.yticks(marks, nlabels, rotation=45)

        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, matrix[i, j], horizontalalignment="center", color="black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        matrix_buffer = io.BytesIO()
        plt.savefig(matrix_buffer, format='png')

        img_arr = np.frombuffer(matrix_buffer.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.clf()
        plt.close()

        return img

    def save_summary_and_close(self, metric_full_path="metrics.csv"):
        self._save_metrics(metric_full_path)
        self._save_file_index()
        return None
    
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
    
    def _save_metrics(self, file_full_path="metrics.csv"):
        data = self._create_metrics_csv()
        self.save_csv(data, os.path.join(self.__save_url, file_full_path))

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

    def _save_file_index(self, index_path: str = "file_index.json") -> None:
        """저장된 파일 목록을 JSON 파일로 저장합니다."""
        index_content = json.dumps(self.__saved_files, indent=2, ensure_ascii=False)
        self.save_file(index_content, index_path)
            
    # def load_file_index(self, index_path: str = "file_index.json") -> Dict[str, list]:
    #     """저장된 파일 목록을 JSON 파일에서 로드합니다."""
    #     load_uri = f"{self.__operation_builder.get_bucket_url()}/{self.__hierarchy_root}/{index_path}"
        
    #     try:
    #         with open(load_uri, 'r', encoding='utf-8') as f:
    #             self.__saved_files = json.load(f)
    #     except FileNotFoundError:
    #         self.__saved_files = {}
            
    #     return self.__saved_files

    # def get_saved_files(self) -> Dict[str, list]:
    #     """현재 저장된 파일 목록을 반환합니다."""
    #     return self.__saved_files.copy()

    def build(self):
        return self