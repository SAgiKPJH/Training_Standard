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

    def save_epoch_file(self, epoch:int, file, file_full_path:str):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if file is None:
            raise ValueError("file must not be None")

        save_path = os.path.join(self.__save_url, str(epoch), file_full_path)
        self._save_to_local(file, save_path)

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

        save_path = os.path.join(self.__save_url, str(epoch), file_full_path)
        self._save_model_to_local(model, save_path)
        self._register_saved_file(file_full_path, file_type='model')
        return self

    def save_epoch_csv(self, epoch:int, csv, file_full_path:str):
        if not isinstance(file_full_path, str):
            raise TypeError("file_full_path must be a string")
        if not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if csv is None:
            raise ValueError("csv must not be None")

        save_path = os.path.join(self.__save_url, str(epoch), file_full_path)
        self._save_csv_to_local(csv, save_path)
        self._register_saved_file(file_full_path, file_type='csv')
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

        # PyTorch 모델인지 확인
        if hasattr(model, 'state_dict'):
            torch.save(model.state_dict(), save_path)
        else:
            # 기타 모델은 pickle로 저장
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)

        # inference_info가 있으면 함께 저장
        if self.__inference_info is not None:
            info_path = save_path.rsplit('.', 1)[0] + '_info.json'
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(self.__inference_info, f, indent=2, ensure_ascii=False)

    def _save_csv_to_local(self, csv_data, save_path: str):
        """CSV 데이터를 로컬 파일 시스템에 저장합니다."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', encoding='utf-8', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(csv_data)

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
        self.save_csv(data, file_full_path)

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
        save_path = os.path.join(self.__save_url, index_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(index_content)

    def build(self):
        return self
