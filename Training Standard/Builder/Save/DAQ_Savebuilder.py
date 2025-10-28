import os
import json
import matplotlib.pyplot as plt
import numpy as np
import io
import cv2
import itertools
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, Optional

"""
Builder class for saving training and validation metrics, model checkpoints, and related files.
Usage Example:
    builder = Save_Builder()
        .init_inference_info(label_info = {...}, etc={'input_size': [224,224,3]})
        .init_save_url("s3://my-bucket/my-model/")
"""
class Save_Builder:
    def __init__(self):
        self.__metrics = dict()
        # keep backward-compatible lists for older callers
        self.__train_loss_list = []
        self.__valid_loss_list = []
        # placeholders used by save methods
        self.__hierarchy_root = None
        self.__operation_builder = None
        self.__save_url = None
        self.__inference_info = None
    
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
    
    def init_save_url(self, save_url: str):
        """
        Sets the base URL where all files will be saved.
        Example:
            builder.set_save_url("s3://my-bucket/my-model/")
        """
        if not isinstance(save_url, str):
            raise TypeError("save_url must be a string")

        self.__save_url = save_url
        return self
    
    def append_metrics(self, metrics: Dict[str, Any]):
        """Convenience API: append an arbitrary dict of metric name -> value as one record.

        Example: append_metrics({'Train Loss': 0.001, 'Train Acc': 99.2, 'best_model': True})

        If the dict contains an 'epoch' key it will be ignored. Use any keys you like;
        the builder will align metric lists automatically.
        """
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict")

        # remove any explicit 'epoch' key if provided
        metrics_to_append = {k: v for k, v in metrics.items() if k != 'epoch'}
        return self.append_epoch(metrics_to_append)
    
    def save_metrics(self, dir_path="metrics", file_name="metrics.csv"):
        # save the flexible metrics CSV (epoch x metrics)
        data = self.create_csv()
        self.save_csv(data, os.path.join(self.__save_url, dir_path, file_name))

    def create_csv(self):
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

    def save_training(self, model, epoch, save_epoch, inputs):
        train_loss_image = self.score_list_graph_image(epoch, self.__train_loss_list, self.__train_loss_list[0], "Train Loss Graph", 'r')
        self.save_file(train_loss_image, "train_loss/train_loss_image.png")

        train_loss_csv = self.score_list_csv(self.__train_loss_list)
        self.save_csv(train_loss_csv, "train_loss/train_loss_csv.csv")

        if epoch % save_epoch == 0 or epoch == epoch:
            self.upload_model(model, "model/model.pth", inputs)

    def save_validateion(self, epoch, label_list, predict_list):
        valid_loss_image = self.score_list_graph_image(epoch, self.__valid_loss_list, self.__valid_loss_list[0], "Valid Loss Graph", 'g')
        self.save_file(valid_loss_image, "valid_loss/valid_loss_graph.png")

        valid_loss_csv = self.score_list_csv(self.__valid_loss_list)
        self.save_csv(valid_loss_csv, "valid_loss/valid_loss_csv.csv")

        confusion_matrix = self.confusion_matrix_image(label_list, predict_list, labels=[self.__label_info[f'label_{i}']['name'] for i in range(self.__num_classes)])
        self.save_file(confusion_matrix, "confusion_matrix/confusion_matrix.png")

    def save_csv(self, csv, path):
        save_uri = f"{self.__operation_builder.get_bucket_url()}/{self.__hierarchy_root}/{path}"
        mpp.intel64.save_csv(csv, save_uri, channel=self.__operation_builder.get_operation_channel(), access_token=self.__operation_builder.get_access_token(), chunk_size=self.__operation_builder.get_chunk_size())
        return self
    
    def save_file(self, file, path):
        save_uri = f"{self.__operation_builder.get_bucket_url()}/{self.__hierarchy_root}/{path}"
        mpp.intel64.save(file, save_uri, channel=self.__operation_builder.get_operation_channel(), access_token=self.__operation_builder.get_access_token(), chunk_size=self.__operation_builder.get_chunk_size())
        return self

    def upload_model(self, model, path, inputs):
        model_save_uri = f"{self.__operation_builder.get_bucket_url()}/{self.__hierarchy_root}/{path}"
        mpp.daq.object_service.upload_model(model, uri=model_save_uri, inference_info=self.__inference_info, example=inputs, channel=self.__operation_builder.get_operation_channel(), access_token=self.__operation_builder.get_access_token(), chunk_size=self.__operation_builder.get_chunk_size())
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

    def score_list_csv(self, score_list, header='Loss'):
        rows = []
        for epoch, score in enumerate(score_list, 1):
            rows.append([epoch, '' if score is None else score])

        result = [['Epoch', header]] + rows
        return result
    
    def build(self):
        return self