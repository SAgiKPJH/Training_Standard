import io
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import mpp

class Save_Builder:
    def __init__(self):
        self.__train_loss_list = list()
        self.__valid_loss_list = list()
        self.__inference_info = None
        self.__hierarchy_root = None
        self.__label_info = None
        self.__operation_builder = None
        self.__num_classes = None

    def initialize(self, operation_builder, num_classes, label_info):
        self.__operation_builder = operation_builder
        self.__num_classes = num_classes
        self.__label_info = label_info
        return self
    
    def init_inference_info(self, input_size):
        self.__inference_info = {'inference_info' : json.dumps({"input_size": input_size, "label_info": self.__label_info})}
        return self
    
    def append_train_loss(self, loss):
        self.__train_loss_list.append(loss)
        return self
    
    def append_valid_loss(self, loss):
        self.__valid_loss_list.append(loss)
        return self
    
    def set_hierarchy_root(self, hierarchy_root):
        self.__hierarchy_root = hierarchy_root
        return self
    
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

        confusion_matrix_img = self.confusion_matrix_image(label_list, predict_list, labels=[self.__label_info[f'label_{i}']['name'] for i in range(self.__num_classes)])
        self.save_file(confusion_matrix_img, "confusion_matrix/confusion_matrix.png")

    def save_train_valid_csv(self):
        data = self.create_csv()
        self.save_csv(data, "train_loss_csv/train_loss_csv.csv")

    def create_csv(self):
        has_valid = self.__valid_loss_list is not None

        if has_valid:
            result = [['Epoch', 'Train Loss', 'Valid Loss', 'Train Accuracy', 'Valid Accuracy']]
            max_length = max(len(self.__train_loss_list), len(self.__valid_loss_list))
        else:
            result = [['Epoch', 'Train Loss', 'Train Accuracy']]
            max_length = len(self.__train_loss_list)

        for epoch in range(max_length):
            train_loss = self.__train_loss_list[epoch] if epoch < len(self.__train_loss_list) else None
            train_acc = 100.0 - train_loss if train_loss is not None else None

            if has_valid:
                valid_loss = self.__valid_loss_list[epoch] if epoch < len(self.__valid_loss_list) else None
                valid_acc = 100.0 - valid_loss if valid_loss is not None else None
                result.append([epoch + 1, train_loss, valid_loss, train_acc, valid_acc])
            else:
                result.append([epoch + 1, train_loss, train_acc])

        return result

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
        result = [['Epoch', header]] + [[epoch, score] for epoch, score in enumerate(score_list, 1)]
        return result
    
    def build(self):
        return self 