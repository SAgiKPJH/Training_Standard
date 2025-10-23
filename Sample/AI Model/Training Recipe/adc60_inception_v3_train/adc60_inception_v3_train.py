import itertools
import shutil
from urllib.parse import urlparse
import uuid
import random
import os
import io
import logging
import time
import json
import grpc
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from google.protobuf.wrappers_pb2 import StringValue
import mpp
from mpp.daq import protos


##!--{"Name":"hyperparameter","Type":"epoch","Key":"epoch","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"batch_size","Key":"batch_size","Value":"","Category":""}
##!--{"Name":"result","Type":"result","Key":"id","Value":"","Category":""}
##!--{"Name":"authentication","Type":"system_address","Key":"operation_service_address","Value":"","Category":""}
##!--{"Name":"authentication","Type":"access_token","Key":"access_token","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id","Value":"","Category":""}

##$--
parameters = '''{
    "hyperparameter":{
        "epoch" : 20,
        "save_epoch" : 1,
        "batch_size" : 16,
        "lr" : 1e-3,
        "optimizer_name" : "Adam",
        "input_size" : 299,
        "normalize_mean" : 0.5,
        "normalize_stdev" : 0.5,
        "using_gpu" : true,
        "using_amp" : true,
        "train_ratio" : 0.8,
        "validation_save_random" : false
    },
    "authentication": {
        "operation_service_address": "",
        "access_token" : ""
    },
    "result":{
        "id":"",
        "volume_id":"default" 
    },
    "gt_dataset":{
        "gt_dataset_id" : ""
    },
    "chunk_size" : 100000
}'''
##$--

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')


# 준비 Builder
class RecipeBuilder:
    def __init__(self):
        self.keyword_arguments = {}
        self.operation_channel = None
        self.access_token = None
        self.device = None
        self.result_uri = None
        self.train_data_info = None
        self.validation_data_info = None
        self.local_download_path = None
        self.label_info = None
        self.class_code_info = None
        self.num_classes = None

    def set_parameters(self, parameters):
        self.keyword_arguments = json.loads(parameters)
        return self

    def set_arguments(self, **keyword_arguments):
        self.keyword_arguments = keyword_arguments
        return self

    def initialize_arguments(self):
        self.keyword_arguments['authentication']['operation_service_address'] = ""
        self.keyword_arguments['authentication']['access_token'] = ""
        self.keyword_arguments['gt_dataset']['gt_dataset_id'] = r""
        self.keyword_arguments['result']['id'] = r""
        return self

    def check_using_gpu(self):
        if self.keyword_arguments['hyperparameter']['using_gpu'] and not torch.cuda.is_available():
            raise Exception("GPU is not avaiable")
        return self

    def initialize_server_settings(self):
        authentication = self.keyword_arguments['authentication']
        if authentication['operation_service_address']:
            self.operation_channel = grpc.insecure_channel(authentication['operation_service_address'])
        self.access_token = authentication['access_token']
        return self

    def initialize_variable(self):
        self.device = 'cuda' if self.keyword_arguments['hyperparameter']['using_gpu'] else 'cpu'
        return self

    def get_result_uri(self):
        bucket_id = self.keyword_arguments['result']['id']
        if urlparse(bucket_id).scheme == '':
            create_bucket(self.operation_channel, self.access_token, bucket_id, bucket_id+"_title", self.keyword_arguments['result']['volume_id'])
            self.result_uri = f"object:///{bucket_id}"
        else:
            self.result_uri = bucket_id
        return self

    def create_train_dataset(self):
        dataset = self.keyword_arguments['gt_dataset']
        if self.operation_channel: 
            classification_gts = get_classification_gts(dataset['gt_dataset_id'], self.operation_channel, self.access_token)
            class_code_set_id = get_classification_gt_dataset(self.operation_channel, self.access_token, dataset['gt_dataset_id'])
        else: 
            classification_gts = dataset['gt_dataset_id']
            class_code_set_id = dataset['gt_dataset_id']

        self.label_info, class_code_info, self.num_classes = generate_label_data(class_code_set_id, self.operation_channel, self.access_token)

        logger.info("Create Train Dataset")
        self.local_download_path = os.path.join(f"/temp/{uuid.uuid4()}")
        self.train_data_info, self.validation_data_info = data_download(self.keyword_arguments['hyperparameter']['train_ratio'],
                                                                        self.local_download_path,
                                                                        classification_gts,
                                                                        class_code_info,
                                                                        self.operation_channel,
                                                                        self.access_token)
        return self

    def build(self):
        return TrainRun(self.operation_channel,
                        self.access_token, 
                        self.device,
                        self.result_uri,
                        self.train_data_info,
                        self.validation_data_info,
                        self.local_download_path,
                        self.label_info, 
                        self.num_classes,
                        **self.keyword_arguments)

    # Run
class TrainRun:
    def __init__(self,
                 operation_channel,
                 access_token,
                 device,
                 result_uri,
                 train_data_info,
                 validation_data_info,
                 local_download_path,
                 label_info,
                 num_classes,
                 **keyword_arguments):
        self.keyword_arguments = keyword_arguments
        self.hyperparameter = {}
        self.operation_channel = operation_channel
        self.access_token = access_token 
        self.device = device
        self.chunk_size = self.keyword_arguments['chunk_size']
        self.result_uri = result_uri
        self.train_data_info = train_data_info
        self.validation_data_info = validation_data_info
        self.local_download_path = local_download_path
        self.label_info = label_info
        self.num_classes = num_classes
        self.transform = None
        self.valid_flag = False
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.total_iteration = None
        self.train_loss_list = list()
        self.valid_loss_list = list()
        self.train_data_loader = None
        self.validation_data_loader = None
        self.train_epoch_loss = 0.0
        self.valid_epoch_loss = 0.0
        self.iteration_elapsed_time = None
        self.running_time = None
        self.inputs = None

    def get_hyperparameter(self):
        self.hyperparameter = self.keyword_arguments['hyperparameter']
        self.hyperparameter['epoch'] = int(self.hyperparameter['epoch'])
        self.hyperparameter['batch_size'] = int(self.hyperparameter['batch_size'])
        return self

    def set_transform(self):
        self.transform = preprocessing(input_size=self.hyperparameter['input_size'],
                                       mean=self.hyperparameter['normalize_mean'],
                                       stdev=self.hyperparameter['normalize_stdev'])
        return self

    def set_train_dataset(self):
        train_dataset = ClassificationDataset(self.train_data_info, self.transform)
        self.train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=self.hyperparameter['batch_size'],
                                                        shuffle=True,
                                                        num_workers=0,
                                                        drop_last=True)
        self.total_iteration = len(self.train_data_loader)
        return self

    def set_valid_dataset(self):
        if len(self.validation_data_info[0]) <= 0:
            return self

        self.valid_flag = True
        validation_dataset = ClassificationDataset(self.validation_data_info, self.transform)
        self.validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                             batch_size=1,
                                                             shuffle=self.hyperparameter['validation_save_random'],
                                                             num_workers=0,
                                                             drop_last=False)
        return self

    def load_model(self):
        self.model = torchvision.models.inception_v3(num_classes=self.num_classes, init_weights=False)
        self.model.to(self.device)
        return self

    def get_optimizer(self):
        self.optimizer = Optimizer(self.hyperparameter['optimizer_name'], self.model.parameters(), self.hyperparameter['lr']).function
        return self

    def do_training(self):
        inference_info = {'inference_info' : json.dumps({"input_size": self.hyperparameter['input_size'], "label_info": self.label_info})}
        self.iteration_start_time = time.time()
        self.running_time = time.time()
        for epoch in range(1, self.hyperparameter['epoch']+1):
            self.do_trainings(epoch, inference_info)

    def do_trainings(self, epoch, inference_info):
        epoch_start_time = time.time()
        self.train_epoch_loss = 0.0
        self.valid_epoch_loss = 0.0
        for n_epoch, batch in enumerate(self.train_data_loader, 1):
            self.do_trainingss(epoch, n_epoch, batch)

        self.train_loss_list.append(self.train_epoch_loss / n_epoch)
        
        self.do_validation(epoch)
        self.save_csv_train(epoch)
        self.save_model(epoch, inference_info)
        self.log(epoch, epoch_start_time)

    def do_trainingss(self, epoch, n_epoch, batch):
        self.inputs = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        with torch.cuda.amp.autocast(enabled=self.hyperparameter['using_amp']):
            output, _ = self.model(self.inputs)
            loss = self.criterion(output, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_epoch_loss += loss.item()
        if n_epoch % (self.total_iteration // 10) == 0 or n_epoch == self.total_iteration:
            iteration_elapsed_time = time.time() - self.iteration_start_time
            logger.info(f"Epoch : {epoch:4d}, Iterations : {n_epoch:4d}/{self.total_iteration:4d}, Loss : {loss : 4.4f}, Time : {iteration_elapsed_time : 4.4f}")
            self.iteration_start_time = time.time()

    def do_validation(self, epoch):
        if self.valid_flag == False:
            return self
        
        predict_list = np.array([])
        label_list = np.array([])

        with torch.no_grad():
            self.model.eval()
            for valid_epoch, valid_batch in enumerate(self.validation_data_loader, 1):
                valid_inputs = valid_batch[0].to(self.device)
                valid_labels = valid_batch[1].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.hyperparameter['using_amp']):
                    valid_outputs = self.model(valid_inputs)
                    vlaid_loss = self.criterion(valid_outputs, valid_labels)

                self.valid_epoch_loss += vlaid_loss.item()

                predict_list = np.concatenate([predict_list, valid_outputs.argmax(dim=1).cpu().numpy()], 0)
                label_list = np.concatenate([label_list, valid_labels.cpu().numpy()], 0)
            self.model.train()

        self.valid_loss_list.append(self.valid_epoch_loss / valid_epoch)

        # self.save_csv_valid(epoch, predict_list, label_list)

    def save_csv_valid(self, epoch, predict_list, label_list):
        hierarchy_root = f"epoch_{epoch}"
        valid_loss_image = score_list_graph_image(self.hyperparameter['epoch'], self.valid_loss_list, self.valid_loss_list[0], "Valid Loss Graph", 'g')
        valid_loss_image_save_uri = f"{self.result_uri}/{hierarchy_root}/valid_losss_graph/valid_loss_graph.png"
        mpp.intel64.save(valid_loss_image, valid_loss_image_save_uri, channel=self.operation_channel, access_token=self.access_token, chunk_size=self.chunk_size)

        valid_loss_csv = score_list_csv(self.valid_loss_list)
        valid_loss_csv_save_uri = f"{result_uri}/{hierarchy_root}/valid_loss_csv/valid_loss_csv.csv"
        mpp.intel64.save_csv(valid_loss_csv, valid_loss_csv_save_uri, channel=self.operation_channel, access_token=self.access_token, chunk_size=self.chunk_size)

        confusion_matrix = confusion_matrix_image(label_list, predict_list, labels=[self.label_info[f'label_{i}']['name'] for i in range(self.num_classes)])
        confusion_matrix_save_uri = f"{result_uri}/{hierarchy_root}/confusion_matrix/confusion_matrix.png"
        mpp.intel64.save(confusion_matrix, confusion_matrix_save_uri, channel=self.operation_channel, access_token=self.access_token, chunk_size=self.chunk_size)

    def save_csv_train(self, epoch):
        hierarchy_root = f"epoch_{epoch}"
        train_loss_image = score_list_graph_image(self.hyperparameter['epoch'], self.train_loss_list, self.train_loss_list[0], "Train Loss Graph", 'r')
        train_loss_image_save_uri = f"{self.result_uri}/{hierarchy_root}/train_loss_image/train_loss_image.png"
        mpp.intel64.save(train_loss_image, train_loss_image_save_uri, channel=self.operation_channel, access_token=self.access_token, chunk_size=self.chunk_size)

        train_loss_csv = create_csv(self.train_loss_list, self.valid_loss_list)
        train_loss_csv_save_uri = f"{self.result_uri}/{hierarchy_root}/train_loss_csv/train_loss_csv.csv"
        mpp.intel64.save_csv(train_loss_csv, train_loss_csv_save_uri, channel=self.operation_channel, access_token=self.access_token, chunk_size=self.chunk_size)

    def save_model(self, epoch, inference_info):
        hierarchy_root = f"epoch_{epoch}"

        if epoch % self.hyperparameter['save_epoch'] == 0 or epoch == self.hyperparameter['epoch']:
            model_save_uri = f"{self.result_uri}/{hierarchy_root}/model/model.pth"
            mpp.daq.object_service.upload_model(self.model, uri=model_save_uri, inference_info=inference_info, example=self.inputs, channel=self.operation_channel, access_token=self.access_token, chunk_size=self.chunk_size)

    def log(self, epoch, epoch_start_time):
        epoch_valid_loss_mean = self.valid_epoch_loss/len(self.validation_data_loader) if self.valid_flag else 0
        epoch_elapsed_time = time.time() - epoch_start_time

        training_running_time = time.time() - self.running_time

        remaining_epochs = self.hyperparameter['epoch'] - epoch
        estimated_remaining_time = remaining_epochs * epoch_elapsed_time
        
        logger.info(f"Epoch:[{epoch:4d}/{self.hyperparameter['epoch']:4d}], "
                    f"Train Loss: {self.train_epoch_loss/self.total_iteration:4.4f}, "
                    f"Valid Loss : {epoch_valid_loss_mean:4.4f}, "
                    f"Time: {epoch_elapsed_time:4.2f}s, ")
                    
        logger.info("MonitoringData:"
                    f"Epoch:[{epoch:4d}/{self.hyperparameter['epoch']:4d}]({epoch_elapsed_time:4.2f}s), "
                    f"Running Time: {training_running_time / 60:.2f} minutes, "
                    f"Remain Time: {estimated_remaining_time / 60:.2f} minutes")

    def run(self):
        try:
            self.get_hyperparameter()
            self.set_transform()
            self.set_train_dataset()
            self.set_valid_dataset()
            self.load_model()
            self.get_optimizer()
            self.do_training()
            
        except Exception as e:
            logger.error(f"Error Message : {e}")
            raise Exception(f"Error Message : {e}")
        finally:
            if os.path.exists(self.local_download_path):
                shutil.rmtree(self.local_download_path)
                logger.info("Temp Folder Delete")

if __name__ =="__main__":

    builder = RecipeBuilder()
    recipe_run = (builder.set_parameters(parameters)
                         .initialize_arguments()
                         .build())
    
    recipe_run.run()

def RecipeRun(**kwargs):
    builder = RecipeBuilder()
    recipe_run = (builder.set_arguments(**kwargs)
                         .check_using_gpu()
                         .initialize_server_settings()
                         .initialize_variable()
                         .get_result_uri()
                         .create_train_dataset()
                         .build())

    recipe_run.run()


class ClassificationDataset(Dataset):
    def __init__(self, train_dataset, transform):
        self.__uri_list = train_dataset[0]
        self.__label_list = train_dataset[1]
        self.__trainform = transform

    def __len__(self):
        return len(self.__uri_list)

    def __getitem__(self, index):
        uri = self.__uri_list[index]

        image = mpp.intel64.load(uri, False)
        image = self.__trainform(image)
        label = self.__label_list[index]

        return image, label


def data_download(train_ratio,
                  local_download_path,
                  classification_gts,
                  class_code_info,
                  operation_channel,
                  access_token):
    train_ratio = min(1, train_ratio)

    train_uri_list = list()
    train_label_list = list()

    validation_uri_list = list()
    validation_label_list = list()
    if operation_channel:
        validation_len = int(len(classification_gts) * (1-train_ratio))
        for i in range(len(classification_gts)):
            image_id = classification_gts[i].image_id
            class_code = classification_gts[i].class_code.value

            uri = f"dataset:///?image_id={image_id}"
            image = mpp.daq.intel64.load(uri, False, channel=operation_channel, access_token=access_token)
            download_path = os.path.join(local_download_path, f"{image_id}.png")
            mpp.intel64.save(image, download_path)

            train_uri_list.append(download_path)
            train_label_list.append(class_code_info[class_code])
    else:
        validation_len = int(len(train_uri_list) * (1-train_ratio))    
        class_code_list = os.listdir(classification_gts)
        for index in range(len(class_code_list)):
            file_list = os.listdir(os.path.join(classification_gts, class_code_list[index]))
            for filename in file_list:
                image_path = os.path.join(classification_gts, class_code_list[index], filename)
                train_uri_list.append(image_path)
                train_label_list.append(class_code_info[index])
            
    for _ in range(validation_len):
        random_index = random.randrange(len(train_uri_list))

        valid_uri = train_uri_list.pop(random_index)
        valid_label = train_label_list.pop(random_index)

        validation_uri_list.append(valid_uri)
        validation_label_list.append(valid_label)


    return (train_uri_list, train_label_list), (validation_uri_list, validation_label_list)


def get_classification_gt_dataset(channel, access_token, gt_dataset_id):
    stub = protos.daq_dataset_classification_gt_dataset_api_v1_pb2_grpc.ClassificationGtDatasetServiceStub(channel)

    classification_gt_dataset = stub.GetClassificationGtDataset(request=protos.daq_dataset_classification_gt_dataset_api_v1_pb2.GetClassificationGtDatasetRequest(
        id=gt_dataset_id), metadata=[('authorization', f'Bearer {access_token}')])

    return classification_gt_dataset.class_code_set_id

def get_classification_gts(gt_dataset_id, channel, access_token):
    stub = protos.daq_dataset_classification_gt_dataset_api_v1_pb2_grpc.ClassificationGtDatasetServiceStub(channel)
    query_parameter = protos.daq_common_pb2.QueryParameter(
                        page_index=0,
                        page_size=-1,
                        where=StringValue(value=f"GtDatasetId=\"{gt_dataset_id}\""),
                        order_by=None)

    response = stub.ListClassificationGts(request=protos.daq_dataset_classification_gt_dataset_api_v1_pb2.ListClassificationGtsRequest(
        query_parameter=query_parameter, with_image=False), metadata=[('authorization', f'Bearer {access_token}')])

    return response.classification_gts


def score_list_graph_image(total_epoch, loss_list, y_max=None, title="", color='r'):
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


def score_list_csv(score_list, header='Loss'):
    result = [['Epoch', header]] + [[epoch, score] for epoch, score in enumerate(score_list, 1)]
    return result

def create_csv(train_loss_list, valid_loss_list):
    result = [['Epoch', 'Train Loss', 'Valid Loss', 'Train Accuracy', 'Valid Accuracy']]
    max_length = max(len(train_loss_list), len(valid_loss_list))

    for epoch in range(max_length):
        train_loss = train_loss_list[epoch] if epoch < len(train_loss_list) else None
        valid_loss = valid_loss_list[epoch] if epoch < len(valid_loss_list) else None
        result.append([epoch + 1, train_loss, valid_loss, 100.0 - train_loss, 100.0 - valid_loss])

    return result

def confusion_matrix_image(true_list, pred_list, labels):
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


def generate_label_data(class_code_set_id, channel, access_token):
    if channel: 
        stub = protos.daq_dataset_class_code_api_v1_pb2_grpc.ClassCodeServiceStub(channel)
        query_parameter = protos.daq_common_pb2.QueryParameter(
                    page_index=0,
                    page_size=-1,
                    where=StringValue(value=f"Id=\"{class_code_set_id}\""),
                    order_by=None)

        response = stub.ListClassCodeSets(request=protos.daq_dataset_class_code_api_v1_pb2.ListClassCodeSetsRequest(
            query_parameter=query_parameter), metadata=[('authorization', f'Bearer {access_token}')])

        class_info = response.class_code_sets[0].class_codes
        num_classes = len(class_info)
        label_info = {"label_count" : num_classes}
        class_code_info = dict()

        for i, class_code in enumerate(class_info):
            label_info[f'label_{i}'] = {"code" : class_code.code, "name" : class_code.name}
            class_code_info[class_code.code] = i
    else:
        class_info = os.listdir(class_code_set_id)
        num_classes = len(class_info)
        label_info = {"label_count" : num_classes}
        class_code_info = dict()

        for i, class_code in enumerate(class_info):
            label_info[f'label_{i}'] = {"code" : i, "name" : class_code}
            class_code_info[i] = i

    return label_info, class_code_info, num_classes


def preprocessing(input_size, mean, stdev):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((input_size, input_size)),
        transforms.Normalize((mean, mean, mean), (stdev, stdev, stdev))])
    return transform


class Optimizer():
    def __init__(self, optimizer_name, model_parameters, lr):
        if optimizer_name.lower() == "adam":
            self.function = optim.Adam(model_parameters, lr=lr)
        elif optimizer_name.lower() == "sgd":
            self.function = optim.SGD(model_parameters, lr=lr)
        elif optimizer_name.lower() == "adagrad":
            self.function = optim.Adagrad(model_parameters, lr=lr)
        else:
            raise Exception("Invalid Optimizer Option")

def create_bucket(channel, access_token, bucket_id, bucket_title, bucket_volume_id, properties=None, description=None):
    if properties:
        properties = StringValue(value=json.dumps(properties))
    if description:
        description = StringValue(value=description)

    stub = protos.daq_object_object_api_v1_pb2_grpc.ObjectServiceStub(channel)
    stub.CreateBucket(request=protos.daq_object_object_api_v1_pb2.CreateBucketRequest(
        id=bucket_id, title=bucket_title, properties=properties, description=description, volume_id=bucket_volume_id
    ),metadata=[('authorization', f'Bearer {access_token}')])
