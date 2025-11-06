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
import cv2
import grpc
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
##!--{"Name":"hyperparameter","Type":"using_advanced_feature","Key":"using_advanced_feature","Value":"","Category":""}
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
        "using_advanced_feature": true,
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

def RecipeRun(**kwargs):

    if kwargs['hyperparameter']['using_gpu'] and not torch.cuda.is_available():
        raise Exception("GPU is not avaiable")

    authentication = kwargs['authentication']
    access_token = authentication['access_token']
    operation_channel = grpc.insecure_channel(authentication['operation_service_address'])

    hyperparameter = kwargs['hyperparameter']
    hyperparameter['epoch'] = int(hyperparameter['epoch'])
    hyperparameter['batch_size'] = int(hyperparameter['batch_size'])
    chunk_size = kwargs['chunk_size']
    device = 'cuda' if kwargs['hyperparameter']['using_gpu'] else 'cpu'

    bucket_id = kwargs['result']['id']
    if urlparse(bucket_id).scheme == '':
        create_bucket(operation_channel, access_token, bucket_id, bucket_id+"_title", kwargs['result']['volume_id'])
        result_uri = f"object:///{bucket_id}"
    else:
        result_uri = bucket_id

    dataset = kwargs['gt_dataset']
    classification_gts = get_classification_gts(dataset['gt_dataset_id'], operation_channel, access_token)
    classification_gt_dataset = get_classification_gt_dataset(operation_channel, access_token, dataset['gt_dataset_id'])
    label_info, class_code_info, num_classes = generate_label_data(classification_gt_dataset.class_code_set_id, operation_channel, access_token)

    logger.info("Temp Folder Create")
    local_download_path = os.path.join(f"/temp/{uuid.uuid4()}")
    train_data_info, validation_data_info = data_download(hyperparameter['train_ratio'],
                                                        local_download_path,
                                                        classification_gts,
                                                        class_code_info,
                                                        operation_channel,
                                                        access_token)

    try:
        transform = preprocessing(input_size=hyperparameter['input_size'],
                              mean=hyperparameter['normalize_mean'],
                              stdev=hyperparameter['normalize_stdev'])

        train_dataset = ClassificationDataset(train_data_info, transform)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=hyperparameter['batch_size'],
                                                    shuffle=True,
                                                    num_workers=0,
                                                    drop_last=True)

        valid_flag = False
        if len(validation_data_info[0]) > 0:
            valid_flag = True
            validation_dataset = ClassificationDataset(validation_data_info, transform)
            validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=1,
                                                    shuffle=hyperparameter['validation_save_random'],
                                                    num_workers=0,
                                                    drop_last=False)

        model = torchvision.models.googlenet(num_classes=num_classes, init_weights=False)
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = Optimizer(hyperparameter['optimizer_name'], model.parameters(), hyperparameter['lr']).function

        total_iteration = len(train_data_loader)
        train_loss_list = list()
        valid_loss_list = list()

        inference_info = {'inference_info' : json.dumps({"input_size": hyperparameter['input_size'], "label_info": label_info})}
        iteration_start_time = time.time()
        for epoch in range(1, hyperparameter['epoch']+1):
            epoch_start_time = time.time()

            train_epoch_loss = 0.0
            valid_epoch_loss = 0.0
            for n_epoch, batch in enumerate(train_data_loader, 1):
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                with torch.cuda.amp.autocast(enabled=hyperparameter['using_amp']):
                    output, *_ = model(inputs)
                    loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()
                if n_epoch % (total_iteration // 10) == 0 or n_epoch == total_iteration:
                    iteration_elapsed_time = time.time() - iteration_start_time
                    logger.info(f"Epoch : {epoch:4d}, Iterations : {n_epoch:4d}/{total_iteration:4d}, Loss : {loss : 4.4f}, Time : {iteration_elapsed_time : 4.4f}")
                    iteration_start_time = time.time()

            train_loss_list.append(train_epoch_loss / n_epoch)
            hierarchy_root = f"epoch_{epoch}"

            if valid_flag:
                predict_list = np.array([])
                label_list = np.array([])

                with torch.no_grad():
                    for valid_epoch, valid_batch in enumerate(validation_data_loader, 1):
                        valid_inputs = valid_batch[0].to(device)
                        valid_labels = valid_batch[1].to(device)

                        with torch.cuda.amp.autocast(enabled=hyperparameter['using_amp']):
                            valid_outputs, *_ = model(valid_inputs)
                            vlaid_loss = criterion(valid_outputs, valid_labels)

                        valid_epoch_loss += vlaid_loss.item()
                        predict_list = np.concatenate([predict_list, valid_outputs.argmax(dim=1).cpu().numpy()], 0)
                        label_list = np.concatenate([label_list, valid_labels.cpu().numpy()], 0)

                valid_loss_list.append(valid_epoch_loss / valid_epoch)
                
                valid_loss_image = score_list_graph_image(hyperparameter['epoch'], valid_loss_list, valid_loss_list[0], "Valid Loss Graph", 'g')
                valid_loss_image_save_uri = f"{result_uri}/{hierarchy_root}/valid_losss_graph/valid_loss_graph.png"
                mpp.intel64.save(valid_loss_image, valid_loss_image_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

                valid_loss_csv = score_list_csv(valid_loss_list)
                valid_loss_csv_save_uri = f"{result_uri}/{hierarchy_root}/valid_loss_csv/valid_loss_csv.csv"
                mpp.intel64.save_csv(valid_loss_csv, valid_loss_csv_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

                confusion_matrix = confusion_matrix_image(label_list, predict_list, labels=[label_info[f'label_{i}']['name'] for i in range(num_classes)])
                confusion_matrix_save_uri = f"{result_uri}/{hierarchy_root}/confusion_matrix/confusion_matrix.png"
                mpp.intel64.save(confusion_matrix, confusion_matrix_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

            train_loss_image = score_list_graph_image(hyperparameter['epoch'], train_loss_list, train_loss_list[0], "Train Loss Graph", 'r')
            train_loss_image_save_uri = f"{result_uri}/{hierarchy_root}/train_loss_image/train_loss_image.png"
            mpp.intel64.save(train_loss_image, train_loss_image_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

            train_loss_csv = score_list_csv(train_loss_list)
            train_loss_csv_save_uri = f"{result_uri}/{hierarchy_root}/train_loss_csv/train_loss_csv.csv"
            mpp.intel64.save_csv(train_loss_csv, train_loss_csv_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

            if epoch % hyperparameter['save_epoch'] == 0 or epoch == hyperparameter['epoch']:
                model_save_uri = f"{result_uri}/{hierarchy_root}/model/model.pth"
                mpp.daq.object_service.upload_model(model, uri=model_save_uri, inference_info=inference_info, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

            epoch_valid_loss_mean = valid_epoch_loss/len(validation_data_loader) if valid_flag else 0
            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info(f"Epoch:[{epoch:4d}/{hyperparameter['epoch']:4d}], "
                    f"Train Loss: {train_epoch_loss/total_iteration:4.4f}, "
                    f"Valid Loss : {epoch_valid_loss_mean:4.4f}, "
                    f"Time: {epoch_elapsed_time:4.2f}s")

    except Exception as e:
        logger.error(f"Error Message : {e}")
        raise Exception(f"Error Message : {e}")

    finally:
        shutil.rmtree(local_download_path)
        logger.info("Temp Folder Delete")


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
    validation_len = int(len(classification_gts) * (1-train_ratio))    

    train_uri_list = list()
    train_label_list = list()

    validation_uri_list = list()
    validation_label_list = list()

    for i in range(len(classification_gts)):
        image_id = classification_gts[i].image_id
        class_code = classification_gts[i].class_code.value

        uri = f"dataset:///?image_id={image_id}"
        image = mpp.daq.intel64.load(uri, False, channel=operation_channel, access_token=access_token)
        download_path = os.path.join(local_download_path, f"{image_id}.png")
        mpp.intel64.save(image, download_path)

        train_uri_list.append(download_path)
        train_label_list.append(class_code_info[class_code])

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

    return classification_gt_dataset

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


if __name__ =="__main__":
    kwargs = json.loads(parameters)
    kwargs['authentication']['operation_service_address'] = ""
    kwargs['authentication']['access_token'] = ""

    RecipeRun(**kwargs)