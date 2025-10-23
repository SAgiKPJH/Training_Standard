import json
import logging
import os, time
from urllib.parse import urlparse
import uuid
import random
import numpy as np
import grpc
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset
import cv2
import io
import matplotlib.pyplot as plt
from google.protobuf.wrappers_pb2 import StringValue
from torchvision import transforms
import os
import shutil
import mpp
from mpp.daq import protos

##!--{"Name":"hyperparameter","Type":"epoch","Key":"epoch","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"batch_size","Key":"batch_size","Value":"","Category":""}
##!--{"Name":"result","Type":"result","Key":"id","Value":"","Category":""}
##!--{"Name":"authentication","Type":"system_address","Key":"operation_service_address","Value":"","Category":""}
##!--{"Name":"authentication","Type":"access_token","Key":"access_token","Value":"","Category":""}
##!--{"Name":"dataset","Type":"dataset","Key":"dataset_id_1","Value":"","Category":""}

##$--
parameters = '''{
    "hyperparameter":{
        "batch_size": 32,
        "sigma": 25,
        "epoch": 300,
        "save_epoch": 1,
        "lr": 1e-3,
        "milestones": [30, 60, 90],
        "gamma": 0.2,
        "loss": "mseloss",
        "optimizer": "adam",
        "scheduler": null,
        "patch_size": 50,
        "aug_mode": 1,
        "scales": [1, 0.9, 0.8],
        "stride": 50,
        "using_gpu" : true,
        "using_amp" : false,
        "train_ratio": 0.8,
        "input_channels" : 1,
        "validation_save_count" : 10,
        "validation_save_random" : false,
        "pretrained": false
    },
    "authentication": {
        "operation_service_address": "",
        "access_token" : ""
    },
    "result": {
        "id": "",
        "volume_id": "default"
    },
    "dataset":{
        "dataset_id_1" : "",
        "dataset_id_2" : "",
        "dataset_id_3" : "",
        "dataset_id_4" : "",
        "dataset_id_5" : "",
        "dataset_id_6" : "",
        "dataset_id_7" : "",
        "dataset_id_8" : "",
        "dataset_id_9" : "",
        "dataset_id_10" : ""
    },
    "chunk_size": 10000
}
'''
##$--


logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def RecipeRun(**kwargs):
    if kwargs['hyperparameter']['using_gpu'] and not torch.cuda.is_available():
        raise Exception("GPU is not avaiable")

    hyperparameter = kwargs['hyperparameter']
    hyperparameter['epoch'] = int(hyperparameter['epoch'])
    hyperparameter['batch_size'] = int(hyperparameter['batch_size'])
    chunk_size = kwargs['chunk_size']
    device = 'cuda' if kwargs['hyperparameter']['using_gpu'] else 'cpu'

    authentication = kwargs['authentication']
    access_token = authentication['access_token']
    selected_operation_address = random.choice(authentication['operation_service_address'].split(','))
    operation_channel = grpc.insecure_channel(selected_operation_address)

    bucket_id = kwargs['result']['id']
    if urlparse(bucket_id).scheme == '':
        create_bucket(operation_channel, access_token, bucket_id, bucket_id+"_title", kwargs['result']['volume_id'])
        result_uri = f"object:///{bucket_id}"
    else:
        result_uri = bucket_id

    logger.info("Temp Folder Create")
    local_download_path = os.path.join(f"/temp/{uuid.uuid4()}")
    convert_gray = True if hyperparameter['input_channels'] == 1 else False
    train_download_uri_list, validation_download_uri_list = data_download(hyperparameter['patch_size'],
                                                                          hyperparameter['stride'],
                                                                          hyperparameter['scales'],
                                                                          hyperparameter['train_ratio'],
                                                                          convert_gray,
                                                                          local_download_path,
                                                                          kwargs['dataset'],
                                                                          operation_channel,
                                                                          access_token)
    try:
        # TODO : Pre-Trained Model
        model = DnCNN(image_channels=hyperparameter['input_channels'])
        model.train()
        

        current_file_path = os.path.dirname(os.path.abspath(__file__))
        pretrained_model_path = os.path.join(current_file_path, "pretrained_model.pth")
        if hyperparameter['pretrained'] and os.path.exists(pretrained_model_path):
            logger.info("Pretrained")
            model= torch.jit.load(pretrained_model_path, map_location=device)
        else:
            model = DnCNN(image_channels=hyperparameter['input_channels'])
        model.train()
        model.to(device)

        logger.info("Model train")

        criterion = Loss(hyperparameter['loss']).function
        optimizer = Optimizer(hyperparameter['optimizer'], model, hyperparameter['lr']).function
        scheduler = MultiStepLR(optimizer, milestones=hyperparameter['milestones'], gamma=hyperparameter['gamma'])

        train_loss_list = list()
        valid_loss_list = list()
        
        train_dataset = DenoisingDataset(train_download_uri_list, hyperparameter['sigma'])
        train_data_loader = DataLoader(dataset=train_dataset, num_workers=0, drop_last=True, batch_size=hyperparameter['batch_size'],
                                shuffle=True)
        
        valid_flag = False
        if len(validation_download_uri_list) != 0:
            valid_flag = True
            valid_dataset = DenoisingDataset(validation_download_uri_list, hyperparameter['sigma'])
            valid_data_loader = DataLoader(dataset=valid_dataset,
                                           num_workers=0,
                                           drop_last=False,
                                           batch_size=1,
                                           shuffle=hyperparameter['validation_save_random'])

        inference_info = {'inference_info' : json.dumps({"input_channels": hyperparameter['input_channels']})}
        total_iteration = len(train_data_loader)
        for epoch in range(1, hyperparameter['epoch'] + 1):
            epoch_start_time = time.time()

            train_epoch_loss = 0.0
            valid_epoch_loss = 0.0
            train_epoch_loss_mean = 0.0

            iteration_start_time = time.time()
            for n_count, batch_yx in enumerate(train_data_loader, 1):
                optimizer.zero_grad()
                batch_x = batch_yx[1].to(device)
                batch_y = batch_yx[0].to(device)

                with torch.cuda.amp.autocast(enabled=hyperparameter['using_amp']):
                    loss = criterion(model(batch_y), batch_x)
                
                train_epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                if n_count % (total_iteration // 10) == 0 or n_count == total_iteration:
                    iteration_elapsed_time = time.time() - iteration_start_time
                    logger.info(f"Epoch: [{epoch:4d}/{hyperparameter['epoch']:4d}], Iter: [{n_count:4d}/{total_iteration:4d}], Loss: {loss.item():4.4f}, Time: {iteration_elapsed_time:4.2f}s")
                    iteration_start_time = time.time()

            scheduler.step()

            train_epoch_loss_mean = train_epoch_loss / (len(train_download_uri_list))
            train_loss_list.append(train_epoch_loss_mean)
            hierarchy_root = f"epoch_{epoch}"

            if valid_flag != 0:
                with torch.no_grad():
                    model.eval()
                    for valid_count, valid_batch_yx in enumerate(valid_data_loader, 1):
                        valid_batch_x = valid_batch_yx[1].to(device)

                        with torch.cuda.amp.autocast(enabled=hyperparameter['using_amp']):
                            predict = model(valid_batch_x)
                            valid_loss = criterion(predict, valid_batch_x)

                        valid_epoch_loss += valid_loss.item()

                        if valid_count <= hyperparameter['validation_save_count']:
                            valid_result = validation_data(valid_batch_x, predict)
                            valid_image_save_uri = f"{result_uri}/{hierarchy_root}/valid_image/validation_{valid_count}.png"
                            mpp.intel64.save(valid_result, valid_image_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)
                            
                    model.train()
                    valid_loss_list.append(valid_epoch_loss / (len(valid_data_loader)))

                valid_loss_image = score_list_graph_image(hyperparameter['epoch'], valid_loss_list, valid_loss_list[0], "Valid Loss Graph", 'g')
                valid_loss_image_save_uri = f"{result_uri}/{hierarchy_root}/valid_losss_graph/valid_loss_graph.png"
                mpp.intel64.save(valid_loss_image, valid_loss_image_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)
                
                valid_loss_csv = score_list_csv(valid_loss_list)
                valid_loss_csv_save_uri = f"{result_uri}/{hierarchy_root}/valid_loss_csv/valid_loss_csv.csv"
                mpp.intel64.save_csv(valid_loss_csv, valid_loss_csv_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

            train_loss_image = score_list_graph_image(hyperparameter['epoch'], train_loss_list, train_loss_list[0], "Train Loss Graph", 'r')
            train_loss_image_save_uri = f"{result_uri}/{hierarchy_root}/train_loss_image/train_loss_image.png"
            mpp.intel64.save(train_loss_image, train_loss_image_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

            train_loss_csv = score_list_csv(train_loss_list)
            train_loss_csv_save_uri = f"{result_uri}/{hierarchy_root}/train_loss_csv/train_loss_csv.csv"
            mpp.intel64.save_csv(train_loss_csv, train_loss_csv_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

            if epoch % hyperparameter['save_epoch'] == 0 or epoch == hyperparameter['epoch']:
                model_save_uri = f"{result_uri}/{hierarchy_root}/model/model.pth"
                mpp.daq.object_service.upload_model(model, uri=model_save_uri, inference_info=inference_info, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

            epoch_valid_loss_mean = valid_epoch_loss/valid_count if valid_flag else 0
            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info(f"Epoch:[{epoch:4d}/{hyperparameter['epoch']:4d}], "
                    f"Train Loss: {train_epoch_loss/n_count:4.4f}, "
                    f"Valid Loss : {epoch_valid_loss_mean:4.4f}, "
                    f"Time: {epoch_elapsed_time:4.2f}s")

    except Exception as e:
        logger.error(f"Error Message : {e}")
        raise Exception("Train Fail")

    finally:
        shutil.rmtree(local_download_path)
        logger.info("Temp Folder Delete")

def get_dataset_sample(dataset_id, operation_channel, access_token):
    stub = protos.daq_dataset_image_dataset_api_v1_pb2_grpc.ImageDatasetServiceStub(operation_channel)
    query_parameter = protos.daq_common_pb2.QueryParameter(
        page_index=0,
        page_size=-1,
        where =StringValue(value=f"DatasetId=\"{dataset_id}\""),
        order_by=None)

    return stub.ListSamples(request=protos.daq_dataset_image_dataset_api_v1_pb2.ListSamplesRequest(
        query_parameter=query_parameter), metadata=[('authorization', f'Bearer {access_token}')])

def data_download(patch_size,
                  stride,
                  scales,
                  train_ratio,
                  convert_gray,
                  download_path_base,
                  dataset_id_list,
                  operation_channel,
                  access_token):
    
    response_image_id_list = list()
    for dataset_id in dataset_id_list.values():
        dataset_samples = get_dataset_sample(dataset_id, operation_channel, access_token)
        for sample in dataset_samples.samples:
            for response_image in sample.images:
                response_image_id_list.append(response_image.id)

    train_ratio = min(1, train_ratio)
    train_len = int(len(response_image_id_list) * train_ratio)
    train_download_uri_list = list()
    validation_download_uri_list = list()

    for response_image_id in response_image_id_list[:train_len]:
        uri = f"dataset:///?image_id={response_image_id}"
        image = mpp.daq.intel64.load(uri, convert_gray, channel=operation_channel, access_token=access_token)
        download_path = os.path.join(download_path_base, 'train', f'{response_image_id}.png')
        path_uri_list = preprocessing(image, patch_size, stride, scales, download_path)
        train_download_uri_list += path_uri_list
    
    for response_image_id in response_image_id_list[train_len:]:
        uri = f"dataset:///?image_id={response_image_id}"
        image = mpp.daq.intel64.load(uri, convert_gray, channel=operation_channel, access_token=access_token)
        download_path = os.path.join(download_path_base, 'validation', f'{response_image_id}.png')
        mpp.intel64.save(image, f"{download_path}")
        validation_download_uri_list.append(download_path)

    return train_download_uri_list, validation_download_uri_list


def preprocessing(image, patch_size, stride, scales, download_path):
    h = image.shape[0]
    w = image.shape[1]

    count = 0
    download_uri_list = list()

    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)

        img_scaled = cv2.resize(image, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        for row in range(0, h_scaled - patch_size + 1, stride):
            for col in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[row:row + patch_size, col:col + patch_size]
                x_aug = data_aug(x, mode=np.random.randint(0, 8))

                path = f"{download_path}_{count}.png"
                download_uri_list.append(path)
                mpp.intel64.save(x_aug, path)
                count += 1

    return download_uri_list


class Loss():
    def __init__(self, loss_name):
        if loss_name.lower() == "mseloss":
            self.function = nn.MSELoss(reduction='sum')

        elif loss_name.lower() == "bceloss":
            self.function = nn.BCELoss()

        elif loss_name.lower() == "crossentropyloss":
            self.function = nn.CrossEntropyLoss()

        else:
            raise Exception("Invalid Loss Option")


class Optimizer():
    def __init__(self, optimizer_name, model, lr):
        if optimizer_name.lower() == "adam":
            self.function = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name.lower() == "sgd":
            self.function = optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name.lower() == "adagrad":
            self.function = optim.Adagrad(model.parameters(), lr=lr)
        else:
            raise Exception("Invalid Optimizer Option")


class DenoisingDataset(Dataset):

    def __init__(self, download_uri_list, sigma):
        super(DenoisingDataset, self).__init__()
        self.download_uri_list = download_uri_list
        self.sigma = sigma
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.count = 0

    def __getitem__(self, index):
        uri = self.download_uri_list[index]

        normal_image = mpp.intel64.load(uri, False)
        normal_image = self.transform(normal_image)
        noise = torch.randn(normal_image.size()).mul_(self.sigma / 255.0)
        noise_image = normal_image + noise

        self.count += 1

        return noise_image, normal_image

    def __len__(self):
        return len(self.download_uri_list)


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


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
   

def validation_data(input_image, predict_image):

    input_image = input_image.detach().to('cpu')
    input_image = input_image.numpy().astype(np.float32).squeeze(0)
    input_image = input_image.transpose(1,2,0)
    min_value =  np.min(input_image)
    max_value =  np.max(input_image)
    input_image = ((input_image - min_value) / (max_value - min_value)) * 255
    input_image = input_image.astype(np.uint8)

    predict_image = predict_image.detach().to('cpu')
    predict_image = predict_image.numpy().astype(np.float32).squeeze(0)
    predict_image = predict_image.transpose(1,2,0)
    min_value =  np.min(predict_image)
    max_value =  np.max(predict_image)
    predict_image = ((predict_image - min_value) / (max_value - min_value)) * 255
    predict_image = predict_image.astype(np.uint8)
            
    row = predict_image.shape[0]
    col = predict_image.shape[1]
    
    plt.clf()
    figure = plt.figure(figsize=((col//100)*2+2, (row//100)+2))
    plt.subplot(1,2,1)
    plt.title("Input")
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

    plt.subplot(1,2,2)
    plt.title("output")
    plt.imshow(cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB))
    figure.canvas.draw()
    result = np.array(figure.canvas.renderer._renderer)

    plt.clf()
    plt.close(figure)
    
    return result


def create_bucket(channel, access_token, bucket_id, bucket_title, bucket_volume_id, properties=None, description=None):
    if properties:
        properties = StringValue(value=json.dumps(properties))
    if description:
        description = StringValue(value=description)

    stub = protos.daq_object_object_api_v1_pb2_grpc.ObjectServiceStub(channel)
    stub.CreateBucket(request=protos.daq_object_object_api_v1_pb2.CreateBucketRequest(
        id=bucket_id, title=bucket_title, properties=properties, description=description, volume_id=bucket_volume_id
    ),metadata=[('authorization', f'Bearer {access_token}')])


if __name__ == '__main__':
    kwargs = json.loads(parameters)
    kwargs['authentication']['operation_service_address'] = ""
    kwargs['authentication']['access_token'] = ""

    RecipeRun(**kwargs)
    print("Success")
