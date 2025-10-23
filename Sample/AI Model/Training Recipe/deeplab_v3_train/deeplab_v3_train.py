import os
import io
import json
import logging
import time
import shutil
from urllib.parse import urlparse
import uuid
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import jaccard_score
from google.protobuf.wrappers_pb2 import StringValue
import mpp
from mpp.daq import protos
import grpc
import network

##!--{"Name":"hyperparameter","Type":"epoch","Key":"epoch","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"batch_size","Key":"batch_size","Value":"","Category":""}
##!--{"Name":"result","Type":"result","Key":"id","Value":"","Category":""}
##!--{"Name":"authentication","Type":"system_address","Key":"operation_service_address","Value":"","Category":""}
##!--{"Name":"authentication","Type":"access_token","Key":"access_token","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id_1","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id_2","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id_3","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id_4","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id_5","Value":"","Category":""}

##$--
parameters = '''{
    "hyperparameter":{
        "epoch" : 100,
        "save_model_epoch" : 1,
        "learning_rate" : 0.01,
        "batch_size" : 2,
        "weight_decay" : 1e-4,
        "model" : "deeplabv3plus_mobilenet",
        "separable_conv" : false,       
        "output_stride" : 16,
        "train_ratio" : 0.8,
        "using_gpu" : true,
        "using_amp" : false,
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
    "gt_dataset":{
        "gt_dataset_id_1" : "",
        "gt_dataset_id_2" : "",
        "gt_dataset_id_3" : "",
        "gt_dataset_id_4" : "",
        "gt_dataset_id_5" : "",
        "gt_dataset_id_6" : "",
        "gt_dataset_id_7" : "",
        "gt_dataset_id_8" : "",
        "gt_dataset_id_9" : "",
        "gt_dataset_id_10" : ""
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

    authentication = kwargs['authentication']
    access_token = authentication['access_token']
    chunk_size = kwargs['chunk_size']

    selected_operation_address = random.choice(authentication['operation_service_address'].split(','))
    operation_channel = grpc.insecure_channel(selected_operation_address)

    hyperparameter = kwargs['hyperparameter']
    hyperparameter['epoch'] = int(hyperparameter['epoch'])
    hyperparameter['batch_size'] = int(hyperparameter['batch_size'])
    device = 'cuda' if kwargs['hyperparameter']['using_gpu'] else 'cpu'

    bucket_id = kwargs['result']['id']
    if urlparse(bucket_id).scheme == '':
        create_bucket(operation_channel, access_token, bucket_id, bucket_id+"_title", kwargs['result']['volume_id'])
        result_uri = f"object:///{bucket_id}"
    else:
        result_uri = bucket_id

    segmentation_gts_list = list()
    label_info = set()
    for gt_data_set_id in kwargs['gt_dataset'].values():
        if gt_data_set_id is None or gt_data_set_id == '':
            continue
        
        segmentation_gts = get_segmentation_gts(operation_channel, access_token, gt_data_set_id)
        segmentation_gts_list.extend(segmentation_gts)
        segmentation_gt_dataset = get_segmentation_gt_dataset(operation_channel, access_token, gt_data_set_id)
        label_info = label_info.union(load_label_info(operation_channel, access_token, segmentation_gt_dataset.class_code_set_id))
    label_info = sorted(label_info)

    logger.info("Temp Folder Create")
    local_download_path = os.path.join(f"/temp/{uuid.uuid4()}")
    train_path_list, valid_path_list = download_image(operation_channel,  
                                                    access_token,
                                                    segmentation_gts_list,
                                                    local_download_path,
                                                    hyperparameter['train_ratio'])
    try:
        train_dataset = SegmentationDataset(data_path_list=train_path_list, label_info=label_info)
        train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=hyperparameter['batch_size'],
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=0)
        total_iteration = len(train_loader)

        valid_flag = False
        if len(valid_path_list) != 0:
            valid_flag = True
            valid_dataset = SegmentationDataset(data_path_list=valid_path_list, label_info=label_info)
            valid_loader = data.DataLoader(dataset=valid_dataset,
                                        batch_size=1,
                                        shuffle=hyperparameter['validation_save_random'],
                                        drop_last=False,
                                        num_workers=0)
                                        
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        pretrained_model_path = os.path.join(current_file_path, "pretrained_model.pth")
        if hyperparameter['pretrained'] and os.path.exists(pretrained_model_path):
            logger.info("Pretrained")
            model = torch.jit.load(pretrained_model_path, map_location=device)
        else:
            model = network.modeling.__dict__[hyperparameter['model']](num_classes=len(label_info), output_stride=hyperparameter['output_stride']).to(device)
        model.train()

        if hyperparameter['separable_conv'] and 'plus' in hyperparameter['model']:
            network.convert_to_separable_conv(model.classifier)
        set_bn_momentum(model.backbone, momentum=0.01)

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * hyperparameter['learning_rate']},
            {'params': model.classifier.parameters(), 'lr': hyperparameter['learning_rate']},
        ], lr=hyperparameter['learning_rate'], momentum=0.9, weight_decay=hyperparameter['weight_decay'])
        scheduler = PolyLR(optimizer, hyperparameter['epoch']*total_iteration, power=0.9)

        train_loss_list = list()
        valid_loss_list = list()
        valid_miou_list = list()
        inference_info = {'inference_info' : json.dumps({"label_info": label_info})}

        for epoch in range(1, hyperparameter['epoch']):
            epoch_start_time = time.time()
            train_epoch_loss = 0.0
            valid_epoch_loss = 0.0
            valid_epoch_miou = 0.0

            iteration_start_time = time.time()
            for n_count, (images, labels) in enumerate(train_loader, 1):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=hyperparameter['using_amp']):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()

                if n_count % (total_iteration // 10) == 0 or n_count == total_iteration:
                    iteration_elapsed_time = time.time() - iteration_start_time
                    logger.info(f"Epoch: [{epoch:4d}/{hyperparameter['epoch']:4d}], Iter: [{n_count:4d}/{total_iteration:4d}], Loss: {loss.item():4.4f}, Time: {iteration_elapsed_time:4.2f}s")
                    iteration_start_time = time.time()
                scheduler.step()

            train_loss_list.append(train_epoch_loss/total_iteration)
            hierarchy_root = f"epoch_{epoch}"

            if valid_flag:
                with torch.no_grad():
                    model.eval()
                    for valid_n_count, (valid_images, valid_labels) in enumerate(valid_loader):
                        valid_images = valid_images.to(device)
                        valid_labels = valid_labels.to(device)

                        with torch.cuda.amp.autocast(enabled=hyperparameter['using_amp']):
                            valid_outputs = model(valid_images)
                            valid_loss = criterion(valid_outputs, valid_labels)

                        valid_epoch_loss += valid_loss.item()

                        if valid_n_count <= hyperparameter['validation_save_count']:
                            valid_miou = iou_score(valid_outputs, valid_labels, len(label_info))
                            valid_epoch_miou+=valid_miou
                            valid_result = validation_data(valid_images, valid_outputs)
                            valid_image_save_uri = f"{result_uri}/{hierarchy_root}/valid_image/validation_{valid_n_count}.png"
                            mpp.intel64.save(valid_result, valid_image_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

                    model.train()
                    
                valid_loss_list.append(valid_epoch_loss/len(valid_loader))
                valid_miou_list.append(valid_epoch_miou/min(hyperparameter['validation_save_count'], valid_n_count))
            
                valid_loss_image = score_list_graph_image(hyperparameter['epoch'], valid_loss_list, valid_loss_list[0], "Valid Loss Graph", 'g')
                valid_loss_image_save_uri = f"{result_uri}/{hierarchy_root}/valid_losss_graph/valid_loss_graph.png"
                mpp.intel64.save(valid_loss_image, valid_loss_image_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

                valid_loss_csv = score_list_csv(valid_loss_list)
                valid_loss_csv_save_uri = f"{result_uri}/{hierarchy_root}/valid_loss_csv/valid_loss_csv.csv"
                mpp.intel64.save_csv(valid_loss_csv, valid_loss_csv_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

                valid_miou_image = score_list_graph_image(hyperparameter['epoch'], valid_miou_list, 1, "Valid mIoU Graph", 'r')
                valid_miou_image_save_uri = f"{result_uri}/{hierarchy_root}/valid_miou/valid_miou.png"
                mpp.intel64.save(valid_miou_image, valid_miou_image_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

                valid_miou_csv = score_list_csv(valid_miou_list, header="mIou")
                valid_miou_csv_save_uri = f"{result_uri}/{hierarchy_root}/valid_miou_csv/valid_miou_csv.csv"
                mpp.intel64.save_csv(valid_miou_csv, valid_miou_csv_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)


            train_loss_image = score_list_graph_image(hyperparameter['epoch'], train_loss_list, train_loss_list[0], "Train Loss Graph", 'r')
            train_loss_image_save_uri = f"{result_uri}/{hierarchy_root}/train_loss_image/train_loss_image.png"
            mpp.intel64.save(train_loss_image, train_loss_image_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

            train_loss_csv = score_list_csv(train_loss_list)
            train_loss_csv_save_uri = f"{result_uri}/{hierarchy_root}/train_loss_csv/train_loss_csv.csv"
            mpp.intel64.save_csv(train_loss_csv, train_loss_csv_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

            if epoch % hyperparameter['save_model_epoch'] == 0 or epoch == hyperparameter['epoch']:
                model_save_uri = f"{result_uri}/{hierarchy_root}/model/model.pth"
                mpp.daq.object_service.upload_model(model, uri=model_save_uri, inference_info=inference_info, example=images, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)

            epoch_valid_loss_mean = valid_epoch_loss/valid_n_count if valid_flag else 0
            epoch_valid_miou_mean = valid_epoch_miou/hyperparameter['validation_save_count'] if valid_flag else 0
            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info(f"Epoch:[{epoch:4d}/{hyperparameter['epoch']:4d}], "
                    f"Train Loss: {train_epoch_loss/n_count:4.4f}, "
                    f"Valid Loss : {epoch_valid_loss_mean:4.4f}, "
                    f"Valid mIou : {epoch_valid_miou_mean:4.4f}, "
                    f"Time: {epoch_elapsed_time:4.2f}s")

    except Exception as e:
        logger.error(f"Error Message : {e}")
        raise Exception("Train Fail")

    finally:
        shutil.rmtree(local_download_path)
        logger.info("Temp Folder Delete")


def load_label_info(channel, access_token, dataset_id):
    stub = protos.daq_dataset_class_code_api_v1_pb2_grpc.ClassCodeServiceStub(channel)
    query_parameter = protos.daq_common_pb2.QueryParameter(
        page_index=1,
        page_size=-1,
        where=StringValue(value=f"Id=\"{dataset_id}\""),
        order_by=None)
    response = stub.ListClassCodeSets(request=protos.daq_dataset_class_code_api_v1_pb2.ListClassCodeSetsRequest(
        query_parameter=query_parameter), metadata=[('authorization', f'Bearer {access_token}')])

    label_info = set()
    class_codes = response.class_code_sets[0].class_codes
    for class_code in class_codes:
        label = json.loads(class_code.label_info.value)['label']
        label_info.add(int(label))
    return label_info

def get_segmentation_gt_dataset(channel, access_token, gt_dataset_id):
    stub = protos.daq_dataset_segmentation_gt_dataset_api_v1_pb2_grpc.SegmentationGtDatasetServiceStub(channel)

    segemntation_gt_dataset = stub.GetSegmentationGtDataset(request=protos.daq_dataset_segmentation_gt_dataset_api_v1_pb2.GetSegmentationGtDatasetRequest(
        id=gt_dataset_id), metadata=[('authorization', f'Bearer {access_token}')])

    return segemntation_gt_dataset

def get_segmentation_gts(channel, access_token, gt_dataset_id):
    stub = protos.daq_dataset_segmentation_gt_dataset_api_v1_pb2_grpc.SegmentationGtDatasetServiceStub(channel)
    query_parameter = protos.daq_common_pb2.QueryParameter(
        page_index= 1,
        page_size= -1,
        where=StringValue(value=f"GtDatasetId=\"{gt_dataset_id}\" AND DataStatus=\"success\""),
        order_by=None)

    response = stub.ListSegmentationGts(request=protos.daq_dataset_segmentation_gt_dataset_api_v1_pb2.ListSegmentationGtsRequest(
        query_parameter=query_parameter, with_image=False), metadata=[('authorization', f'Bearer {access_token}')])

    return response.segmentation_gts


def validation_data(input_tensor, predict_tensor, palette = [0,0,0,255,255,255]):
    input_image = input_tensor.cpu().data.numpy().astype(np.float32)
    input_image = input_image.squeeze(0)
    input_image = input_image.transpose(1,2,0)

    min_value =  np.min(input_image)
    max_value =  np.max(input_image)
    input_image = ((input_image - min_value) / (max_value - min_value)) * 255
    input_image = input_image.astype(np.uint8)

    predict_image = torch.argmax(predict_tensor, 1)
    predict_image = predict_image.cpu().data.numpy()
    predict_image = predict_image.squeeze(0)
    predict_image = Image.fromarray(predict_image.astype(np.uint8))
    predict_image.putpalette(palette)

    row = input_image.shape[0]
    col = input_image.shape[1]

    plt.clf()
    figure = plt.figure(figsize=((col//100)*2+2, (row//100)+2))
    plt.subplot(1,2,1)
    plt.title("Input")
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

    plt.subplot(1,2,2)
    plt.title("Output")
    plt.imshow(predict_image)
    figure.canvas.draw()
    result = np.array(figure.canvas.renderer._renderer)

    plt.clf()
    plt.close(figure)

    return result


def score_list_csv(score_list, header='Loss'):
    result = [['Epoch', header]] + [[epoch, score] for epoch, score in enumerate(score_list, 1)]
    return result


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

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]


def download_image(operation_channel, access_token, segmentation_gts, download_path, train_ratio):
    train_ratio = min(1, train_ratio)
    train_len = int(len(segmentation_gts) * train_ratio)

    train_image_path_list = list()
    train_gt_path_list = list()
    valid_image_path_list = list()
    valid_gt_path_list = list()

    for train_index in range(train_len):
        image_id = segmentation_gts[train_index].image_id
        gt_id = segmentation_gts[train_index].id

        if segmentation_gts[train_index].data_status != 'success':
            continue

        image_uri = f"dataset:///?image_id={image_id}"
        gt_uri = f"dataset:///?image_id={gt_id}"

        image = mpp.daq.intel64.load(image_uri, False, channel=operation_channel, access_token=access_token)

        ground_truth = mpp.daq.dataset_service.load_segmentation_gt(gt_uri, True, channel=operation_channel, access_token=access_token)

        image_path = os.path.join(download_path, "train", "image", f"{train_index}.png")
        gt_path = os.path.join(download_path, "train", "ground_truth", f"{train_index}.png")

        mpp.intel64.save(image, image_path)
        mpp.intel64.save(ground_truth, gt_path)

        train_image_path_list.append(image_path)
        train_gt_path_list.append(gt_path)

    for valid_index in range(train_len, len(segmentation_gts)):
        image_id = segmentation_gts[valid_index].image_id
        gt_id = segmentation_gts[valid_index].id

        if segmentation_gts[valid_index].data_status != 'success':
            continue

        image_uri = f"dataset:///?image_id={image_id}"
        gt_uri = f"dataset:///?image_id={gt_id}"

        image = mpp.daq.intel64.load(image_uri, False, channel=operation_channel, access_token=access_token)
        ground_truth = mpp.daq.dataset_service.load_segmentation_gt(gt_uri, True, channel=operation_channel, access_token=access_token)

        image_path = os.path.join(download_path, "validation", "image", f"{valid_index}.png")
        gt_path = os.path.join(download_path, "validation", "ground_truth", f"{valid_index}.png")

        mpp.intel64.save(image, image_path)
        mpp.intel64.save(ground_truth, gt_path)

        valid_image_path_list.append(image_path)
        valid_gt_path_list.append(gt_path)

    return (train_image_path_list, train_gt_path_list), (valid_image_path_list, valid_gt_path_list)


class SegmentationDataset(data.Dataset):
    def __init__(self, data_path_list, label_info):
        super(SegmentationDataset, self).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_path_list = data_path_list[0]
        self.mask_path_list = data_path_list[1]
        assert (len(self.image_path_list) == len(self.mask_path_list))

        self.label_info = label_info
        self._key= np.full(255+2, -1)

        for i, label in enumerate(label_info):
            self._key[label+1] = i

        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def __getitem__(self, index):
        img = cv2.imread(self.image_path_list[index], 1)
        mask = cv2.imread(self.mask_path_list[index], 0)
        img = self.transform(img)
        mask = self.__mask_transform(mask)
        logger.info(f"__getitem__ img:{img.shape=} gt:{mask.shape=}")
        return img, mask

    def __mask_transform(self, mask):
        mask = np.array(mask).astype('int32')
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)

        target = self._key[index].reshape(mask.shape)
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.image_path_list)


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def iou_score(pred, target, label_nums):
    pred = torch.argmax(pred, 1)
    pred = pred.cpu().data.numpy().reshape(-1)
    target = target.cpu().data.numpy().reshape(-1)
    score = jaccard_score(y_true=target, y_pred=pred, labels=[x for x in range(0, label_nums)], average='macro')

    return score 


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
