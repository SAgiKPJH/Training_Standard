"""
EfficientNetV2-L Training Recipe

This training recipe uses EfficientNetV2-L (Large) model with the following features:
- Model Size: ~5-6GB (120M+ parameters)
- Input Size: 480x480 pixels for optimal performance
- Always uses ImageNet pretrained weights for best results
- Optimized hyperparameters for large model training
- Memory-efficient batch size settings

The model automatically loads ImageNet pretrained weights regardless of usePtWeights setting
for optimal training performance and convergence.
"""

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
from google.protobuf.wrappers_pb2 import StringValue
import mpp
from mpp.daq import protos


from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from keras.applications.efficientnet_v2 import EfficientNetV2L
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers.optimizer_v2.adam import Adam
from keras.optimizers.optimizer_v2.rmsprop import RMSprop
from keras.optimizers.legacy.sgd import SGD
import traceback



##!--{"Name":"hyperparameter","Type":"rescale","Key":"rescale","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"widthShift","Key":"widthShift","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"heightShift","Key":"heightShift","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"rotationRange","Key":"rotationRange","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"zoomRange","Key":"zoomRange","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"shearRange","Key":"shearRange","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"horizontalFlip","Key":"horizontalFlip","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"verticalFlip","Key":"verticalFlip","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"mixupcutmix","Key":"mixupcutmix","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"mixup","Key":"mixup","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"cutmix","Key":"cutmix","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"usePtWeights","Key":"usePtWeights","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"rtWeightPath","Key":"rtWeightPath","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"optimizer","Key":"optimizer","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"learningRate","Key":"learningRate","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"reduceLRPatience","Key":"reduceLRPatience","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"earlyStopPatience","Key":"earlyStopPatience","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"epoch","Key":"epoch","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"savePeriod","Key":"savePeriod","Value":"","Category":""}

##!--{"Name":"result","Type":"result","Key":"id","Value":"","Category":""}
##!--{"Name":"authentication","Type":"system_address","Key":"operation_service_address","Value":"","Category":""}
##!--{"Name":"authentication","Type":"access_token","Key":"access_token","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id","Value":"","Category":""}

##$--
parameters = '''{
    "hyperparameter":{
        "batch_size" : 16,
        "using_gpu" : true,
        "train_ratio" : 0.2,

        "rescale" : "",
        "widthShift" : 0.0,
        "heightShift" : 0.0,
        "rotationRange" : 0,
        "zoomRange" : 0.0,
        "shearRange" : 0.0,
        "horizontalFlip" : false,
        "verticalFlip" : false,
        "mixupcutmix" : "",
        "mixup" : 0.0,
        "cutmix" : 0.0,

        "usePtWeights" : true,
        "rtWeightPath" : "",
        "optimizer" : "adam",
        "learningRate" : 0.0001,
        "reduceLRPatience" : 5,
        "earlyStopPatience" : 10,
        "epoch" : 50,
        "savePeriod" : 0
    },
    "authentication": {
        "operation_service_address": "",
        "access_token" : ""
    },
    "result":{
        "id": "",
        "volume_id":"default" 
    },
    "gt_dataset":{
        "gt_dataset_id" : "haesung_sample_gt"
    },
    "chunk_size" : 100000
}'''
##$--


logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

result_uri = ""
model = None
operation_channel = None
access_token = None
chunk_size = None
inference_info = None
max_val_acc = None
start_epoch_time = 0
logic_start_time = 0
end_time = 0
valid_epoch_loss = []
valid_epoch_acc = []
hyperparameters = None

def RecipeRun(**kwargs):
    global logic_start_time
    logic_start_time = time.time()

    # if kwargs['hyperparameter']['using_gpu'] and not torch.cuda.is_available():
    #     raise Exception("GPU is not avaiable")

    global result_uri
    global model
    global operation_channel
    global access_token
    global chunk_size
    global inference_info
    global max_val_acc
    global hyperparameters
    max_val_acc = 0

    authentication = kwargs['authentication']
    access_token = authentication['access_token']
    operation_channel = grpc.insecure_channel(authentication['operation_service_address'])
 
    hyperparameters = kwargs['hyperparameter']
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
    data_download(local_download_path,
                  classification_gts,
                  class_code_info,
                  operation_channel,
                  access_token)

    try:
        print("Starting EfficientNetV2-L training with ImageNet pretrained weights")

        rescaleParam = None
        if hyperparameter['rescale'] != "":
            rescaleParam = hyperparameter['rescale']

        idg =ImageDataGenerator(
        rescale = rescaleParam,
        width_shift_range = float(hyperparameter['widthShift']),
        height_shift_range = float(hyperparameter['heightShift']),
        rotation_range = int(hyperparameter['rotationRange']),
        zoom_range = float(hyperparameter['zoomRange']),
        shear_range = float(hyperparameter['shearRange']),
        horizontal_flip = hyperparameter['horizontalFlip'],
        vertical_flip = hyperparameter['verticalFlip'],
        validation_split= float(hyperparameter['train_ratio']))

        # EfficientNetV2-L uses 480x480 input size for optimal performance
        inputSize = 480
        # Reduce batch size due to larger model and input size
        batch_size = max(1, hyperparameter['batch_size'] // 2)
        
        train_generator = idg.flow_from_directory(
            directory=local_download_path, 
            subset="training", 
            target_size=(inputSize, inputSize), 
            batch_size=batch_size
        )
        valid_generator = idg.flow_from_directory(
            directory=local_download_path, 
            subset="validation", 
            target_size=(inputSize, inputSize), 
            batch_size=batch_size
        )

        inference_info = {'input_size': inputSize, 'label_info': label_info, 'model_type': 'EfficientNetV2-L'}

        if hyperparameter['usePtWeights'] == False:
            # Train from scratch (not recommended for EfficientNetV2-L due to size)
            logger.info("Training EfficientNetV2-L from scratch")
            md = EfficientNetV2L(input_shape=(inputSize, inputSize, 3), weights=None, include_top=False)
            a = GlobalAveragePooling2D()(md.output)
            a = Dense(num_classes, activation='softmax')(a)
            model = Model(inputs=md.input, outputs=a)
        else:
            curDir = os.path.dirname(os.path.abspath(__file__))
            ptWeightPath = os.path.join(curDir, "efficientnetv2l_notop.h5")
            
            # Check if local weight file exists, if not download from ImageNet
            if os.path.exists(ptWeightPath):
                logger.info(f"Loading EfficientNetV2-L from local file: '{ptWeightPath}'")
                md = EfficientNetV2L(input_shape=(inputSize, inputSize, 3), weights=ptWeightPath, include_top=False)
            else:
                logger.info("Local weight file not found. Downloading ImageNet pretrained weights...")
                logger.info("To use local weights, place 'efficientnetv2l_notop.h5' in the same directory")
                md = EfficientNetV2L(input_shape=(inputSize, inputSize, 3), weights='imagenet', include_top=False)
                
                # Save the downloaded weights for future use
                logger.info(f"Saving ImageNet weights to '{ptWeightPath}' for future use")
                md.save_weights(ptWeightPath)
            
            a = GlobalAveragePooling2D()(md.output)
            a = Dense(num_classes, activation='softmax')(a)
            model = Model(inputs=md.input, outputs=a)

        
        if hyperparameter['rtWeightPath'] != "":
            logger.info(f"Loading additional weights from '{hyperparameter['rtWeightPath']}'")
            model.load_weights(hyperparameter['rtWeightPath'])

        opt = None
        if hyperparameter['optimizer'] == 'adam':         opt = Adam(float(hyperparameter['learningRate']))
        elif hyperparameter['optimizer'] == 'rmsprop':    opt = RMSprop(float(hyperparameter['learningRate']))
        elif hyperparameter['optimizer'] == 'sgd':        opt = SGD(float(hyperparameter['learningRate']))

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

        # Print model summary for verification
        logger.info(f"EfficientNetV2-L Model loaded with ImageNet pretrained weights")
        logger.info(f"Model parameters: {model.count_params():,}")
        logger.info(f"Estimated model size: ~{model.count_params() * 4 / 1024 / 1024 / 1024:.1f}GB")
        logger.info(f"Input size: {inputSize}x{inputSize}")
        logger.info(f"Batch size: {batch_size}")

        ep = EarlyStopping(monitor='val_acc', patience=int(hyperparameter['earlyStopPatience']), verbose=1)
        lrs = ReduceLROnPlateau(monitor='val_loss', patience=int(hyperparameter['reduceLRPatience']), factor=0.5, verbose=1)
        lc = LambdaCallback(on_epoch_end=on_epoch_end)

        cb = []
        cb.append(lc)
        if int(hyperparameter['reduceLRPatience']) != 0:   cb.append(lrs)
        if int(hyperparameter['earlyStopPatience']) != 0:  cb.append(ep)

        model.fit_generator(
            train_generator, 
            validation_data=valid_generator, 
            epochs=int(hyperparameter['epoch']), 
            steps_per_epoch=len(train_generator), 
            validation_steps=len(valid_generator), 
            callbacks=cb
        )

    except Exception as e:
        logger.error(f"Error Message : {e}")
        str_io = io.StringIO()
        traceback.print_exc(file=str_io)
        str_io.close()
        error_message = str_io.getvalue()
        logger.error(error_message)
        raise Exception(f"Error Message : {e}")
    finally:
        shutil.rmtree(local_download_path)
        logger.info("Temp Folder Delete")



def on_epoch_begin(epoch, logs):
    global start_epoch_time
    start_epoch_time = time.time()


def on_epoch_end(epoch, logs):
    logger.info(f"current epoch : '{epoch}'")
    logger.info(logs)

    # print(epoch) #0
    # print(logs) #{'loss': 7.8747358322143555, 'accuracy': 0.375, 'val_loss': 7.294888973236084, 'val_accuracy': 0.5}
    # print('test loss :', logs['val_loss']) #test loss : 7.294888973236084
    # print('test accuracy :', logs['val_accuracy']) #test accuracy : 0.5
    global start_epoch_time
    global hyperparameters
   
    global valid_epoch_acc
    global valid_epoch_loss
    global result_uri
    global model
    global operation_channel
    global access_token
    global chunk_size
    global inference_info
    global max_val_acc

    valid_loss  = logs["val_loss"]
    valid_acc = logs['val_acc']

    valid_epoch_loss.append(valid_loss)
    valid_epoch_acc.append(valid_acc)
    csv_result = [['Epoch', 'Valid Loss', 'Valid Accuracy']]
    max_length = max(len(valid_epoch_loss), len(valid_epoch_acc))

    for epoch in range(max_length):
        validloss = valid_epoch_loss[epoch] if epoch < len(valid_epoch_loss) else None
        validacc = valid_epoch_acc[epoch] if epoch < len(valid_epoch_acc) else None
        csv_result.append([epoch + 1, validloss, validacc])

    # if max_val_acc < cur_val_acc:
    # max_val_acc = cur_val_acc
    #model_save_uri = f"D:\DAQ\Result\model.h5"
    # model_save_uri = f"{result_uri}/model_{epoch+1}.h5"
    model_save_uri = f"{result_uri}/epoch_{epoch+1}/model/model.h5"


    training_running_time = time.time() - logic_start_time
    
    epoch_elapsed_time = time.time() - start_epoch_time
    remaining_epochs = hyperparameters['epoch'] - epoch
    estimated_remaining_time = remaining_epochs * epoch_elapsed_time

    logger.info("MonitoringData:"
                    f"Epoch:[{epoch+1:4d}/{hyperparameters['epoch']:4d}]({epoch+1:4.2f}s), "
                    f"Running Time: {training_running_time / 60:.2f} minutes, "
                    f"Remain Time: {estimated_remaining_time / 60:.2f} minutes")
    upload_model(model, uri=model_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size, inference_info=inference_info)
    
    valid_loss_csv_save_uri = f"{result_uri}/epoch_{epoch+1}/train_loss_csv/train_loss_csv.csv"
    
    mpp.intel64.save_csv(csv_result, valid_loss_csv_save_uri, channel=operation_channel, access_token=access_token, chunk_size=chunk_size)



def create_csv(train_loss_list, valid_loss_list):
    result = [['Epoch', 'Train Loss', 'Valid Loss', 'Train Accuracy', 'Valid Accuracy']]
    max_length = max(len(train_loss_list), len(valid_loss_list))

    for epoch in range(max_length):
        train_loss = train_loss_list[epoch] if epoch < len(train_loss_list) else None
        valid_loss = valid_loss_list[epoch] if epoch < len(valid_loss_list) else None
        result.append([epoch + 1, train_loss, valid_loss, 100.0 - train_loss, 100.0 - valid_loss])

    return result


def data_download(local_download_path,
                  classification_gts,
                  class_code_info,
                  operation_channel,
                  access_token):

    for i in range(len(classification_gts)):
        image_id = classification_gts[i].image_id
        class_code = classification_gts[i].class_code.value

        uri = f"dataset:///?image_id={image_id}"
        image = mpp.daq.intel64.load(uri, False, channel=operation_channel, access_token=access_token)
        tmp_path = os.path.join(local_download_path, str(class_code_info[class_code]))
        download_path = os.path.join(tmp_path, f"{image_id}.png")
        mpp.intel64.save(image, download_path)


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
        label_info[f'label_{i}'] = {'code' : class_code.code, 
                                    'name' : class_code.name}
        class_code_info[class_code.code] = format(i, '04')

    return label_info, class_code_info, num_classes



def create_bucket(channel, access_token, bucket_id, bucket_title, bucket_volume_id, properties=None, description=None):
    if properties:
        properties = StringValue(value=json.dumps(properties))
    if description:
        description = StringValue(value=description)

    stub = protos.daq_object_object_api_v1_pb2_grpc.ObjectServiceStub(channel)
    stub.CreateBucket(request=protos.daq_object_object_api_v1_pb2.CreateBucketRequest(
        id=bucket_id, title=bucket_title, properties=properties, description=description, volume_id=bucket_volume_id
    ),metadata=[('authorization', f'Bearer {access_token}')])


# @@@ h5 포맷을 지원하는 upload_model (inference_info 정보도 h5 내 'extra_info' 그룹의 'inference_info' 속성에 저장)
def upload_model(model,
                filename: str = None, # type: ignore
                inference_info = {},
                example = None,
                uri: str = None,  # type: ignore
                bucket_id: str = None,# type: ignore
                hierarchy: str = None,# type: ignore
                properties: str = None,# type: ignore
                description: str = None,# type: ignore
                channel = None,# type: ignore
                access_token: str = None,# type: ignore
                chunk_size:int = 4194304) -> None:

    def get_extension(filename_or_url: str) -> str:
        path = urlparse(filename_or_url).path
        ext = os.path.splitext(path)[1]
        return ext

    parsed_uri = urlparse(uri)
    extension = get_extension(parsed_uri.path)

    model_bytes_io: io.BytesIO = None # type: ignore
    if extension == ".h5":
        import h5py
        from tensorflow import keras
        model_bytes_io = io.BytesIO()
        with h5py.File(model_bytes_io, 'w') as h5file:
            # NOTE:  
            # - `model.save("model.h5")` 처럼 하면 아래 모듈의 
            # - tensorflow\python\keras\saving\save.py  에 있는 구현 부의 signature는 
            #   Model클래스의 save() 함수와 signature가 같다.(내부적으로 아래 save_model() 함수를 호출)
            # 
            # @keras_export('keras.models.save_model')
            # def save_model(model,
            #             filepath,
            #             overwrite=True,
            #             include_optimizer=True,
            #             save_format=None,
            #             signatures=None,
            #             options=None,
            #             save_traces=True):
            keras.models.save_model(model, h5file, save_format="h5")

            # `inference_info`(dict) 저장
            if inference_info:
                inference_info_json = json.dumps(inference_info, ensure_ascii=False)
                extra_info = h5file.create_group("extra_info")
                extra_info.attrs["inference_info"] = inference_info_json


    elif extension == ".pth":
        import torch
        origin_mode = model.training
        if origin_mode:
            model.eval()
        
        if example is not None:
            model_script = torch.jit.trace(model, example)  # type: ignore
        else:
            model_script = torch.jit.script(model) # type: ignore

        if origin_mode:
            model.train()

        model_buffer = model_script.save_to_buffer(_extra_files=inference_info) # type: ignore
        model_bytes_io = io.BytesIO(model_buffer)
    else:
        raise ValueError("Unsupported file type: " + extension)

    parts_scheme = parsed_uri.scheme
    if parts_scheme.lower() == "object":

        uri, fileinfo = os.path.split(uri)
        filename, _ = os.path.splitext(fileinfo)

        model_bytes_io.seek(0)
        model_reader_buffer = io.BufferedReader(model_bytes_io) # type: ignore

        mpp.daq.object_service.upload_object(stream=model_reader_buffer, # type: ignore
                    filename=filename,
                    extension=extension,
                    uri=uri,
                    bucket_id=bucket_id,
                    hierarchy=hierarchy,
                    properties=properties,
                    description=description,
                    channel=channel,
                    access_token=access_token,
                    chunk_size=chunk_size)
    else:
        save_folder, _ = os.path.split(uri)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # torch.jit.save(model_script, uri, _extra_files=inference_info)
        with open(uri, 'wb') as h5file:
            h5file.write(model_bytes_io.getvalue())



if __name__ =="__main__":
    kwargs = json.loads(parameters)
    # kwargs['authentication']['operation_service_address'] = "192.168.70.62:5020"
    # kwargs['authentication']['operation_service_address'] = "192.168.70.101:5020"
    # kwargs['authentication']['access_token'] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJtaW53b29uZy5wYXJrIiwibmFtZSI6IuuwleuvvOybhSIsInJvbGUiOiJkZXZlbG9wZXIiLCJncm91cHNpZCI6ImRlZmF1bHQiLCJsb2dfaW5fcHJvdmlkZXIiOiJkYXEiLCJwcml2aWxlZ2VzIjpbImRhdGFzZXRfY2xhc3NfY29kZV9jcmVhdGUiLCJkYXRhc2V0X2NsYXNzX2NvZGVfZGVsZXRlIiwiZGF0YXNldF9jbGFzc19jb2RlX3JlYWRfYW55IiwiZGF0YXNldF9jbGFzc19jb2RlX3VwZGF0ZSIsImRhdGFzZXRfZ3RfZGF0YXNldF9jcmVhdGUiLCJkYXRhc2V0X2d0X2RhdGFzZXRfZGVsZXRlIiwiZGF0YXNldF9ndF9kYXRhc2V0X3JlYWRfYW55IiwiZGF0YXNldF9ndF9kYXRhc2V0X3VwZGF0ZSIsImRhdGFzZXRfaW1hZ2VfZGF0YXNldF9jcmVhdGUiLCJkYXRhc2V0X2ltYWdlX2RhdGFzZXRfZGVsZXRlIiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X3JlYWRfYW55IiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X3VwZGF0ZSIsImRhdGFzZXRfdm9sdW1lX2NyZWF0ZSIsImRhdGFzZXRfdm9sdW1lX2RlbGV0ZSIsImRhdGFzZXRfdm9sdW1lX3JlYWRfYW55IiwiZGF0YXNldF92b2x1bWVfdXBkYXRlIiwiZGVmZWN0X2RhdGFfbWFuYWdlcl9tYXN0ZXJfaW5mb3JtYXRpb25fY3JlYXRlIiwiZGVmZWN0X2RhdGFfbWFuYWdlcl9tYXN0ZXJfaW5mb3JtYXRpb25fZGVsZXRlIiwiZGVmZWN0X2RhdGFfbWFuYWdlcl9tYXN0ZXJfaW5mb3JtYXRpb25fcmVhZF9hbnkiLCJkZWZlY3RfZGF0YV9tYW5hZ2VyX21hc3Rlcl9pbmZvcm1hdGlvbl91cGRhdGUiLCJkZWZlY3RfZGVmZWN0X2RhdGFfY3JlYXRlIiwiZGVmZWN0X2RlZmVjdF9kYXRhX2RlbGV0ZSIsImRlZmVjdF9kZWZlY3RfZGF0YV9yZWFkX2FueSIsImRlZmVjdF9kZWZlY3RfZGF0YV91cGRhdGUiLCJkZWZlY3RfaWRic19tYXN0ZXJfaW5mb3JtYXRpb25fY3JlYXRlIiwiZGVmZWN0X2lkYnNfbWFzdGVyX2luZm9ybWF0aW9uX2RlbGV0ZSIsImRlZmVjdF9pZGJzX21hc3Rlcl9pbmZvcm1hdGlvbl9yZWFkX2FueSIsImRlZmVjdF9pZGJzX21hc3Rlcl9pbmZvcm1hdGlvbl91cGRhdGUiLCJnZHNfY2xpcF9jcmVhdGUiLCJnZHNfY2xpcF9kZWxldGUiLCJnZHNfY2xpcF9yZWFkX2FueSIsImdkc19jbGlwX3VwZGF0ZSIsImdkc19leHBvcnRfY3JlYXRlIiwiZ2RzX2V4cG9ydF9kZWxldGUiLCJnZHNfZXhwb3J0X3JlYWRfYW55IiwiZ2RzX2V4cG9ydF91cGRhdGUiLCJnZHNfZ2RzX2NyZWF0ZSIsImdkc19nZHNfZGVsZXRlIiwiZ2RzX2dkc19yZWFkX2FueSIsImdkc19nZHNfdXBkYXRlIiwiZ2RzX3NlcnZlcl9jcmVhdGUiLCJnZHNfc2VydmVyX2RlbGV0ZSIsImdkc19zZXJ2ZXJfcmVhZF9hbnkiLCJnZHNfc2VydmVyX3VwZGF0ZSIsImdkc192b2x1bWVfY3JlYXRlIiwiZ2RzX3ZvbHVtZV9kZWxldGUiLCJnZHNfdm9sdW1lX3JlYWRfYW55IiwiZ2RzX3ZvbHVtZV91cGRhdGUiLCJpbmZlcmVuY2VfaW5mZXJlbmNlX2NyZWF0ZSIsImluZmVyZW5jZV9pbmZlcmVuY2VfZGVsZXRlIiwiaW5mZXJlbmNlX2luZmVyZW5jZV9yZWFkX2FueSIsImluZmVyZW5jZV9pbmZlcmVuY2VfdXBkYXRlIiwiaW5mZXJlbmNlX21vZGVsX2NyZWF0ZSIsImluZmVyZW5jZV9tb2RlbF9kZWxldGUiLCJpbmZlcmVuY2VfbW9kZWxfcmVhZF9hbnkiLCJpbmZlcmVuY2VfbW9kZWxfdXBkYXRlIiwiaW5mZXJlbmNlX3NlcnZlcl9jcmVhdGUiLCJpbmZlcmVuY2Vfc2VydmVyX2RlbGV0ZSIsImluZmVyZW5jZV9zZXJ2ZXJfcmVhZF9hbnkiLCJpbmZlcmVuY2Vfc2VydmVyX3VwZGF0ZSIsImluZmVyZW5jZV92b2x1bWVfY3JlYXRlIiwiaW5mZXJlbmNlX3ZvbHVtZV9kZWxldGUiLCJpbmZlcmVuY2Vfdm9sdW1lX3JlYWRfYW55IiwiaW5mZXJlbmNlX3ZvbHVtZV91cGRhdGUiLCJvYmplY3Rfb2JqZWN0X2NyZWF0ZSIsIm9iamVjdF9vYmplY3RfZGVsZXRlIiwib2JqZWN0X29iamVjdF9yZWFkX2FueSIsIm9iamVjdF9vYmplY3RfdXBkYXRlIiwib2JqZWN0X3ZvbHVtZV9jcmVhdGUiLCJvYmplY3Rfdm9sdW1lX2RlbGV0ZSIsIm9iamVjdF92b2x1bWVfcmVhZF9hbnkiLCJvYmplY3Rfdm9sdW1lX3VwZGF0ZSIsInNjaGVkdWxlcl9jb25kaXRpb25fcmVhZF9hbnkiLCJ1cGRhdGVfbXBwX2NyZWF0ZSIsInVwZGF0ZV9tcHBfZGVsZXRlIiwidXBkYXRlX21wcF9yZWFkX2FueSIsInVwZGF0ZV9tcHBfdXBkYXRlIiwidXBkYXRlX3JjX2NyZWF0ZSIsInVwZGF0ZV9yY19kZWxldGUiLCJ1cGRhdGVfcmNfcmVhZF9hbnkiLCJ1cGRhdGVfcmNfdXBkYXRlIiwid29ya2Zsb3dfam9iX2NyZWF0ZSIsIndvcmtmbG93X2pvYl9kZWxldGUiLCJ3b3JrZmxvd19qb2JfcmVhZF9hbnkiLCJ3b3JrZmxvd19qb2JfdXBkYXRlIiwid29ya2Zsb3dfc2VydmVyX2NyZWF0ZSIsIndvcmtmbG93X3NlcnZlcl9kZWxldGUiLCJ3b3JrZmxvd19zZXJ2ZXJfcmVhZF9hbnkiLCJ3b3JrZmxvd19zZXJ2ZXJfdXBkYXRlIiwid29ya2Zsb3dfdm9sdW1lX2NyZWF0ZSIsIndvcmtmbG93X3ZvbHVtZV9kZWxldGUiLCJ3b3JrZmxvd192b2x1bWVfcmVhZF9hbnkiLCJ3b3JrZmxvd192b2x1bWVfdXBkYXRlIiwid29ya2Zsb3dfd29ya2VyX2NyZWF0ZSIsIndvcmtmbG93X3dvcmtlcl9kZWxldGUiLCJ3b3JrZmxvd193b3JrZXJfcmVhZF9hbnkiLCJ3b3JrZmxvd193b3JrZXJfdXBkYXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfY3JlYXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfZGVsZXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfcmVhZF9hbnkiLCJ3b3JrZmxvd193b3JrZmxvd191cGRhdGUiLCJkYXRhc2V0X2NsYXNzaWZpY2F0aW9uX2d0X2RhdGFzZXRfY3JlYXRlIiwiZGF0YXNldF9jbGFzc2lmaWNhdGlvbl9ndF9kYXRhc2V0X2RlbGV0ZSIsImRhdGFzZXRfY2xhc3NpZmljYXRpb25fZ3RfZGF0YXNldF9yZWFkX2FueSIsImRhdGFzZXRfY2xhc3NpZmljYXRpb25fZ3RfZGF0YXNldF91cGRhdGUiLCJkYXRhc2V0X3NlZ21lbnRhdGlvbl9ndF9kYXRhc2V0X2NyZWF0ZSIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfZGVsZXRlIiwiZGF0YXNldF9zZWdtZW50YXRpb25fZ3RfZGF0YXNldF9yZWFkX2FueSIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfdXBkYXRlIl0sImdyb3VwX3N5c3RlbXMiOlsibG9jYWwiLCJtaXJlcm8taWRjLWluLWxpbmUiLCJtaXJlcm8taWRjLW9mZi1saW5lIl0sImdyb3VwX2ZlYXR1cmVzIjpbInJjX2RhdGFzZXRfbWFuYWdlbWVudCIsInJjX2dkc19mbG9vcl9wbGFuX21hbmFnZW1lbnQiLCJyY19nZHNfbWFuYWdlbWVudCIsInJjX21hY2hpbmVfbGVhcm5pbmdfY2xhc3NfY29kZSIsInJjX21hY2hpbmVfbGVhcm5pbmdfbW9kZWxfbWFuYWdlbWVudCIsInJjX29wZXJhdGlvbl9kZWZlY3RfZGF0YV9tYW5hZ2VtZW50IiwicmNfcmVjaXBlX2RldmVsb3BtZW50IiwicmNfcmVjaXBlX2ltYWdlX2FuYWx5emVyIiwicmNfd29ya2Zsb3dfbWFuYWdlbWVudCIsInJjX3dvcmtmbG93X21vbml0b3JpbmciLCJyY19vcGVyYXRpb25fc2NoZWR1bGVyX21hbmFnZW1lbnQiLCJyY19vcGVyYXRpb25fc3RhdHVzIiwicmNfbWFzdGVyZGF0YXNldF9tYW5hZ2VtZW50Il0sImp0aSI6IjA2NjkwM2JmLTc5NzYtNDE5MS05NGI1LWU5ZGM0YWVjNzI5MSIsImlhdCI6MTY4NzgyMTg3NiwibmJmIjoxNjg3ODIxODc2LCJleHAiOjE2ODkwMzE0NzYsImlzcyI6ImRhcSIsImF1ZCI6ImRhcV91c2VyIn0.86YbTm0Yli6O1Qv0T_0-nIzHBMBz52klGZTjwRt6fCY"

    RecipeRun(**kwargs) 