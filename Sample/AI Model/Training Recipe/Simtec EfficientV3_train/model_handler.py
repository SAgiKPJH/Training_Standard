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
from keras.applications.efficientnet import EfficientNetB3
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers.optimizer_v2.adam import Adam
from keras.optimizers.optimizer_v2.rmsprop import RMSprop
from keras.optimizers.legacy.sgd import SGD
import traceback

import keras.backend as K 

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
        "batch_size" : 32,
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
        "learningRate" : 0.001,
        "reduceLRPatience" : 0,
        "earlyStopPatience" : 0,
        "epoch" : 5,
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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    # if kwargs['hyperparameter']['using_gpu'] and not torch.cuda.is_available():
    #     raise Exception("GPU is not avaiable")

    global logic_start_time
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
    # bucket_id = "ADC_DEMO_SAMPLE_090201(ADC_DEMO_SAMPLE_090201)_debugtest"
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
        print("start")
        logic_start_time = time.time()


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

        inputSize = 224
        train_generator = idg.flow_from_directory(directory = local_download_path, subset="training", target_size = (inputSize,inputSize), batch_size=8)
        valid_generator = idg.flow_from_directory(directory = local_download_path, subset="validation", target_size = (inputSize,inputSize), batch_size=8)

        inference_info = {'input_size': inputSize, 'label_info': label_info}

        if hyperparameter['usePtWeights'] == False:
            md = EfficientNetB3(input_shape = (inputSize, inputSize, 3), weights = None, include_top = True, classes = num_classes)
            model = Model(inputs=md.input, outputs=md.output)
        else:
            curDir = os.path.dirname(os.path.abspath(__file__))
            ptWeightPath = os.path.join(curDir, "efficientnetb3_notop.h5")
            logger.info(f"loading pre-trained model from '{ptWeightPath}'")
            md = EfficientNetB3(input_shape = (inputSize, inputSize, 3), weights = ptWeightPath, include_top = False)
            a = GlobalAveragePooling2D()(md.output)
            a = Dense(num_classes, activation = 'softmax')(a)
            model = Model(inputs=md.input, outputs = a)

        
        if hyperparameter['rtWeightPath'] != "":
            model.load_weights(hyperparameter['rtWeightPath'])

        opt = None
        if hyperparameter['optimizer'] == 'adam':         opt = Adam(float(hyperparameter['learningRate']))
        elif hyperparameter['optimizer'] == 'rmsprop':    opt = RMSprop(float(hyperparameter['learningRate']))
        elif hyperparameter['optimizer'] == 'sgd':        opt = SGD(float(hyperparameter['learningRate']))

        with K.tf.device(device) : 
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

            ep = EarlyStopping(monitor='val_acc', patience = int(hyperparameter['earlyStopPatience']), verbose=1)
            lrs = ReduceLROnPlateau(monitor='val_loss', patience = int(hyperparameter['reduceLRPatience']), factor=0.5, verbose =1)
            lc = LambdaCallback(on_epoch_begin=on_epoch_begin,on_epoch_end=on_epoch_end)

            cb = []
            cb.append(lc)
            if int(hyperparameter['reduceLRPatience']) != 0:   cb.append(lrs)
            if int(hyperparameter['earlyStopPatience']) != 0:  cb.append(ep)

            model.fit_generator(train_generator, validation_data = valid_generator, epochs = int(hyperparameter['epoch']), steps_per_epoch = len(train_generator), validation_steps = len(valid_generator), callbacks=cb)



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
    valid_epoch_acc.append(valid_acc * 100)
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
    kwargs['authentication']['operation_service_address'] = "192.168.70.31:5020"
    #kwargs['authentication']['operation_service_address'] = "192.168.70.62:5020"
    #kwargs['authentication']['operation_service_address'] = "192.168.70.101:5020"
    #kwargs['authentication']['access_token'] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJtaW53b29uZy5wYXJrIiwibmFtZSI6IuuwleuvvOybhSIsInJvbGUiOiJkZXZlbG9wZXIiLCJncm91cHNpZCI6ImRlZmF1bHQiLCJsb2dfaW5fcHJvdmlkZXIiOiJkYXEiLCJwcml2aWxlZ2VzIjpbImRhdGFzZXRfY2xhc3NfY29kZV9jcmVhdGUiLCJkYXRhc2V0X2NsYXNzX2NvZGVfZGVsZXRlIiwiZGF0YXNldF9jbGFzc19jb2RlX3JlYWRfYW55IiwiZGF0YXNldF9jbGFzc19jb2RlX3VwZGF0ZSIsImRhdGFzZXRfZ3RfZGF0YXNldF9jcmVhdGUiLCJkYXRhc2V0X2d0X2RhdGFzZXRfZGVsZXRlIiwiZGF0YXNldF9ndF9kYXRhc2V0X3JlYWRfYW55IiwiZGF0YXNldF9ndF9kYXRhc2V0X3VwZGF0ZSIsImRhdGFzZXRfaW1hZ2VfZGF0YXNldF9jcmVhdGUiLCJkYXRhc2V0X2ltYWdlX2RhdGFzZXRfZGVsZXRlIiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X3JlYWRfYW55IiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X3VwZGF0ZSIsImRhdGFzZXRfdm9sdW1lX2NyZWF0ZSIsImRhdGFzZXRfdm9sdW1lX2RlbGV0ZSIsImRhdGFzZXRfdm9sdW1lX3JlYWRfYW55IiwiZGF0YXNldF92b2x1bWVfdXBkYXRlIiwiZGVmZWN0X2RhdGFfbWFuYWdlcl9tYXN0ZXJfaW5mb3JtYXRpb25fY3JlYXRlIiwiZGVmZWN0X2RhdGFfbWFuYWdlcl9tYXN0ZXJfaW5mb3JtYXRpb25fZGVsZXRlIiwiZGVmZWN0X2RhdGFfbWFuYWdlcl9tYXN0ZXJfaW5mb3JtYXRpb25fcmVhZF9hbnkiLCJkZWZlY3RfZGF0YV9tYW5hZ2VyX21hc3Rlcl9pbmZvcm1hdGlvbl91cGRhdGUiLCJkZWZlY3RfZGVmZWN0X2RhdGFfY3JlYXRlIiwiZGVmZWN0X2RlZmVjdF9kYXRhX2RlbGV0ZSIsImRlZmVjdF9kZWZlY3RfZGF0YV9yZWFkX2FueSIsImRlZmVjdF9kZWZlY3RfZGF0YV91cGRhdGUiLCJkZWZlY3RfaWRic19tYXN0ZXJfaW5mb3JtYXRpb25fY3JlYXRlIiwiZGVmZWN0X2lkYnNfbWFzdGVyX2luZm9ybWF0aW9uX2RlbGV0ZSIsImRlZmVjdF9pZGJzX21hc3Rlcl9pbmZvcm1hdGlvbl9yZWFkX2FueSIsImRlZmVjdF9pZGJzX21hc3Rlcl9pbmZvcm1hdGlvbl91cGRhdGUiLCJnZHNfY2xpcF9jcmVhdGUiLCJnZHNfY2xpcF9kZWxldGUiLCJnZHNfY2xpcF9yZWFkX2FueSIsImdkc19jbGlwX3VwZGF0ZSIsImdkc19leHBvcnRfY3JlYXRlIiwiZ2RzX2V4cG9ydF9kZWxldGUiLCJnZHNfZXhwb3J0X3JlYWRfYW55IiwiZ2RzX2V4cG9ydF91cGRhdGUiLCJnZHNfZ2RzX2NyZWF0ZSIsImdkc19nZHNfZGVsZXRlIiwiZ2RzX2dkc19yZWFkX2FueSIsImdkc19nZHNfdXBkYXRlIiwiZ2RzX3NlcnZlcl9jcmVhdGUiLCJnZHNfc2VydmVyX2RlbGV0ZSIsImdkc19zZXJ2ZXJfcmVhZF9hbnkiLCJnZHNfc2VydmVyX3VwZGF0ZSIsImdkc192b2x1bWVfY3JlYXRlIiwiZ2RzX3ZvbHVtZV9kZWxldGUiLCJnZHNfdm9sdW1lX3JlYWRfYW55IiwiZ2RzX3ZvbHVtZV91cGRhdGUiLCJpbmZlcmVuY2VfaW5mZXJlbmNlX2NyZWF0ZSIsImluZmVyZW5jZV9pbmZlcmVuY2VfZGVsZXRlIiwiaW5mZXJlbmNlX2luZmVyZW5jZV9yZWFkX2FueSIsImluZmVyZW5jZV9pbmZlcmVuY2VfdXBkYXRlIiwiaW5mZXJlbmNlX21vZGVsX2NyZWF0ZSIsImluZmVyZW5jZV9tb2RlbF9kZWxldGUiLCJpbmZlcmVuY2VfbW9kZWxfcmVhZF9hbnkiLCJpbmZlcmVuY2VfbW9kZWxfdXBkYXRlIiwiaW5mZXJlbmNlX3NlcnZlcl9jcmVhdGUiLCJpbmZlcmVuY2Vfc2VydmVyX2RlbGV0ZSIsImluZmVyZW5jZV9zZXJ2ZXJfcmVhZF9hbnkiLCJpbmZlcmVuY2Vfc2VydmVyX3VwZGF0ZSIsImluZmVyZW5jZV92b2x1bWVfY3JlYXRlIiwiaW5mZXJlbmNlX3ZvbHVtZV9kZWxldGUiLCJpbmZlcmVuY2Vfdm9sdW1lX3JlYWRfYW55IiwiaW5mZXJlbmNlX3ZvbHVtZV91cGRhdGUiLCJvYmplY3Rfb2JqZWN0X2NyZWF0ZSIsIm9iamVjdF9vYmplY3RfZGVsZXRlIiwib2JqZWN0X29iamVjdF9yZWFkX2FueSIsIm9iamVjdF9vYmplY3RfdXBkYXRlIiwib2JqZWN0X3ZvbHVtZV9jcmVhdGUiLCJvYmplY3Rfdm9sdW1lX2RlbGV0ZSIsIm9iamVjdF92b2x1bWVfcmVhZF9hbnkiLCJvYmplY3Rfdm9sdW1lX3VwZGF0ZSIsInNjaGVkdWxlcl9jb25kaXRpb25fcmVhZF9hbnkiLCJ1cGRhdGVfbXBwX2NyZWF0ZSIsInVwZGF0ZV9tcHBfZGVsZXRlIiwidXBkYXRlX21wcF9yZWFkX2FueSIsInVwZGF0ZV9tcHBfdXBkYXRlIiwidXBkYXRlX3JjX2NyZWF0ZSIsInVwZGF0ZV9yY19kZWxldGUiLCJ1cGRhdGVfcmNfcmVhZF9hbnkiLCJ1cGRhdGVfcmNfdXBkYXRlIiwid29ya2Zsb3dfam9iX2NyZWF0ZSIsIndvcmtmbG93X2pvYl9kZWxldGUiLCJ3b3JrZmxvd19qb2JfcmVhZF9hbnkiLCJ3b3JrZmxvd19qb2JfdXBkYXRlIiwid29ya2Zsb3dfc2VydmVyX2NyZWF0ZSIsIndvcmtmbG93X3NlcnZlcl9kZWxldGUiLCJ3b3JrZmxvd19zZXJ2ZXJfcmVhZF9hbnkiLCJ3b3JrZmxvd19zZXJ2ZXJfdXBkYXRlIiwid29ya2Zsb3dfdm9sdW1lX2NyZWF0ZSIsIndvcmtmbG93X3ZvbHVtZV9kZWxldGUiLCJ3b3JrZmxvd192b2x1bWVfcmVhZF9hbnkiLCJ3b3JrZmxvd192b2x1bWVfdXBkYXRlIiwid29ya2Zsb3dfd29ya2VyX2NyZWF0ZSIsIndvcmtmbG93X3dvcmtlcl9kZWxldGUiLCJ3b3JrZmxvd193b3JrZXJfcmVhZF9hbnkiLCJ3b3JrZmxvd193b3JrZXJfdXBkYXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfY3JlYXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfZGVsZXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfcmVhZF9hbnkiLCJ3b3JrZmxvd193b3JrZmxvd191cGRhdGUiLCJkYXRhc2V0X2NsYXNzaWZpY2F0aW9uX2d0X2RhdGFzZXRfY3JlYXRlIiwiZGF0YXNldF9jbGFzc2lmaWNhdGlvbl9ndF9kYXRhc2V0X2RlbGV0ZSIsImRhdGFzZXRfY2xhc3NpZmljYXRpb25fZ3RfZGF0YXNldF9yZWFkX2FueSIsImRhdGFzZXRfY2xhc3NpZmljYXRpb25fZ3RfZGF0YXNldF91cGRhdGUiLCJkYXRhc2V0X3NlZ21lbnRhdGlvbl9ndF9kYXRhc2V0X2NyZWF0ZSIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfZGVsZXRlIiwiZGF0YXNldF9zZWdtZW50YXRpb25fZ3RfZGF0YXNldF9yZWFkX2FueSIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfdXBkYXRlIl0sImdyb3VwX3N5c3RlbXMiOlsibG9jYWwiLCJtaXJlcm8taWRjLWluLWxpbmUiLCJtaXJlcm8taWRjLW9mZi1saW5lIl0sImdyb3VwX2ZlYXR1cmVzIjpbInJjX2RhdGFzZXRfbWFuYWdlbWVudCIsInJjX2dkc19mbG9vcl9wbGFuX21hbmFnZW1lbnQiLCJyY19nZHNfbWFuYWdlbWVudCIsInJjX21hY2hpbmVfbGVhcm5pbmdfY2xhc3NfY29kZSIsInJjX21hY2hpbmVfbGVhcm5pbmdfbW9kZWxfbWFuYWdlbWVudCIsInJjX29wZXJhdGlvbl9kZWZlY3RfZGF0YV9tYW5hZ2VtZW50IiwicmNfcmVjaXBlX2RldmVsb3BtZW50IiwicmNfcmVjaXBlX2ltYWdlX2FuYWx5emVyIiwicmNfd29ya2Zsb3dfbWFuYWdlbWVudCIsInJjX3dvcmtmbG93X21vbml0b3JpbmciLCJyY19vcGVyYXRpb25fc2NoZWR1bGVyX21hbmFnZW1lbnQiLCJyY19vcGVyYXRpb25fc3RhdHVzIiwicmNfbWFzdGVyZGF0YXNldF9tYW5hZ2VtZW50Il0sImp0aSI6IjA2NjkwM2JmLTc5NzYtNDE5MS05NGI1LWU5ZGM0YWVjNzI5MSIsImlhdCI6MTY4NzgyMTg3NiwibmJmIjoxNjg3ODIxODc2LCJleHAiOjE2ODkwMzE0NzYsImlzcyI6ImRhcSIsImF1ZCI6ImRhcV91c2VyIn0.86YbTm0Yli6O1Qv0T_0-nIzHBMBz52klGZTjwRt6fCY"
    kwargs['authentication']['access_token'] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbmlzdHJhdG9yIiwibmFtZSI6Iuq0gOumrOyekCIsInJvbGUiOiJzdXBlcl9hZG1pbmlzdHJhdG9yIiwiZ3JvdXBzaWQiOiJkZWZhdWx0IiwibG9nX2luX3Byb3ZpZGVyIjoiZGFxIiwicHJpdmlsZWdlcyI6WyJhY2NvdW50X2dyb3VwX2NyZWF0ZSIsImFjY291bnRfZ3JvdXBfZGVsZXRlIiwiYWNjb3VudF9ncm91cF9kZWxldGVfYW55IiwiYWNjb3VudF9ncm91cF9yZWFkIiwiYWNjb3VudF9ncm91cF9yZWFkX2FueSIsImFjY291bnRfZ3JvdXBfdXBkYXRlIiwiYWNjb3VudF9ncm91cF91cGRhdGVfYW55IiwiYWNjb3VudF91c2VyX2NyZWF0ZSIsImFjY291bnRfdXNlcl9kZWxldGUiLCJhY2NvdW50X3VzZXJfZGVsZXRlX2FueSIsImFjY291bnRfdXNlcl9yZWFkIiwiYWNjb3VudF91c2VyX3JlYWRfYW55IiwiYWNjb3VudF91c2VyX3VwZGF0ZSIsImFjY291bnRfdXNlcl91cGRhdGVfYW55IiwiZGF0YXNldF9jbGFzc19jb2RlX2NyZWF0ZSIsImRhdGFzZXRfY2xhc3NfY29kZV9kZWxldGUiLCJkYXRhc2V0X2NsYXNzX2NvZGVfZGVsZXRlX2FueSIsImRhdGFzZXRfY2xhc3NfY29kZV9yZWFkIiwiZGF0YXNldF9jbGFzc19jb2RlX3JlYWRfYW55IiwiZGF0YXNldF9jbGFzc19jb2RlX3VwZGF0ZSIsImRhdGFzZXRfY2xhc3NfY29kZV91cGRhdGVfYW55IiwiZGF0YXNldF9jbGFzc2lmaWNhdGlvbl9ndF9kYXRhc2V0X2NyZWF0ZSIsImRhdGFzZXRfY2xhc3NpZmljYXRpb25fZ3RfZGF0YXNldF9kZWxldGUiLCJkYXRhc2V0X2NsYXNzaWZpY2F0aW9uX2d0X2RhdGFzZXRfZGVsZXRlX2FueSIsImRhdGFzZXRfY2xhc3NpZmljYXRpb25fZ3RfZGF0YXNldF9yZWFkIiwiZGF0YXNldF9jbGFzc2lmaWNhdGlvbl9ndF9kYXRhc2V0X3JlYWRfYW55IiwiZGF0YXNldF9jbGFzc2lmaWNhdGlvbl9ndF9kYXRhc2V0X3VwZGF0ZSIsImRhdGFzZXRfY2xhc3NpZmljYXRpb25fZ3RfZGF0YXNldF91cGRhdGVfYW55IiwiZGF0YXNldF9ndF9kYXRhc2V0X2NyZWF0ZSIsImRhdGFzZXRfZ3RfZGF0YXNldF9kZWxldGUiLCJkYXRhc2V0X2d0X2RhdGFzZXRfZGVsZXRlX2FueSIsImRhdGFzZXRfZ3RfZGF0YXNldF9yZWFkIiwiZGF0YXNldF9ndF9kYXRhc2V0X3JlYWRfYW55IiwiZGF0YXNldF9ndF9kYXRhc2V0X3VwZGF0ZSIsImRhdGFzZXRfZ3RfZGF0YXNldF91cGRhdGVfYW55IiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X2NyZWF0ZSIsImRhdGFzZXRfaW1hZ2VfZGF0YXNldF9kZWxldGUiLCJkYXRhc2V0X2ltYWdlX2RhdGFzZXRfZGVsZXRlX2FueSIsImRhdGFzZXRfaW1hZ2VfZGF0YXNldF9yZWFkIiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X3JlYWRfYW55IiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X3VwZGF0ZSIsImRhdGFzZXRfaW1hZ2VfZGF0YXNldF91cGRhdGVfYW55IiwiZGF0YXNldF9zZWdtZW50YXRpb25fZ3RfZGF0YXNldF9jcmVhdGUiLCJkYXRhc2V0X3NlZ21lbnRhdGlvbl9ndF9kYXRhc2V0X2RlbGV0ZSIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfZGVsZXRlX2FueSIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfcmVhZCIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfcmVhZF9hbnkiLCJkYXRhc2V0X3NlZ21lbnRhdGlvbl9ndF9kYXRhc2V0X3VwZGF0ZSIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfdXBkYXRlX2FueSIsImRhdGFzZXRfdm9sdW1lX2NyZWF0ZSIsImRhdGFzZXRfdm9sdW1lX2RlbGV0ZSIsImRhdGFzZXRfdm9sdW1lX2RlbGV0ZV9hbnkiLCJkYXRhc2V0X3ZvbHVtZV9yZWFkIiwiZGF0YXNldF92b2x1bWVfcmVhZF9hbnkiLCJkYXRhc2V0X3ZvbHVtZV91cGRhdGUiLCJkYXRhc2V0X3ZvbHVtZV91cGRhdGVfYW55IiwiZGVmZWN0X2RlZmVjdF9kYXRhX2NyZWF0ZSIsImRlZmVjdF9kZWZlY3RfZGF0YV9kZWxldGUiLCJkZWZlY3RfZGVmZWN0X2RhdGFfZGVsZXRlX2FueSIsImRlZmVjdF9kZWZlY3RfZGF0YV9yZWFkIiwiZGVmZWN0X2RlZmVjdF9kYXRhX3JlYWRfYW55IiwiZGVmZWN0X2RlZmVjdF9kYXRhX3VwZGF0ZSIsImRlZmVjdF9kZWZlY3RfZGF0YV91cGRhdGVfYW55IiwiZGVmZWN0X21hc3Rlcl9kYXRhX2ZpZWxkX2NyZWF0ZSIsImRlZmVjdF9tYXN0ZXJfZGF0YV9maWVsZF9kZWxldGUiLCJkZWZlY3RfbWFzdGVyX2RhdGFfZmllbGRfZGVsZXRlX2FueSIsImRlZmVjdF9tYXN0ZXJfZGF0YV9maWVsZF9ncm91cF9oaXN0b3J5X2NyZWF0ZSIsImRlZmVjdF9tYXN0ZXJfZGF0YV9maWVsZF9ncm91cF9oaXN0b3J5X2RlbGV0ZSIsImRlZmVjdF9tYXN0ZXJfZGF0YV9maWVsZF9ncm91cF9oaXN0b3J5X2RlbGV0ZV9hbnkiLCJkZWZlY3RfbWFzdGVyX2RhdGFfZmllbGRfZ3JvdXBfaGlzdG9yeV9yZWFkIiwiZGVmZWN0X21hc3Rlcl9kYXRhX2ZpZWxkX2dyb3VwX2hpc3RvcnlfcmVhZF9hbnkiLCJkZWZlY3RfbWFzdGVyX2RhdGFfZmllbGRfZ3JvdXBfaGlzdG9yeV91cGRhdGUiLCJkZWZlY3RfbWFzdGVyX2RhdGFfZmllbGRfZ3JvdXBfaGlzdG9yeV91cGRhdGVfYW55IiwiZGVmZWN0X21hc3Rlcl9kYXRhX2ZpZWxkX3JlYWQiLCJkZWZlY3RfbWFzdGVyX2RhdGFfZmllbGRfcmVhZF9hbnkiLCJkZWZlY3RfbWFzdGVyX2RhdGFfZmllbGRfdXBkYXRlIiwiZGVmZWN0X21hc3Rlcl9kYXRhX2ZpZWxkX3VwZGF0ZV9hbnkiLCJkZWZlY3RfbWFzdGVyX2RhdGFfaGlzdG9yeV9jcmVhdGUiLCJkZWZlY3RfbWFzdGVyX2RhdGFfaGlzdG9yeV9kZWxldGUiLCJkZWZlY3RfbWFzdGVyX2RhdGFfaGlzdG9yeV9kZWxldGVfYW55IiwiZGVmZWN0X21hc3Rlcl9kYXRhX2hpc3RvcnlfcmVhZCIsImRlZmVjdF9tYXN0ZXJfZGF0YV9oaXN0b3J5X3JlYWRfYW55IiwiZGVmZWN0X21hc3Rlcl9kYXRhX2hpc3RvcnlfdXBkYXRlIiwiZGVmZWN0X21hc3Rlcl9kYXRhX2hpc3RvcnlfdXBkYXRlX2FueSIsImRlZmVjdF9tYXN0ZXJfZGF0YV9zZXRfY3JlYXRlIiwiZGVmZWN0X21hc3Rlcl9kYXRhX3NldF9kZWxldGUiLCJkZWZlY3RfbWFzdGVyX2RhdGFfc2V0X2RlbGV0ZV9hbnkiLCJkZWZlY3RfbWFzdGVyX2RhdGFfc2V0X3JlYWQiLCJkZWZlY3RfbWFzdGVyX2RhdGFfc2V0X3JlYWRfYW55IiwiZGVmZWN0X21hc3Rlcl9kYXRhX3NldF91cGRhdGUiLCJkZWZlY3RfbWFzdGVyX2RhdGFfc2V0X3VwZGF0ZV9hbnkiLCJnZHNfY2xpcF9jcmVhdGUiLCJnZHNfY2xpcF9kZWxldGUiLCJnZHNfY2xpcF9kZWxldGVfYW55IiwiZ2RzX2NsaXBfcmVhZCIsImdkc19jbGlwX3JlYWRfYW55IiwiZ2RzX2NsaXBfdXBkYXRlIiwiZ2RzX2NsaXBfdXBkYXRlX2FueSIsImdkc19leHBvcnRfY3JlYXRlIiwiZ2RzX2V4cG9ydF9kZWxldGUiLCJnZHNfZXhwb3J0X2RlbGV0ZV9hbnkiLCJnZHNfZXhwb3J0X3JlYWQiLCJnZHNfZXhwb3J0X3JlYWRfYW55IiwiZ2RzX2V4cG9ydF91cGRhdGUiLCJnZHNfZXhwb3J0X3VwZGF0ZV9hbnkiLCJnZHNfZ2RzX2NyZWF0ZSIsImdkc19nZHNfZGVsZXRlIiwiZ2RzX2dkc19kZWxldGVfYW55IiwiZ2RzX2dkc19yZWFkIiwiZ2RzX2dkc19yZWFkX2FueSIsImdkc19nZHNfdXBkYXRlIiwiZ2RzX2dkc191cGRhdGVfYW55IiwiZ2RzX3NlcnZlcl9jcmVhdGUiLCJnZHNfc2VydmVyX2RlbGV0ZSIsImdkc19zZXJ2ZXJfZGVsZXRlX2FueSIsImdkc19zZXJ2ZXJfcmVhZCIsImdkc19zZXJ2ZXJfcmVhZF9hbnkiLCJnZHNfc2VydmVyX3VwZGF0ZSIsImdkc19zZXJ2ZXJfdXBkYXRlX2FueSIsImdkc192b2x1bWVfY3JlYXRlIiwiZ2RzX3ZvbHVtZV9kZWxldGUiLCJnZHNfdm9sdW1lX2RlbGV0ZV9hbnkiLCJnZHNfdm9sdW1lX3JlYWQiLCJnZHNfdm9sdW1lX3JlYWRfYW55IiwiZ2RzX3ZvbHVtZV91cGRhdGUiLCJnZHNfdm9sdW1lX3VwZGF0ZV9hbnkiLCJpbmZlcmVuY2VfaW5mZXJlbmNlX2NyZWF0ZSIsImluZmVyZW5jZV9pbmZlcmVuY2VfZGVsZXRlIiwiaW5mZXJlbmNlX2luZmVyZW5jZV9kZWxldGVfYW55IiwiaW5mZXJlbmNlX2luZmVyZW5jZV9yZWFkIiwiaW5mZXJlbmNlX2luZmVyZW5jZV9yZWFkX2FueSIsImluZmVyZW5jZV9pbmZlcmVuY2VfdXBkYXRlIiwiaW5mZXJlbmNlX2luZmVyZW5jZV91cGRhdGVfYW55IiwiaW5mZXJlbmNlX21vZGVsX2NyZWF0ZSIsImluZmVyZW5jZV9tb2RlbF9kZWxldGUiLCJpbmZlcmVuY2VfbW9kZWxfZGVsZXRlX2FueSIsImluZmVyZW5jZV9tb2RlbF9wb2xpY3lfY2hhbmdlIiwiaW5mZXJlbmNlX21vZGVsX3JlYWQiLCJpbmZlcmVuY2VfbW9kZWxfcmVhZF9hbnkiLCJpbmZlcmVuY2VfbW9kZWxfdXBkYXRlIiwiaW5mZXJlbmNlX21vZGVsX3VwZGF0ZV9hbnkiLCJpbmZlcmVuY2Vfc2VydmVyX2NyZWF0ZSIsImluZmVyZW5jZV9zZXJ2ZXJfZGVsZXRlIiwiaW5mZXJlbmNlX3NlcnZlcl9kZWxldGVfYW55IiwiaW5mZXJlbmNlX3NlcnZlcl9yZWFkIiwiaW5mZXJlbmNlX3NlcnZlcl9yZWFkX2FueSIsImluZmVyZW5jZV9zZXJ2ZXJfdXBkYXRlIiwiaW5mZXJlbmNlX3NlcnZlcl91cGRhdGVfYW55IiwiaW5mZXJlbmNlX3ZvbHVtZV9jcmVhdGUiLCJpbmZlcmVuY2Vfdm9sdW1lX2RlbGV0ZSIsImluZmVyZW5jZV92b2x1bWVfZGVsZXRlX2FueSIsImluZmVyZW5jZV92b2x1bWVfcmVhZCIsImluZmVyZW5jZV92b2x1bWVfcmVhZF9hbnkiLCJpbmZlcmVuY2Vfdm9sdW1lX3VwZGF0ZSIsImluZmVyZW5jZV92b2x1bWVfdXBkYXRlX2FueSIsIm9iamVjdF9vYmplY3RfY3JlYXRlIiwib2JqZWN0X29iamVjdF9kZWxldGUiLCJvYmplY3Rfb2JqZWN0X2RlbGV0ZV9hbnkiLCJvYmplY3Rfb2JqZWN0X3JlYWQiLCJvYmplY3Rfb2JqZWN0X3JlYWRfYW55Iiwib2JqZWN0X29iamVjdF91cGRhdGUiLCJvYmplY3Rfb2JqZWN0X3VwZGF0ZV9hbnkiLCJvYmplY3Rfdm9sdW1lX2NyZWF0ZSIsIm9iamVjdF92b2x1bWVfZGVsZXRlIiwib2JqZWN0X3ZvbHVtZV9kZWxldGVfYW55Iiwib2JqZWN0X3ZvbHVtZV9yZWFkIiwib2JqZWN0X3ZvbHVtZV9yZWFkX2FueSIsIm9iamVjdF92b2x1bWVfdXBkYXRlIiwib2JqZWN0X3ZvbHVtZV91cGRhdGVfYW55Iiwic2NoZWR1bGVyX2NvbmRpdGlvbl9jcmVhdGUiLCJzY2hlZHVsZXJfY29uZGl0aW9uX2RlbGV0ZSIsInNjaGVkdWxlcl9jb25kaXRpb25fZGVsZXRlX2FueSIsInNjaGVkdWxlcl9jb25kaXRpb25fcmVhZCIsInNjaGVkdWxlcl9jb25kaXRpb25fcmVhZF9hbnkiLCJzY2hlZHVsZXJfY29uZGl0aW9uX3VwZGF0ZSIsInNjaGVkdWxlcl9jb25kaXRpb25fdXBkYXRlX2FueSIsInNjaGVkdWxlcl9ldmVudF9oaXN0b3J5X2NyZWF0ZSIsInNjaGVkdWxlcl9ldmVudF9oaXN0b3J5X2RlbGV0ZSIsInNjaGVkdWxlcl9ldmVudF9oaXN0b3J5X2RlbGV0ZV9hbnkiLCJzY2hlZHVsZXJfZXZlbnRfaGlzdG9yeV9yZWFkIiwic2NoZWR1bGVyX2V2ZW50X2hpc3RvcnlfcmVhZF9hbnkiLCJzY2hlZHVsZXJfZXZlbnRfaGlzdG9yeV91cGRhdGUiLCJzY2hlZHVsZXJfZXZlbnRfaGlzdG9yeV91cGRhdGVfYW55IiwidXBkYXRlX21wcF9jcmVhdGUiLCJ1cGRhdGVfbXBwX2RlbGV0ZSIsInVwZGF0ZV9tcHBfZGVsZXRlX2FueSIsInVwZGF0ZV9tcHBfcmVhZCIsInVwZGF0ZV9tcHBfcmVhZF9hbnkiLCJ1cGRhdGVfbXBwX3VwZGF0ZSIsInVwZGF0ZV9tcHBfdXBkYXRlX2FueSIsInVwZGF0ZV9yY19jcmVhdGUiLCJ1cGRhdGVfcmNfZGVsZXRlIiwidXBkYXRlX3JjX2RlbGV0ZV9hbnkiLCJ1cGRhdGVfcmNfcmVhZCIsInVwZGF0ZV9yY19yZWFkX2FueSIsInVwZGF0ZV9yY191cGRhdGUiLCJ1cGRhdGVfcmNfdXBkYXRlX2FueSIsIndvcmtmbG93X2pvYl9jcmVhdGUiLCJ3b3JrZmxvd19qb2JfZGVsZXRlIiwid29ya2Zsb3dfam9iX2RlbGV0ZV9hbnkiLCJ3b3JrZmxvd19qb2JfcmVhZCIsIndvcmtmbG93X2pvYl9yZWFkX2FueSIsIndvcmtmbG93X2pvYl91cGRhdGUiLCJ3b3JrZmxvd19qb2JfdXBkYXRlX2FueSIsIndvcmtmbG93X3NlcnZlcl9jcmVhdGUiLCJ3b3JrZmxvd19zZXJ2ZXJfZGVsZXRlIiwid29ya2Zsb3dfc2VydmVyX2RlbGV0ZV9hbnkiLCJ3b3JrZmxvd19zZXJ2ZXJfcmVhZCIsIndvcmtmbG93X3NlcnZlcl9yZWFkX2FueSIsIndvcmtmbG93X3NlcnZlcl91cGRhdGUiLCJ3b3JrZmxvd19zZXJ2ZXJfdXBkYXRlX2FueSIsIndvcmtmbG93X3ZvbHVtZV9jcmVhdGUiLCJ3b3JrZmxvd192b2x1bWVfZGVsZXRlIiwid29ya2Zsb3dfdm9sdW1lX2RlbGV0ZV9hbnkiLCJ3b3JrZmxvd192b2x1bWVfcmVhZCIsIndvcmtmbG93X3ZvbHVtZV9yZWFkX2FueSIsIndvcmtmbG93X3ZvbHVtZV91cGRhdGUiLCJ3b3JrZmxvd192b2x1bWVfdXBkYXRlX2FueSIsIndvcmtmbG93X3dvcmtlcl9jcmVhdGUiLCJ3b3JrZmxvd193b3JrZXJfZGVsZXRlIiwid29ya2Zsb3dfd29ya2VyX2RlbGV0ZV9hbnkiLCJ3b3JrZmxvd193b3JrZXJfcmVhZCIsIndvcmtmbG93X3dvcmtlcl9yZWFkX2FueSIsIndvcmtmbG93X3dvcmtlcl91cGRhdGUiLCJ3b3JrZmxvd193b3JrZXJfdXBkYXRlX2FueSIsIndvcmtmbG93X3dvcmtmbG93X2NyZWF0ZSIsIndvcmtmbG93X3dvcmtmbG93X2RlbGV0ZSIsIndvcmtmbG93X3dvcmtmbG93X2RlbGV0ZV9hbnkiLCJ3b3JrZmxvd193b3JrZmxvd19yZWFkIiwid29ya2Zsb3dfd29ya2Zsb3dfcmVhZF9hbnkiLCJ3b3JrZmxvd193b3JrZmxvd191cGRhdGUiLCJ3b3JrZmxvd193b3JrZmxvd191cGRhdGVfYW55IiwiZGF0YXNldF9jbGFzc19jb2RlX21hcF9jcmVhdGUiLCJkYXRhc2V0X2NsYXNzX2NvZGVfbWFwX2RlbGV0ZSIsImRhdGFzZXRfY2xhc3NfY29kZV9tYXBfZGVsZXRlX2FueSIsImRhdGFzZXRfY2xhc3NfY29kZV9tYXBfcmVhZCIsImRhdGFzZXRfY2xhc3NfY29kZV9tYXBfcmVhZF9hbnkiLCJkYXRhc2V0X2NsYXNzX2NvZGVfbWFwX3VwZGF0ZSIsImRhdGFzZXRfY2xhc3NfY29kZV9tYXBfdXBkYXRlX2FueSIsInB1YmxpY19zZXR0aW5nc19jcmVhdGUiLCJwdWJsaWNfc2V0dGluZ3NfZGVsZXRlIiwicHVibGljX3NldHRpbmdzX2RlbGV0ZV9hbnkiLCJwdWJsaWNfc2V0dGluZ3NfcmVhZCIsInB1YmxpY19zZXR0aW5nc19yZWFkX2FueSIsInB1YmxpY19zZXR0aW5nc191cGRhdGUiLCJwdWJsaWNfc2V0dGluZ3NfdXBkYXRlX2FueSIsIndvcmtmbG93X3dvcmtmbG93X2hpc3RvcnlfY3JlYXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfaGlzdG9yeV9kZWxldGUiLCJ3b3JrZmxvd193b3JrZmxvd19oaXN0b3J5X2RlbGV0ZV9hbnkiLCJ3b3JrZmxvd193b3JrZmxvd19oaXN0b3J5X3JlYWQiLCJ3b3JrZmxvd193b3JrZmxvd19oaXN0b3J5X3JlYWRfYW55Iiwid29ya2Zsb3dfd29ya2Zsb3dfaGlzdG9yeV91cGRhdGUiLCJ3b3JrZmxvd193b3JrZmxvd19oaXN0b3J5X3VwZGF0ZV9hbnkiXSwiZ3JvdXBfc3lzdGVtcyI6WyJsb2NhbCIsIm1pcmVyby1hZGMtYWxsLWluLW9uZSJdLCJncm91cF9mZWF0dXJlcyI6WyJhZGNfdGVzdF91c2VyX21hbmFnZW1lbnQiLCJhZGNfaW1hZ2VfYW5hbHl6ZSIsImFkY19zeXN0ZW1fc2VydmVyIiwiYWRjX3JlY2lwZV9jcmVhdG9yIiwiYWRjX3BhcmFtZXRlcl9tYW5hZ2VtZW50IiwiYWRjX2xvYWRlcl9zY2hlZHVsZV9tYW5hZ2VtZW50IiwiYWRjX2xvYWRlcl90eXBlX21hbmFnZW1lbnQiLCJhZGNfcmVjaXBlX21hbmFnZW1lbnQiLCJhZGNfaW5saW5lX2RhdGEiLCJhZGNfY2xhc3NfY29kZV9zZXQiLCJhZGNfZGF0YXNldCIsImFkY19tb2RlbCIsImFkY19tYXN0ZXJkYXRhX21hbmFnZW1lbnQiLCJhZGNfdXNlcl9tYW5hZ2VtZW50IiwicmNfY2xhc3NfY29kZV9tYXBfbWFuYWdlbWVudCIsInJjX2RhdGFzZXRfbWFuYWdlbWVudCIsInJjX2RlZmVjdF9jb2xsZWN0aW9uIiwicmNfbWFjaGluZV9sZWFybmluZ19jbGFzc19jb2RlIiwicmNfbWFjaGluZV9sZWFybmluZ19tb2RlbF9tYW5hZ2VtZW50IiwicmNfbWFzdGVyZGF0YXNldF9tYW5hZ2VtZW50IiwicmNfb3BlcmF0aW9uX2RlZmVjdF9kYXRhX21hbmFnZW1lbnQiLCJyY19vcGVyYXRpb25fbW9uaXRvcmluZyIsInJjX29wZXJhdGlvbl9zY2hlZHVsZXJfbWFuYWdlbWVudCIsInJjX29wZXJhdGlvbl9zdGF0dXMiLCJyY19yZWNpcGVfZGV2ZWxvcG1lbnQiLCJyY19yZWNpcGVfaW1hZ2VfYW5hbHl6ZXIiLCJyY19yZWNpcGVfbWFuYWdlbWVudCIsInJjX3dvcmtmbG93X21hbmFnZW1lbnQiLCJyY193b3JrZmxvd19tb25pdG9yaW5nIl0sImp0aSI6ImYzNDZmNmFlLWViMTgtNGY0My05ZDFiLWI5ODNlM2Q5ZWU5ZCIsImlhdCI6MTcyNTQ5Njc1MCwibmJmIjoxNzI1NDk2NzUwLCJleHAiOjE3MjY3MDYzNTAsImlzcyI6ImRhcSIsImF1ZCI6ImRhcV91c2VyIn0.uhVOFSoJMf4a9_aMd2MAW2QH5g1SFNksF0kz1fgDNHk"

    RecipeRun(**kwargs)