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
import grpc
import mpp
from mpp.daq import protos
from google.protobuf.wrappers_pb2 import StringValue

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import *
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback

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
        "train_ratio" : 0.8,

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
        "gt_dataset_id" : "HSegmentationTest_gt"
    },
    "chunk_size" : 100000
}'''
##$--

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def RecipeRun(**kwargs):

    operation_builder = Operation_Builder(**kwargs)
    operation_config = operation_builder \
            .initialize() \
            .build()
    
    hyperparameter_builder = HyperparameterBuilder(operation_config['hyperparameter']) \
            .initialize() \
            .build()
    
    dataset_builder = Segmentation_Dataset_Builder(operation_config['gt_dataset'])
    result = dataset_builder \
            .initialize() \
            .init_label_data(
                operation_channel= operation_builder.get_operation_channel(),
                access_token= operation_builder.get_access_token() 
            ) \
            .init_dataset_gts(
                operation_channel= operation_builder.get_operation_channel(),
                access_token= operation_builder.get_access_token()
            ) \
            .create_train_dataset(
                operation_channel= operation_builder.get_operation_channel(),
                access_token= operation_builder.get_access_token(),
                train_ratio= hyperparameter_builder.get_train_ratio(),
                transform= None, #hyperparameter_builder.get_transform(),
                batch_size= hyperparameter_builder.get_batch_size(),
                validation_save_random= hyperparameter_builder.get_validation_save_random(),
                input_size=hyperparameter_builder.get_input_size()
            ) \
            .build()

    if result is False:
        logger.error(f"Dataset Build Failed")
        dataset_builder.temp_folder_delete()
        return None
    
    try:

        save_builder = Save_Builder() \
            .initialize(
                operation_builder,
                num_classes=None, # dataset_builder.get_num_classes(),
                label_info=dataset_builder.get_label_info()
            ) \
            .build()
        
        model_builder = Keras_Model_Builder()
        model_builder \
            .initialize(
                epoch_total=hyperparameter_builder.get_epoch(),
                save_epoch=hyperparameter_builder.get_save_epoch(),
                save_builder=save_builder
            ) \
            .init_callback(
                earlyStopPatience=hyperparameter_builder.get_early_stop_patience(),
                reduceLRPatience=hyperparameter_builder.get_reduce_learning_rate_patience()
            ) \
            .init_optimizer(
                optimizer_name=hyperparameter_builder.get_optimizer(),
            ) \
            .init_criterion(
                criterion=hyperparameter_builder.get_criterion()
            ) \
            .init_model(
                device=operation_builder.get_device(),
                input_size=hyperparameter_builder.get_input_size(),
                num_classes=dataset_builder.get_num_classes(),
            ) \
            .build()
        
        model_builder.train(
            train_data=dataset_builder.get_train_loader(),
            valid_data=dataset_builder.get_validation_loader(),
            device=operation_builder.get_device(),
            using_amp=hyperparameter_builder.get_using_amp(),
            steps_per_epoch=dataset_builder.get_steps_per_epoch(),
            validation_steps=dataset_builder.get_validation_steps()
        )
    
    except Exception as e:
        logger.error(f"Error Message : {e}")
        raise Exception(f"Train Failed, Error Message : {e}")

    finally:
        dataset_builder.temp_folder_delete()

class Operation_Builder:
    def __init__(self, **keyword_arguments):
        self.__keyword_arguments = keyword_arguments
        self.__operation_channel = None
        self.__access_token = None
        self.__bucket_url = None
        self.__chunk_size = None
        self.__device = None
            
    def initialize(self):
        self.__init_device()
        self.__init_access_token()
        self.__init_operation_channel()
        self.__init_bucket_url()
        self.__init_chunk_size()
        return self

    def __init_device(self):
        use_gpu = self.__keyword_arguments['hyperparameter']['using_gpu']
        if use_gpu and not tf.config.list_physical_devices('GPU'):
            raise Exception("GPU is not available")
        self.__device = 'cuda' if use_gpu else 'cpu'
        
    def __init_operation_channel(self):
        address = self.__keyword_arguments['authentication']['operation_service_address']
        selected_address = random.choice(address.split(","))
        if not address or not selected_address:
            raise Exception("Address is required")
        self.__operation_channel = grpc.insecure_channel(selected_address)
    
    def __init_access_token(self):
        access_token = self.__keyword_arguments['authentication']['access_token']
        if not access_token:
            raise Exception("Access token is required")
        self.__access_token = access_token

    def __init_bucket_url(self):
        bucket_id = self.__keyword_arguments['result']['id']
        volume_id = self.__keyword_arguments['result']['volume_id']

        if not bucket_id:
            raise Exception("Bucket URL is required")
        if urlparse(bucket_id).scheme == '':
            self.__create_bucket(bucket_id, bucket_id+"_title", volume_id)
            self.__bucket_url = f"object:///{bucket_id}"
        else:
            self.__bucket_url = bucket_id

    def __create_bucket(self, bucket_id, bucket_title, bucket_volume_id, properties=None):
        if properties:
            properties = StringValue(value=json.dumps(properties))

        stub = protos.daq_object_object_api_v1_pb2_grpc.ObjectServiceStub(self.__operation_channel)
        stub.CreateBucket(request=protos.daq_object_object_api_v1_pb2.CreateBucketRequest(
            id=bucket_id, title=bucket_title, properties=properties, description=None, volume_id=bucket_volume_id
        ),metadata=[('authorization', f'Bearer {self.__access_token}')])

    def __init_chunk_size(self):
        chunk_size = self.__keyword_arguments['chunk_size']
        if not chunk_size:
            raise Exception("Chunk size is required")
        if chunk_size <= 0:
            raise Exception("Chunk size must be greater than 0")
        self.__chunk_size = chunk_size

    def get_operation_channel(self):
        return self.__operation_channel
    
    def get_access_token(self):
        return self.__access_token
    
    def get_device(self):
        return self.__device
    
    def get_bucket_url(self):
        return self.__bucket_url
    
    def get_chunk_size(self):
        return self.__chunk_size

    def build(self):
        return self.__keyword_arguments
    
class HyperparameterBuilder:
    def __init__(self, keyword_arguments):
        self.__hyperparams = keyword_arguments

    def initialize(self):
        self.__hyperparams['epoch'] = int(self.__hyperparams['epoch'])
        self.__hyperparams['batch_size'] = int(self.__hyperparams['batch_size'])
        return self

    def get_train_ratio(self):
        return self.__hyperparams['train_ratio']
    
    def get_transform(self):
        raise NotImplementedError("get_transform() is deprecated and should not be used.")

    def get_batch_size(self):
        return self.__hyperparams['batch_size']
    
    def get_validation_save_random(self):
        return self.__hyperparams.get('validation_save_random', False)
    
    def get_optimizer(self):
        keys = ['optimizer', 'optimizer_name']
        for key in keys:
            if key in self.__hyperparams:
                return self.__hyperparams[key]

        raise KeyError(f"Optimizer not found. Tried keys: {', '.join(keys)}")
        
    def get_learning_rate(self):
        keys = ['lr', 'learningRate', 'LearningRate', 'Learningrate']
        for key in keys:
            if key in self.__hyperparams:
                return self.__hyperparams[key]

        raise KeyError(f"Learning rate not found. Tried keys: {', '.join(keys)}")
    
    def get_input_size(self):
        return self.__hyperparams.get('input_size', 224)
    
    def get_epoch(self):
        return self.__hyperparams['epoch']
    
    def get_save_epoch(self):
        return self.__hyperparams.get('save_epoch', 1)
    
    def get_using_amp(self):
        return self.__hyperparams.get('using_amp', True)

    def get_early_stop_patience(self):
        keys = ['earlyStopPatience']
        for key in keys:
            if key in self.__hyperparams:
                return self.__hyperparams[key]
        return 0
    
    def get_reduce_learning_rate_patience(self):
        keys = ['reduceLRPatience']
        for key in keys:
            if key in self.__hyperparams:
                return self.__hyperparams[key]
        return 0
    
    def get_criterion(self):
        keys = ['criterion', 'loss']
        for key in keys:
            if key in self.__hyperparams:
                return self.__hyperparams[key]
        return 'sparse_categorical_crossentropy' #'binary_crossentropy'

    def build(self):
        return self

class Segmentation_Dataset_Builder:
    def __init__(self, keyword_arguments):
        self.__gt_dataset_id = keyword_arguments['gt_dataset_id']
        self.__local_download_path = None
        self.__train_data_loader = None
        self.__validation_data_loader = None
        self.__segmentation_gts = None
        self.__label_info = None
        self.__class_code_info = None
        self.__num_classes = None
        self.__steps_per_epoch = None
        self.__validation_steps = None

    def initialize(self):
        self.__local_download_path = os.path.join(f"/temp/{uuid.uuid4()}")
        return self

    def init_label_data(self, operation_channel, access_token):
        stub = protos.daq_dataset_segmentation_gt_dataset_api_v1_pb2_grpc.SegmentationGtDatasetServiceStub(operation_channel)
        segemntation_gt_dataset = stub.GetSegmentationGtDataset(request=protos.daq_dataset_segmentation_gt_dataset_api_v1_pb2.GetSegmentationGtDatasetRequest(
            id=self.__gt_dataset_id), metadata=[('authorization', f'Bearer {access_token}')])

        class_code_set_id = segemntation_gt_dataset.class_code_set_id

        stub = protos.daq_dataset_class_code_api_v1_pb2_grpc.ClassCodeServiceStub(operation_channel)
        query_parameter = protos.daq_common_pb2.QueryParameter(
            page_index=1,
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

        self.__label_info = label_info
        self.__class_code_info = class_code_info
        self.__num_classes = num_classes

        return self
    
    def init_dataset_gts(self, operation_channel, access_token):
        segmentation_gts_list = list()

        stub = protos.daq_dataset_segmentation_gt_dataset_api_v1_pb2_grpc.SegmentationGtDatasetServiceStub(operation_channel)
        query_parameter = protos.daq_common_pb2.QueryParameter(
            page_index= 1,
            page_size= -1,
            where=StringValue(value=f"GtDatasetId=\"{self.__gt_dataset_id}\" AND DataStatus=\"success\""),
            order_by=None)

        response = stub.ListSegmentationGts(request=protos.daq_dataset_segmentation_gt_dataset_api_v1_pb2.ListSegmentationGtsRequest(
            query_parameter=query_parameter, with_image=False), metadata=[('authorization', f'Bearer {access_token}')])

        segmentation_gts = response.segmentation_gts
        segmentation_gts_list.extend(segmentation_gts)
        
        self.__segmentation_gts = segmentation_gts_list

        return self

    def create_train_dataset(self, operation_channel, access_token, train_ratio, transform, batch_size, validation_save_random, input_size):
        logger.info("Create Train Dataset")
        try:
            src_path, gt_path = self.data_download(operation_channel,
                                               access_token,
                                               self.__segmentation_gts,
                                               self.__local_download_path)
            
            image_size = input_size
            seed = 111
            image_datagen = ImageDataGenerator(validation_split = 1 - train_ratio, rescale = 1/255)
            mask_datagen = ImageDataGenerator(validation_split=1 - train_ratio)

            image_generator = image_datagen.flow_from_directory(src_path, class_mode = None, seed=seed, target_size=(image_size,image_size),subset = 'training')
            val_image_generator = image_datagen.flow_from_directory(src_path, class_mode = None, seed=seed, target_size=(image_size,image_size),subset = 'validation')
            mask_generator = mask_datagen.flow_from_directory(gt_path, class_mode = None, seed=seed, target_size=(image_size,image_size),subset = 'training', color_mode='grayscale')
            val_mask_generator = mask_datagen.flow_from_directory(gt_path, class_mode = None, seed=seed, target_size=(image_size,image_size),subset = 'validation', color_mode='grayscale')
            
            self.__steps_per_epoch = len(image_generator)
            self.__validation_steps = len(val_image_generator)
            self.__train_data_loader = zip(image_generator, mask_generator)
            self.__validation_data_loader = zip(val_image_generator, val_mask_generator)

        except Exception as e:
            logger.error(f"Error Message : {e}")
            raise Exception(f"Create Train Dataset Failed, Error Message : {e}")

        return self
        
    def get_label_info(self):
        return self.__label_info
    
    def get_class_code_info(self):
        return self.__class_code_info

    def get_num_classes(self):
        return self.__num_classes
    
    def data_download(self, operation_channel, access_token, segmentation_gts, download_path):
        src_path = os.path.join(download_path, "source")
        gt_path = os.path.join(download_path, "ground_truth")

        for i in range(len(segmentation_gts)):
            image_id = segmentation_gts[i].image_id
            gt_id = segmentation_gts[i].id

            if segmentation_gts[i].data_status != 'success':
                continue

            src_uri = f"dataset:///?image_id={image_id}"
            gt_uri = f"dataset:///?image_id={gt_id}"

            source = mpp.daq.intel64.load(src_uri, False, channel=operation_channel, access_token=access_token)
            ground_truth = mpp.daq.dataset_service.load_segmentation_gt(gt_uri, True, channel=operation_channel, access_token=access_token)

            srcImage_path = os.path.join(src_path, "source", f"{i}.png")
            gtImage_path = os.path.join(gt_path, "ground_truth", f"{i}.png")

            mpp.intel64.save(source, srcImage_path)
            mpp.intel64.save(ground_truth, gtImage_path)

        return src_path, gt_path

    def get_train_loader(self):
        return self.__train_data_loader
    
    def get_validation_loader(self):
        return self.__validation_data_loader
    
    def temp_folder_delete(self):
        if os.path.exists(self.__local_download_path):
            shutil.rmtree(self.__local_download_path)
            logger.info("Temp Folder Delete")

    def get_steps_per_epoch(self):
        if self.__steps_per_epoch is None:
            raise ValueError("Steps per epoch not set. Please call create_train_dataset() first.")
        return self.__steps_per_epoch
    
    def get_validation_steps(self):
        if self.__validation_steps is None:
            raise ValueError("Validation steps not set. Please call create_train_dataset() first.")
        return self.__validation_steps

    def build(self):
        return self.__train_data_loader is not None and self.__validation_data_loader is not None

class Keras_Model_Builder:
    def __init__(self):
        self.__model = None
        self.__optimizer = None
        self.__criterion = None
        self.__iteration_start_time = None
        self.__epoch_start_time = None
        self.__epoch_total = None
        self.__save_epoch = None
        self.__callbacks = None
        self.__save_builder = None

    def initialize(self, epoch_total, save_epoch, save_builder):
        self.__epoch_total = int(epoch_total)
        self.__save_epoch = save_epoch
        self.__save_builder = save_builder
        return self
    
    def init_callback(self, earlyStopPatience, reduceLRPatience):
        ep = EarlyStopping(monitor='val_acc', patience = int(earlyStopPatience), verbose=1)
        lrs = ReduceLROnPlateau(monitor='val_loss', patience = int(reduceLRPatience), factor=0.5, verbose =1)
        lc = LambdaCallback(on_epoch_begin=self.on_epoch_begin, on_epoch_end=self.on_epoch_end)

        cb = []
        cb.append(lc)
        if int(reduceLRPatience) != 0:   cb.append(lrs)
        if int(earlyStopPatience) != 0:  cb.append(ep)

        self.__callbacks = cb

        return self
    
    def on_epoch_begin(self, epoch, logs):
        self.__epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs):
        logger.info(f"current epoch : '{epoch}'")
        train_loss = logs.get('loss', 100.0)
        valid_loss = logs.get('val_loss', 100.0)
        train_acc = logs.get('acc', 0.0)
        valid_acc = logs.get('val_acc', 0.0)
        self.monitoring(epoch + 1, valid_loss, train_loss)
        logger.info(logs)
        self.__save_builder.append_train_loss(train_loss)
        self.__save_builder.append_valid_loss(valid_loss)
        self.__save_builder.append_train_acc(train_acc)
        self.__save_builder.append_valid_acc(valid_acc)
        self.__save_builder.set_hierarchy_root(f"epoch_{epoch + 1}")
        self.__save_builder.save_train_valid_csv()
        self.__save_builder.save_training(self.__model, epoch + 1, self.__save_epoch, None)

    def init_model(self, device, input_size, num_classes):
        image_size = input_size
        model = self.efficient_unet(input_shape = (image_size, image_size, 3), num_classes=num_classes)
        model.compile(optimizer = self.__optimizer, loss = self.__criterion, metrics = ['acc'])

        self.__model = model
        return self

    def CBAR_block(self, input, num_filters) :
        x = keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same') (input)
        x = keras.layers.BatchNormalization() (x)
        x = keras.layers.Activation('relu') (x)

        x = keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same') (x)
        x = keras.layers.BatchNormalization() (x)
        x = keras.layers.Activation('relu') (x)

        xd = keras.layers.Conv2D(filters=num_filters, kernel_size =1) (input)
        x = keras.layers.Add() ([x,xd])

        return x

    def get_efficient(self, name='B3', input_shape=(None,None,1)) :
        return EfficientNetB3(
           include_top = False,
           weights = 'imagenet',
           input_tensor = None,
           input_shape = input_shape,
           pooling = None)

    def efficient_unet(self, input_shape = (224,224,3), num_classes=1) :
        encoder_model = self.get_efficient(name='B3', input_shape=input_shape)
        new_input = encoder_model.input
        encoder_output = encoder_model.get_layer(name='block7a_project_bn').output

        fn_bottle_neck = encoder_output.shape[-1]
        bottleneck = self.CBAR_block(encoder_output, fn_bottle_neck)

        c1 = encoder_model.get_layer(name= 'block5c_drop').output
        fn_1 = c1.shape[-1]

        upsamling1 = keras.layers.UpSampling2D() (bottleneck)
        concatenation1 = keras.layers.UpSampling2D() (bottleneck)
        concatenation1 = keras.layers.concatenate([upsamling1, c1], axis=3)
        decoder1 = self.CBAR_block(concatenation1, fn_1)

        c2 = encoder_model.get_layer(name = 'block3b_drop').output
        fn_2 = c2.shape[-1]

        upsampling2 = keras.layers.UpSampling2D() (decoder1)
        concatenation2 = keras.layers.concatenate([upsampling2, c2], axis=3)
        decoder2 = self.CBAR_block(concatenation2, fn_2)

        c3 = encoder_model.get_layer(name = 'block2b_drop').output
        fn_3 = c3.shape[-1]

        upsampling3 = keras.layers.UpSampling2D() (decoder2)
        concatenation3 = keras.layers.concatenate([upsampling3, c3], axis=3)
        decoder3 = self.CBAR_block(concatenation3, fn_3)

        c4 = encoder_model.get_layer(name = 'block1a_project_bn').output
        fn_4 = c4.shape[-1]

        upsampling4 = keras.layers.UpSampling2D() (decoder3)
        concatenation4 = keras.layers.concatenate([upsampling4, c4], axis=3)
        decoder4 = self.CBAR_block(concatenation4, fn_4)

        fn_5 = fn_4

        upsampling5 = keras.layers.UpSampling2D() (decoder4)
        concatenation5 = keras.layers.concatenate([upsampling5, new_input], axis=3)
        decoder5 = self.CBAR_block(concatenation5, fn_5)

        if num_classes ==1 or num_classes == 2 :
            final_filter_num = 1
            final_activation = 'sigmoid'
        else :
            final_filter_num = num_classes
            final_activation = 'softmax'

        new_output = keras.layers.Conv2D(filters = final_filter_num, kernel_size = 1, activation = final_activation) (decoder5)

        print("output shape", new_output.shape)
        efficient_unet = keras.Model(inputs = new_input, outputs = new_output)

        return efficient_unet
    
    def init_optimizer(self, optimizer_name):
        self.__optimizer = optimizer_name
        return self

    def init_criterion(self, criterion):
        self.__criterion = criterion
        return self
    
    def train(self, train_data, valid_data, device, using_amp, steps_per_epoch, validation_steps):
        self.__model.fit(
            train_data,
            validation_data = valid_data,
            epochs = self.__epoch_total,
            validation_steps = validation_steps,
            steps_per_epoch = steps_per_epoch,
            callbacks=self.__callbacks
        )

        return self
    
    def monitoring(self, epoch, valid_loss, train_loss):
        epoch_elapsed_time = time.time() - self.__epoch_start_time

        remaining_epochs = self.__epoch_total - epoch
        estimated_time_per_epoch = epoch_elapsed_time if epoch > 1 else 0  # 첫 번째 epoch의 경우 시간을 0으로 설정
        estimated_remaining_time = remaining_epochs * estimated_time_per_epoch
        logger.info("MonitoringData:"
            f"Epoch:[{epoch:4d}/{self.__epoch_total:4d}], "
            f"Train Loss: {train_loss:4.4f}, "
            f"Valid Loss : {valid_loss:4.4f}, "
            f"Time: {epoch_elapsed_time:4.2f}s, "
            f"Estimated Remaining Time: {estimated_remaining_time / 60:.2f} minutes")
        
    def build(self):
        return self.__model

class Save_Builder:
    def __init__(self):
        self.__train_loss_list = list()
        self.__valid_loss_list = list()
        self.__train_acc_list = list()
        self.__valid_acc_list = list()
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
    
    def append_train_loss(self, loss):
        self.__train_loss_list.append(loss)
        return self
    
    def append_valid_loss(self, loss):
        self.__valid_loss_list.append(loss)
        return self

    def append_train_acc(self, loss):
        self.__train_acc_list.append(loss)
        return self
    
    def append_valid_acc(self, loss):
        self.__valid_acc_list.append(loss)
        return self
    
    def set_hierarchy_root(self, hierarchy_root):
        self.__hierarchy_root = hierarchy_root
        return self
    
    def save_training(self, model, epoch, save_epoch, inputs):
        # train_loss_image = self.score_list_graph_image(epoch, self.__train_loss_list, self.__train_loss_list[0], "Train Loss Graph", 'r')
        # self.save_file(train_loss_image, "train_loss/train_loss_image.png")
    
        # train_loss_csv = self.score_list_csv(self.__train_loss_list)
        # self.save_csv(train_loss_csv, "train_loss/train_loss_csv.csv")
    
        if epoch % save_epoch == 0 or epoch == epoch:
            self.upload_model(model, "model/model.h5", inputs)

    #def save_validateion(self, epoch, label_list, predict_list):
    #    valid_loss_image = self.score_list_graph_image(epoch, self.__valid_loss_list, self.__valid_loss_list[0], "Valid Loss Graph", 'g')
    #    self.save_file(valid_loss_image, "valid_loss/valid_loss_graph.png")
    #
    #    valid_loss_csv = self.score_list_csv(self.__valid_loss_list)
    #    self.save_csv(valid_loss_csv, "valid_loss/valid_loss_csv.csv")
    #
    #    confusion_matrix = self.confusion_matrix_image(label_list, predict_list, labels=[self.__label_info[f'label_{i}']['name'] for i in range(self.__num_classes)])
    #    self.save_file(confusion_matrix, "confusion_matrix/confusion_matrix.png")

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
            train_acc = self.__train_acc_list[epoch] if epoch < len(self.__train_acc_list) else None

            if has_valid:
                valid_loss = self.__valid_loss_list[epoch] if epoch < len(self.__valid_loss_list) else None
                valid_acc = self.__valid_acc_list[epoch] if epoch < len(self.__valid_acc_list) else None
                result.append([epoch + 1, train_loss, valid_loss, train_acc, valid_acc])
            else:
                result.append([epoch + 1, train_loss, train_acc])

        return result

    def save_csv(self, csv, path):
        save_uri = f"{self.__operation_builder.get_bucket_url()}/{self.__hierarchy_root}/{path}"
        mpp.intel64.save_csv(csv, save_uri, channel=self.__operation_builder.get_operation_channel(), access_token=self.__operation_builder.get_access_token(), chunk_size=self.__operation_builder.get_chunk_size())
        return self
    
    # def save_file(self, file, path):
    #     save_uri = f"{self.__operation_builder.get_bucket_url()}/{self.__hierarchy_root}/{path}"
    #     mpp.intel64.save(file, save_uri, channel=self.__operation_builder.get_operation_channel(), access_token=self.__operation_builder.get_access_token(), chunk_size=self.__operation_builder.get_chunk_size())
    #     return self

    def upload_model(self, model, path, inputs):
        model_save_uri = f"{self.__operation_builder.get_bucket_url()}/{self.__hierarchy_root}/{path}"
        mpp.daq.object_service.upload_model(model, uri=model_save_uri, inference_info=self.__inference_info, example=inputs, channel=self.__operation_builder.get_operation_channel(), access_token=self.__operation_builder.get_access_token(), chunk_size=self.__operation_builder.get_chunk_size())
        return self
    
    def build(self):
        return self

if __name__ == '__main__':
    kwargs = json.loads(parameters)
    kwargs['authentication']['operation_service_address'] = "192.168.70.62:5020"
    kwargs['authentication']['access_token'] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJtaW53b29uZy5wYXJrIiwibmFtZSI6IuuwleuvvOybhSIsInJvbGUiOiJkZXZlbG9wZXIiLCJncm91cHNpZCI6ImRlZmF1bHQiLCJsb2dfaW5fcHJvdmlkZXIiOiJkYXEiLCJwcml2aWxlZ2VzIjpbImRhdGFzZXRfY2xhc3NfY29kZV9jcmVhdGUiLCJkYXRhc2V0X2NsYXNzX2NvZGVfZGVsZXRlIiwiZGF0YXNldF9jbGFzc19jb2RlX3JlYWRfYW55IiwiZGF0YXNldF9jbGFzc19jb2RlX3VwZGF0ZSIsImRhdGFzZXRfZ3RfZGF0YXNldF9jcmVhdGUiLCJkYXRhc2V0X2d0X2RhdGFzZXRfZGVsZXRlIiwiZGF0YXNldF9ndF9kYXRhc2V0X3JlYWRfYW55IiwiZGF0YXNldF9ndF9kYXRhc2V0X3VwZGF0ZSIsImRhdGFzZXRfaW1hZ2VfZGF0YXNldF9jcmVhdGUiLCJkYXRhc2V0X2ltYWdlX2RhdGFzZXRfZGVsZXRlIiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X3JlYWRfYW55IiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X3VwZGF0ZSIsImRhdGFzZXRfdm9sdW1lX2NyZWF0ZSIsImRhdGFzZXRfdm9sdW1lX2RlbGV0ZSIsImRhdGFzZXRfdm9sdW1lX3JlYWRfYW55IiwiZGF0YXNldF92b2x1bWVfdXBkYXRlIiwiZGVmZWN0X2RhdGFfbWFuYWdlcl9tYXN0ZXJfaW5mb3JtYXRpb25fY3JlYXRlIiwiZGVmZWN0X2RhdGFfbWFuYWdlcl9tYXN0ZXJfaW5mb3JtYXRpb25fZGVsZXRlIiwiZGVmZWN0X2RhdGFfbWFuYWdlcl9tYXN0ZXJfaW5mb3JtYXRpb25fcmVhZF9hbnkiLCJkZWZlY3RfZGF0YV9tYW5hZ2VyX21hc3Rlcl9pbmZvcm1hdGlvbl91cGRhdGUiLCJkZWZlY3RfZGVmZWN0X2RhdGFfY3JlYXRlIiwiZGVmZWN0X2RlZmVjdF9kYXRhX2RlbGV0ZSIsImRlZmVjdF9kZWZlY3RfZGF0YV9yZWFkX2FueSIsImRlZmVjdF9kZWZlY3RfZGF0YV91cGRhdGUiLCJkZWZlY3RfaWRic19tYXN0ZXJfaW5mb3JtYXRpb25fY3JlYXRlIiwiZGVmZWN0X2lkYnNfbWFzdGVyX2luZm9ybWF0aW9uX2RlbGV0ZSIsImRlZmVjdF9pZGJzX21hc3Rlcl9pbmZvcm1hdGlvbl9yZWFkX2FueSIsImRlZmVjdF9pZGJzX21hc3Rlcl9pbmZvcm1hdGlvbl91cGRhdGUiLCJnZHNfY2xpcF9jcmVhdGUiLCJnZHNfY2xpcF9kZWxldGUiLCJnZHNfY2xpcF9yZWFkX2FueSIsImdkc19jbGlwX3VwZGF0ZSIsImdkc19leHBvcnRfY3JlYXRlIiwiZ2RzX2V4cG9ydF9kZWxldGUiLCJnZHNfZXhwb3J0X3JlYWRfYW55IiwiZ2RzX2V4cG9ydF91cGRhdGUiLCJnZHNfZ2RzX2NyZWF0ZSIsImdkc19nZHNfZGVsZXRlIiwiZ2RzX2dkc19yZWFkX2FueSIsImdkc19nZHNfdXBkYXRlIiwiZ2RzX3NlcnZlcl9jcmVhdGUiLCJnZHNfc2VydmVyX2RlbGV0ZSIsImdkc19zZXJ2ZXJfcmVhZF9hbnkiLCJnZHNfc2VydmVyX3VwZGF0ZSIsImdkc192b2x1bWVfY3JlYXRlIiwiZ2RzX3ZvbHVtZV9kZWxldGUiLCJnZHNfdm9sdW1lX3JlYWRfYW55IiwiZ2RzX3ZvbHVtZV91cGRhdGUiLCJpbmZlcmVuY2VfaW5mZXJlbmNlX2NyZWF0ZSIsImluZmVyZW5jZV9pbmZlcmVuY2VfZGVsZXRlIiwiaW5mZXJlbmNlX2luZmVyZW5jZV9yZWFkX2FueSIsImluZmVyZW5jZV9pbmZlcmVuY2VfdXBkYXRlIiwiaW5mZXJlbmNlX21vZGVsX2NyZWF0ZSIsImluZmVyZW5jZV9tb2RlbF9kZWxldGUiLCJpbmZlcmVuY2VfbW9kZWxfcmVhZF9hbnkiLCJpbmZlcmVuY2VfbW9kZWxfdXBkYXRlIiwiaW5mZXJlbmNlX3NlcnZlcl9jcmVhdGUiLCJpbmZlcmVuY2Vfc2VydmVyX2RlbGV0ZSIsImluZmVyZW5jZV9zZXJ2ZXJfcmVhZF9hbnkiLCJpbmZlcmVuY2Vfc2VydmVyX3VwZGF0ZSIsImluZmVyZW5jZV92b2x1bWVfY3JlYXRlIiwiaW5mZXJlbmNlX3ZvbHVtZV9kZWxldGUiLCJpbmZlcmVuY2Vfdm9sdW1lX3JlYWRfYW55IiwiaW5mZXJlbmNlX3ZvbHVtZV91cGRhdGUiLCJvYmplY3Rfb2JqZWN0X2NyZWF0ZSIsIm9iamVjdF9vYmplY3RfZGVsZXRlIiwib2JqZWN0X29iamVjdF9yZWFkX2FueSIsIm9iamVjdF9vYmplY3RfdXBkYXRlIiwib2JqZWN0X3ZvbHVtZV9jcmVhdGUiLCJvYmplY3Rfdm9sdW1lX2RlbGV0ZSIsIm9iamVjdF92b2x1bWVfcmVhZF9hbnkiLCJvYmplY3Rfdm9sdW1lX3VwZGF0ZSIsInNjaGVkdWxlcl9jb25kaXRpb25fcmVhZF9hbnkiLCJ1cGRhdGVfbXBwX2NyZWF0ZSIsInVwZGF0ZV9tcHBfZGVsZXRlIiwidXBkYXRlX21wcF9yZWFkX2FueSIsInVwZGF0ZV9tcHBfdXBkYXRlIiwidXBkYXRlX3JjX2NyZWF0ZSIsInVwZGF0ZV9yY19kZWxldGUiLCJ1cGRhdGVfcmNfcmVhZF9hbnkiLCJ1cGRhdGVfcmNfdXBkYXRlIiwid29ya2Zsb3dfam9iX2NyZWF0ZSIsIndvcmtmbG93X2pvYl9kZWxldGUiLCJ3b3JrZmxvd19qb2JfcmVhZF9hbnkiLCJ3b3JrZmxvd19qb2JfdXBkYXRlIiwid29ya2Zsb3dfc2VydmVyX2NyZWF0ZSIsIndvcmtmbG93X3NlcnZlcl9kZWxldGUiLCJ3b3JrZmxvd19zZXJ2ZXJfcmVhZF9hbnkiLCJ3b3JrZmxvd19zZXJ2ZXJfdXBkYXRlIiwid29ya2Zsb3dfdm9sdW1lX2NyZWF0ZSIsIndvcmtmbG93X3ZvbHVtZV9kZWxldGUiLCJ3b3JrZmxvd192b2x1bWVfcmVhZF9hbnkiLCJ3b3JrZmxvd192b2x1bWVfdXBkYXRlIiwid29ya2Zsb3dfd29ya2VyX2NyZWF0ZSIsIndvcmtmbG93X3dvcmtlcl9kZWxldGUiLCJ3b3JrZmxvd193b3JrZXJfcmVhZF9hbnkiLCJ3b3JrZmxvd193b3JrZXJfdXBkYXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfY3JlYXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfZGVsZXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfcmVhZF9hbnkiLCJ3b3JrZmxvd193b3JrZmxvd191cGRhdGUiLCJkYXRhc2V0X2NsYXNzaWZpY2F0aW9uX2d0X2RhdGFzZXRfY3JlYXRlIiwiZGF0YXNldF9jbGFzc2lmaWNhdGlvbl9ndF9kYXRhc2V0X2RlbGV0ZSIsImRhdGFzZXRfY2xhc3NpZmljYXRpb25fZ3RfZGF0YXNldF9yZWFkX2FueSIsImRhdGFzZXRfY2xhc3NpZmljYXRpb25fZ3RfZGF0YXNldF91cGRhdGUiLCJkYXRhc2V0X3NlZ21lbnRhdGlvbl9ndF9kYXRhc2V0X2NyZWF0ZSIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfZGVsZXRlIiwiZGF0YXNldF9zZWdtZW50YXRpb25fZ3RfZGF0YXNldF9yZWFkX2FueSIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfdXBkYXRlIl0sImdyb3VwX3N5c3RlbXMiOlsibG9jYWwiLCJtaXJlcm8taWRjLWluLWxpbmUiLCJtaXJlcm8taWRjLW9mZi1saW5lIl0sImdyb3VwX2ZlYXR1cmVzIjpbInJjX2RhdGFzZXRfbWFuYWdlbWVudCIsInJjX2dkc19mbG9vcl9wbGFuX21hbmFnZW1lbnQiLCJyY19nZHNfbWFuYWdlbWVudCIsInJjX21hY2hpbmVfbGVhcm5pbmdfY2xhc3NfY29kZSIsInJjX21hY2hpbmVfbGVhcm5pbmdfbW9kZWxfbWFuYWdlbWVudCIsInJjX29wZXJhdGlvbl9kZWZlY3RfZGF0YV9tYW5hZ2VtZW50IiwicmNfcmVjaXBlX2RldmVsb3BtZW50IiwicmNfcmVjaXBlX2ltYWdlX2FuYWx5emVyIiwicmNfd29ya2Zsb3dfbWFuYWdlbWVudCIsInJjX3dvcmtmbG93X21vbml0b3JpbmciLCJyY19vcGVyYXRpb25fc2NoZWR1bGVyX21hbmFnZW1lbnQiLCJyY19vcGVyYXRpb25fc3RhdHVzIiwicmNfbWFzdGVyZGF0YXNldF9tYW5hZ2VtZW50Il0sImp0aSI6IjA2NjkwM2JmLTc5NzYtNDE5MS05NGI1LWU5ZGM0YWVjNzI5MSIsImlhdCI6MTY4NzgyMTg3NiwibmJmIjoxNjg3ODIxODc2LCJleHAiOjE2ODkwMzE0NzYsImlzcyI6ImRhcSIsImF1ZCI6ImRhcV91c2VyIn0.86YbTm0Yli6O1Qv0T_0-nIzHBMBz52klGZTjwRt6fCY"

    RecipeRun(**kwargs)
