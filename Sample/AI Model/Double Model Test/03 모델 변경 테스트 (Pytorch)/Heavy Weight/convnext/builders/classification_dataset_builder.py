import os
import uuid
import random
import shutil
import logging
import torch
from torch.utils.data import Dataset
from google.protobuf.wrappers_pb2 import StringValue
import mpp
from mpp.daq import protos

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)

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
        
class Classification_Dataset_Builder:
    def __init__(self, keyword_arguments):
        self.__gt_dataset_id = keyword_arguments['gt_dataset_id']
        self.__local_download_path = None
        self.__train_data_loader = None
        self.__validation_data_loader = None
        self.__classification_gts = None
        self.__label_info = None
        self.__class_code_info = None
        self.__num_classes = None

    def initialize(self):
        self.__local_download_path = os.path.join(f"/temp/{uuid.uuid4()}")
        return self

    def init_label_data(self, operation_channel, access_token):
        if operation_channel:
            stub = protos.daq_dataset_classification_gt_dataset_api_v1_pb2_grpc.ClassificationGtDatasetServiceStub(operation_channel)
            classification_gt_dataset = stub.GetClassificationGtDataset(request=protos.daq_dataset_classification_gt_dataset_api_v1_pb2.GetClassificationGtDatasetRequest(
                id=self.__gt_dataset_id), metadata=[('authorization', f'Bearer {access_token}')])
            class_code_set_id = classification_gt_dataset.class_code_set_id
        else: 
            class_code_set_id = self.__gt_dataset_id

        if operation_channel: 
            stub = protos.daq_dataset_class_code_api_v1_pb2_grpc.ClassCodeServiceStub(operation_channel)
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
        
        self.__label_info = label_info
        self.__class_code_info = class_code_info
        self.__num_classes = num_classes
        return self
    
    def init_dataset_gts(self, operation_channel, access_token):
        if operation_channel: 
            stub = protos.daq_dataset_classification_gt_dataset_api_v1_pb2_grpc.ClassificationGtDatasetServiceStub(operation_channel)
            query_parameter = protos.daq_common_pb2.QueryParameter(
                                page_index=0,
                                page_size=-1,
                                where=StringValue(value=f"GtDatasetId=\"{self.__gt_dataset_id}\""),
                                order_by=None)

            response = stub.ListClassificationGts(request=protos.daq_dataset_classification_gt_dataset_api_v1_pb2.ListClassificationGtsRequest(
                query_parameter=query_parameter, with_image=False), metadata=[('authorization', f'Bearer {access_token}')])

            self.__classification_gts = response.classification_gts
        else: 
            self.__classification_gts = self.__gt_dataset_id
        
        return self

    def create_train_dataset(self, operation_channel, access_token, train_ratio, transform, batch_size, validation_save_random):
        logger.info("Create Train Dataset")
        try:
            train_data_info, validation_data_info = self.data_download(train_ratio,
                                                                       self.__local_download_path,
                                                                       self.__classification_gts,
                                                                       self.__class_code_info,
                                                                       operation_channel,
                                                                       access_token)

            train_dataset = ClassificationDataset(train_data_info, transform)
            train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size,
                                                            shuffle=True,
                                                            num_workers=0,
                                                            drop_last=True)

            valid_flag = False
            if len(validation_data_info[0]) > 0:
                valid_flag = True
                validation_dataset = ClassificationDataset(validation_data_info, transform)
                validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                                     batch_size=1,
                                                                     shuffle=validation_save_random,
                                                                     num_workers=0,
                                                                     drop_last=False)

            self.__train_data_loader = train_data_loader
            self.__validation_data_loader = validation_data_loader if valid_flag else None

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
    
    def data_download(self, train_ratio, local_download_path, classification_gts, class_code_info, operation_channel, access_token):
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

    def get_train_loader(self):
        return self.__train_data_loader
    
    def get_validation_loader(self):
        return self.__validation_data_loader
    
    def temp_folder_delete(self):
        if os.path.exists(self.__local_download_path):
            shutil.rmtree(self.__local_download_path)
            logger.info("Temp Folder Delete")

    def build(self):
        return self.__train_data_loader is not None and self.__validation_data_loader is not None 