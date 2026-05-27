import os
import mpp
from mpp.daq import protos
from google.protobuf.wrappers_pb2 import StringValue

class DAQ_Classification_ClassCodeBuilder:
    def __init__(self):
        self.__label_info = None
        self.__class_code_info = None
        self.__num_classes = None
        
        self.__operation_channel = None
        self.__access_token = None
    
    def get_label_info(self):
        return self.__label_info
    
    def get_class_code_info(self):
        return self.__class_code_info

    def get_class_count(self):
        return self.__num_classes
    
    def init_url_info(self, operation_channel, access_token):
        self.__operation_channel = operation_channel
        self.__access_token = access_token
        return self
    
    def init_label_data(self, gt_dataset_id):
        class_code_set_id = self._fetch_class_code_set_id(gt_dataset_id)
        label_info, class_code_info, num_classes = self._build_label_info(class_code_set_id)
        
        self.__label_info = label_info
        self.__class_code_info = class_code_info
        self.__num_classes = num_classes
        return self

    def _fetch_class_code_set_id(self, gt_dataset_id):
        if not self.__operation_channel:
            return gt_dataset_id
        stub = protos.daq_dataset_classification_gt_dataset_api_v1_pb2_grpc.ClassificationGtDatasetServiceStub(self.__operation_channel)
        classification_gt_dataset = stub.GetClassificationGtDataset(
            request=protos.daq_dataset_classification_gt_dataset_api_v1_pb2.GetClassificationGtDatasetRequest(id=gt_dataset_id),
            metadata=[('authorization', f'Bearer {self.__access_token}')])
        return classification_gt_dataset.class_code_set_id

    def _build_label_info(self, class_code_set_id):
        if self.__operation_channel:
            return self._build_label_info_from_api(class_code_set_id)
        return self._build_label_info_from_local(class_code_set_id)

    def _build_label_info_from_api(self, class_code_set_id):
        stub = protos.daq_dataset_class_code_api_v1_pb2_grpc.ClassCodeServiceStub(self.__operation_channel)
        query_parameter = protos.daq_common_pb2.QueryParameter(
            page_index=0, page_size=-1,
            where=StringValue(value=f"Id=\"{class_code_set_id}\""),
            order_by=None)
        response = stub.ListClassCodeSets(
            request=protos.daq_dataset_class_code_api_v1_pb2.ListClassCodeSetsRequest(query_parameter=query_parameter),
            metadata=[('authorization', f'Bearer {self.__access_token}')])

        class_info = response.class_code_sets[0].class_codes
        class_info = sorted(class_info, key=lambda c: int(c.code))
        num_classes = len(class_info)
        label_info = {"label_count": num_classes}
        class_code_info = {class_code.code: i for i, class_code in enumerate(class_info)}

        for i, class_code in enumerate(class_info):
            label_info[f'label_{i}'] = {"code": class_code.code, "name": class_code.name}

        return label_info, class_code_info, num_classes

    def _build_label_info_from_local(self, class_code_set_id):
        class_info = os.listdir(class_code_set_id)
        num_classes = len(class_info)
        label_info = {"label_count": num_classes}
        class_code_info = {i: i for i in range(num_classes)}

        for i, class_code in enumerate(class_info):
            label_info[f'label_{i}'] = {"code": i, "name": class_code}

        return label_info, class_code_info, num_classes
    
    def build(self):
        return self