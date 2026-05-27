import mpp
from mpp.daq import protos
from google.protobuf.wrappers_pb2 import StringValue
import json


class DAQ_Segmentation_ClassCodeBuilder:
    """DAQ Segmentation ClassCode Builder.
    Segmentation GT Dataset의 class_code_set_id에서 라벨 정보를 가져옵니다.
    """

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
        stub = protos.daq_dataset_segmentation_gt_dataset_api_v1_pb2_grpc.SegmentationGtDatasetServiceStub(self.__operation_channel)
        seg_gt_dataset = stub.GetSegmentationGtDataset(
            request=protos.daq_dataset_segmentation_gt_dataset_api_v1_pb2.GetSegmentationGtDatasetRequest(id=gt_dataset_id),
            metadata=[('authorization', f'Bearer {self.__access_token}')])
        return seg_gt_dataset.class_code_set_id

    def _build_label_info(self, class_code_set_id):
        stub = protos.daq_dataset_class_code_api_v1_pb2_grpc.ClassCodeServiceStub(self.__operation_channel)
        query_parameter = protos.daq_common_pb2.QueryParameter(
            page_index=0, page_size=-1,
            where=StringValue(value=f"Id=\"{class_code_set_id}\""),
            order_by=None)
        response = stub.ListClassCodeSets(
            request=protos.daq_dataset_class_code_api_v1_pb2.ListClassCodeSetsRequest(query_parameter=query_parameter),
            metadata=[('authorization', f'Bearer {self.__access_token}')])

        class_info = response.class_code_sets[0].class_codes

        # label_info.label 필드에서 mask pixel value를 추출, 정렬
        def _label_value(c):
            try:
                return int(json.loads(c.label_info.value)['label'])
            except Exception:
                try:
                    return int(c.code)
                except Exception:
                    return 0

        class_info = sorted(class_info, key=_label_value)
        num_classes = len(class_info)
        label_info = {"label_count": num_classes}
        class_code_info = {}
        for i, c in enumerate(class_info):
            label_val = _label_value(c)
            label_info[f'label_{i}'] = {"code": label_val, "name": c.name}
            class_code_info[label_val] = i

        return label_info, class_code_info, num_classes

    def build(self):
        return self
