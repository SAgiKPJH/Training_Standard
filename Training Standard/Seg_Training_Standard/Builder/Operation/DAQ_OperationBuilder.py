import json
import random
import grpc
import torch
from urllib.parse import urlparse
from mpp.daq import protos
from google.protobuf.wrappers_pb2 import StringValue

DEFAULT_COMPRESSION = grpc.Compression.Gzip
DEFAULT_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 2147483647),
    ("grpc.max_receive_message_length", 2147483647),
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 30_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.http2.min_ping_interval_without_data_ms", 5_000)
]

class Operation_Builder:
    def __init__(self, **keyword_arguments):
        self.__keyword_arguments = keyword_arguments
        self.__operation_channel = None
        self.__access_token = None
        self.__bucket_url = None
        self.__chunk_size = None
        self.__device = None
        
    def initialize(self):
        self.__init_access_token()
        self.__init_operation_channel()
        self.__init_bucket_url()
        self.__init_chunk_size()
        return self

    def __init_operation_channel(self):
        address = self.__keyword_arguments['authentication']['operation_service_address']
        selected_address = random.choice(address.split(","))
        if not address or not selected_address:
            raise Exception("Address is required")
        self.__operation_channel = grpc.insecure_channel(selected_address,
                                                         options=DEFAULT_GRPC_OPTIONS,
                                                         compression=DEFAULT_COMPRESSION)
    
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

    def get_keyword_arguments(self):
        return self.__keyword_arguments
    
    def get_gt_dataset_id(self):
        return self.__keyword_arguments['gt_dataset']['gt_dataset_id']

    def get_json_hyperparameter(self):
        return json.dumps(self.__keyword_arguments['hyperparameter'])

    def build(self):
        return self
