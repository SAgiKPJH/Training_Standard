import json
import random
import grpc
import torch
from urllib.parse import urlparse
from mpp.daq import protos
from google.protobuf.wrappers_pb2 import StringValue

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
        device = self.__keyword_arguments['hyperparameter']['using_gpu']
        if device and not torch.cuda.is_available():
            raise Exception("GPU is not available")
        self.__device = 'cuda' if device else 'cpu'
        
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
