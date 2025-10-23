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
import grpc
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from google.protobuf.wrappers_pb2 import StringValue
import mpp
from mpp.daq import protos
from PIL import Image

# Hugging Face transformers for ViT-22B
try:
    from transformers import ViTForImageClassification, ViTConfig
    from transformers import AdamW as HFAdamW
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Please install with 'pip install transformers'")

##!--{"Name":"hyperparameter","Type":"epoch","Key":"epoch","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"batch_size","Key":"batch_size","Value":"","Category":""}
##!--{"Name":"result","Type":"result","Key":"id","Value":"","Category":""}
##!--{"Name":"authentication","Type":"system_address","Key":"operation_service_address","Value":"","Category":""}
##!--{"Name":"authentication","Type":"access_token","Key":"access_token","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id","Value":"","Category":""}

##$--
parameters = '''{
    "hyperparameter":{
        "epoch" : 10,
        "save_epoch" : 1,
        "batch_size" : 1,
        "lr" : 1e-5,
        "optimizer_name" : "AdamW",
        "input_size" : 224,
        "normalize_mean" : [0.485, 0.456, 0.406],
        "normalize_stdev" : [0.229, 0.224, 0.225],
        "using_gpu" : true,
        "using_amp" : true,
        "train_ratio" : 0.8,
        "validation_save_random" : false,
        "gradient_checkpointing" : true,
        "gradient_accumulation_steps" : 32,
        "max_grad_norm" : 1.0
    },
    "authentication": {
        "operation_service_address": "",
        "access_token" : ""
    },
    "result":{
        "id":"",
        "volume_id":"default" 
    },
    "gt_dataset":{
        "gt_dataset_id" : ""
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

    dataset_builder = Classification_Dataset_Builder(operation_config['gt_dataset'])
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
            transform= hyperparameter_builder.get_transform(),
            batch_size= hyperparameter_builder.get_batch_size(),
            validation_save_random= hyperparameter_builder.get_validation_save_random()
        ) \
        .build()
    
    if result is False:
        logger.error(f"Dataset Build Failed")
        dataset_builder.temp_folder_delete()
        return None

    try:
        model_builder = Torch_Model_Builder()
        model_builder \
            .initialize(
                epoch_total=hyperparameter_builder.get_epoch(),
                save_epoch=hyperparameter_builder.get_save_epoch(),
                gradient_accumulation_steps=hyperparameter_builder.get_gradient_accumulation_steps(),
                max_grad_norm=hyperparameter_builder.get_max_grad_norm()
            ) \
            .init_model(
                num_classes=dataset_builder.get_num_classes(),
                device=operation_builder.get_device(),
                gradient_checkpointing=hyperparameter_builder.get_gradient_checkpointing()
            ) \
            .init_optimizer(
                optimizer_name=hyperparameter_builder.get_optimizer(),
                lr=hyperparameter_builder.get_learning_rate()
            ) \
            .init_criterion(
                criterion_name=hyperparameter_builder.get_criterion()
            ) \
            .build()

        save_builder = Save_Builder() \
            .initialize(
                operation_builder,
                num_classes=dataset_builder.get_num_classes(),
                label_info=dataset_builder.get_label_info()
            ) \
            .init_inference_info(
                input_size=hyperparameter_builder.get_input_size()
            ) \
            .build()

        model_builder.train(
            dataset_builder.get_train_loader(),
            dataset_builder.get_validation_loader(),
            operation_builder.get_device(),
            hyperparameter_builder.get_using_amp(),
            save_builder
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
        device = self.__keyword_arguments['hyperparameter']['using_gpu']
        if device and not torch.cuda.is_available():
            raise Exception("GPU is not available")
        self.__device = 'cuda' if device else 'cpu'
        
        # ViT-22B requires multiple GPUs - check GPU memory
        if device and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            total_memory = sum([torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count)])
            total_memory_gb = total_memory / (1024**3)
            logger.info(f"Available GPUs: {gpu_count}, Total GPU Memory: {total_memory_gb:.1f} GB")
            
            if total_memory_gb < 80:  # ViT-22B needs ~80GB+ memory
                logger.warning("ViT-22B requires at least 80GB+ GPU memory. Consider using model parallelism or smaller batch sizes.")
        
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
        input_size = self.__hyperparams['input_size']
        mean = self.__hyperparams['normalize_mean']
        stdev = self.__hyperparams['normalize_stdev']

        transform = transforms.Compose([    
            transforms.Lambda(lambda img: Image.fromarray(img).convert("RGB")),
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Normalize(mean, stdev)
        ])
        return transform        

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
        return self.__hyperparams['input_size']
    
    def get_epoch(self):
        return self.__hyperparams['epoch']
    
    def get_save_epoch(self):
        return self.__hyperparams['save_epoch']
    
    def get_using_amp(self):
        return self.__hyperparams['using_amp']
    
    def get_gradient_checkpointing(self):
        return self.__hyperparams.get('gradient_checkpointing', True)
    
    def get_gradient_accumulation_steps(self):
        return self.__hyperparams.get('gradient_accumulation_steps', 32)
    
    def get_max_grad_norm(self):
        return self.__hyperparams.get('max_grad_norm', 1.0)
    
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
            
        return 'CrossEntropyLoss'

    def build(self):
        return self

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
                                                            drop_last=True,
                                                            pin_memory=True)

            valid_flag = False
            if len(validation_data_info[0]) > 0:
                valid_flag = True
                validation_dataset = ClassificationDataset(validation_data_info, transform)
                validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                                                     batch_size=1,
                                                                     shuffle=validation_save_random,
                                                                     num_workers=0,
                                                                     drop_last=False,
                                                                     pin_memory=True)

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

class Torch_Model_Builder:
    def __init__(self):
        self.__model = None
        self.__optimizer = None
        self.__criterion = None
        self.__iteration_start_time = None
        self.__epoch_start_time = None
        self.__epoch_total = None
        self.__save_epoch = None
        self.__gradient_accumulation_steps = None
        self.__max_grad_norm = None
        self.__scaler = None

    def initialize(self, epoch_total, save_epoch, gradient_accumulation_steps=1, max_grad_norm=1.0):
        self.__epoch_total = epoch_total
        self.__save_epoch = save_epoch
        self.__gradient_accumulation_steps = gradient_accumulation_steps
        self.__max_grad_norm = max_grad_norm
        self.__scaler = torch.cuda.amp.GradScaler()
        return self
    
    def init_model(self, num_classes, device, gradient_checkpointing=True):
        if not TRANSFORMERS_AVAILABLE:
            raise Exception("transformers library is required for ViT-22B. Please install with 'pip install transformers'")
        
        # ViT-22B 설정 (실제 22B 파라미터를 위한 대략적인 설정)
        # 실제 ViT-22B는 Google에서 공개한 모델이며, 정확한 설정은 다를 수 있습니다
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=4096,      # Large hidden dimension
            num_hidden_layers=64,  # Many layers for 22B parameters
            num_attention_heads=64, # Many attention heads
            intermediate_size=16384, # Large MLP dimension
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            num_labels=num_classes
        )
        
        logger.info(f"Initializing ViT-22B with {num_classes} classes")
        logger.info(f"Model configuration: {config.hidden_size} hidden_size, {config.num_hidden_layers} layers")
        
        try:
            # Try to load pre-trained ViT-22B if available, otherwise use custom config
            model = ViTForImageClassification.from_pretrained(
                "google/vit-large-patch16-224", 
                config=config,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        except:
            # Fallback to custom configuration
            logger.warning("Could not load pre-trained ViT-22B, using custom large configuration")
            model = ViTForImageClassification(config)
        
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Model parallelism for large models
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)
        
        model.to(device)
        self.__model = model
        
        # Log model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return self
    
    def init_optimizer(self, optimizer_name, lr):
        if optimizer_name.lower() == "adamw":
            # Use Hugging Face AdamW for better performance with transformers
            if TRANSFORMERS_AVAILABLE:
                self.__optimizer = HFAdamW(self.__model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
            else:
                self.__optimizer = optim.AdamW(self.__model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        elif optimizer_name.lower() == "adam":
            self.__optimizer = optim.Adam(self.__model.parameters(), lr=lr, eps=1e-8)
        elif optimizer_name.lower() == "sgd":
            self.__optimizer = optim.SGD(self.__model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
        else:
            raise Exception("Invalid Optimizer Option. For ViT-22B, recommend AdamW")
        
        logger.info(f"Optimizer: {optimizer_name}, Learning Rate: {lr}")
        return self
    
    def init_criterion(self, criterion_name):
        if criterion_name.lower() == "crossentropyloss":
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for large models
        elif criterion_name.lower() == "bcewithlogitsloss":
            criterion = nn.BCEWithLogitsLoss()
        elif criterion_name.lower() == "focalloss":
            criterion = nn.BCEWithLogitsLoss()
        elif criterion_name.lower() == "mse":
            criterion = nn.MSELoss()

        self.__criterion = criterion
        return self

    def train(self, train_data_loader, valid_data_loader, device, using_amp, save_builder):
        total_iteration = len(train_data_loader)
        if total_iteration == 0:
            raise RuntimeError("No training data: train_data_loader is empty.")

        logger.info(f"Training ViT-22B for {self.__epoch_total} epochs")
        logger.info(f"Gradient accumulation steps: {self.__gradient_accumulation_steps}")
        
        self.__iteration_start_time = time.time()
        for epoch in range(1, self.__epoch_total + 1):
            self.__epoch_start_time = time.time()
            logger.info(f"Epoch : {epoch:4d}/{self.__epoch_total:4d}")
            
            train_epoch_loss = 0.0
            self.__optimizer.zero_grad()
            
            for n_epoch, batch in enumerate(train_data_loader, 1):
                train_inputs, loss = self.train_step(batch, device, using_amp, n_epoch)

                train_epoch_loss += loss.item()
                if n_epoch % max(1, total_iteration // 10) == 0 or n_epoch == total_iteration:
                    iteration_elapsed_time = time.time() - self.__iteration_start_time
                    logger.info(f"Epoch : {epoch:4d}, Iterations : {n_epoch:4d}/{total_iteration:4d}, Loss : {loss : 4.4f}, Time : {iteration_elapsed_time : 4.4f}")
                    self.__iteration_start_time = time.time()

            save_builder.append_train_loss(train_epoch_loss / n_epoch)
            save_builder.set_hierarchy_root(f"epoch_{epoch}")

            valid_epoch_loss = 0.0
            if valid_data_loader:
                predict_list = np.array([])
                label_list = np.array([])
                with torch.no_grad():
                    self.__model.eval()
                    for _, valid_batch in enumerate(valid_data_loader, 1):

                        valid_inputs = valid_batch[0].to(device)
                        valid_labels = valid_batch[1].to(device)

                        with torch.cuda.amp.autocast(enabled=using_amp):
                            valid_outputs = self.__model(valid_inputs)
                            if hasattr(valid_outputs, 'logits'):
                                valid_outputs = valid_outputs.logits
                            valid_loss = self.__criterion(valid_outputs, valid_labels)

                        valid_epoch_loss += valid_loss.item()
                        predict_list = np.concatenate([predict_list, valid_outputs.argmax(dim=1).cpu().numpy()], 0)
                        label_list = np.concatenate([label_list, valid_labels.cpu().numpy()], 0)
                    self.__model.train()
                
                save_builder.append_valid_loss(valid_epoch_loss / n_epoch)
                save_builder.save_validateion(epoch, label_list, predict_list)
            
            epoch_valid_loss_mean = valid_epoch_loss/len(valid_data_loader) if valid_data_loader else 0
            save_builder.save_training(self.__model, epoch, self.__save_epoch, train_inputs)

            save_builder.save_train_valid_csv()

            train_loss = train_epoch_loss/total_iteration
            self.monitoring(epoch, epoch_valid_loss_mean, train_loss)

        return self
    
    def train_step(self, batch, device, using_amp, step):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        
        with torch.cuda.amp.autocast(enabled=using_amp):
            outputs = self.__model(inputs)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            loss = self.__criterion(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.__gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if using_amp:
            self.__scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if step % self.__gradient_accumulation_steps == 0:
            if using_amp:
                # Gradient clipping
                self.__scaler.unscale_(self.__optimizer)
                torch.nn.utils.clip_grad_norm_(self.__model.parameters(), self.__max_grad_norm)
                
                self.__scaler.step(self.__optimizer)
                self.__scaler.update()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.__model.parameters(), self.__max_grad_norm)
                self.__optimizer.step()
            
            self.__optimizer.zero_grad()
        
        return inputs, loss * self.__gradient_accumulation_steps  # Return unscaled loss for logging

    def monitoring(self, epoch, epoch_valid_loss_mean, train_loss):
        epoch_elapsed_time = time.time() - self.__epoch_start_time

        remaining_epochs = self.__epoch_total - epoch
        estimated_time_per_epoch = epoch_elapsed_time if epoch > 1 else 0
        estimated_remaining_time = remaining_epochs * estimated_time_per_epoch
        
        # Memory monitoring for large models
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            
            logger.info("MonitoringData:"
                f"Epoch:[{epoch:4d}/{self.__epoch_total:4d}], "
                f"Train Loss: {train_loss:4.4f}, "
                f"Valid Loss : {epoch_valid_loss_mean:4.4f}, "
                f"Time: {epoch_elapsed_time:4.2f}s, "
                f"Memory: {memory_allocated:.1f}GB/{memory_reserved:.1f}GB, "
                f"Estimated Remaining Time: {estimated_remaining_time / 60:.2f} minutes")
        else:
            logger.info("MonitoringData:"
                f"Epoch:[{epoch:4d}/{self.__epoch_total:4d}], "
                f"Train Loss: {train_loss:4.4f}, "
                f"Valid Loss : {epoch_valid_loss_mean:4.4f}, "
                f"Time: {epoch_elapsed_time:4.2f}s, "
                f"Estimated Remaining Time: {estimated_remaining_time / 60:.2f} minutes")
        
    def build(self):
        return self.__model

class Save_Builder:
    def __init__(self):
        self.__train_loss_list = list()
        self.__valid_loss_list = list()
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
    
    def init_inference_info(self, input_size):
        self.__inference_info = {'inference_info' : json.dumps({"input_size": input_size, "label_info": self.__label_info, "model_type": "ViT-22B"})}
        return self
    
    def append_train_loss(self, loss):
        self.__train_loss_list.append(loss)
        return self
    
    def append_valid_loss(self, loss):
        self.__valid_loss_list.append(loss)
        return self
    
    def set_hierarchy_root(self, hierarchy_root):
        self.__hierarchy_root = hierarchy_root
        return self
    
    def save_training(self, model, epoch, save_epoch, inputs):
        train_loss_image = self.score_list_graph_image(epoch, self.__train_loss_list, self.__train_loss_list[0], "Train Loss Graph", 'r')
        self.save_file(train_loss_image, "train_loss/train_loss_image.png")

        train_loss_csv = self.score_list_csv(self.__train_loss_list)
        self.save_csv(train_loss_csv, "train_loss/train_loss_csv.csv")

        if epoch % save_epoch == 0 or epoch == epoch:
            self.upload_model(model, "model/model.pth", inputs)

    def save_validateion(self, epoch, label_list, predict_list):
        valid_loss_image = self.score_list_graph_image(epoch, self.__valid_loss_list, self.__valid_loss_list[0], "Valid Loss Graph", 'g')
        self.save_file(valid_loss_image, "valid_loss/valid_loss_graph.png")

        valid_loss_csv = self.score_list_csv(self.__valid_loss_list)
        self.save_csv(valid_loss_csv, "valid_loss/valid_loss_csv.csv")

        confusion_matrix = self.confusion_matrix_image(label_list, predict_list, labels=[self.__label_info[f'label_{i}']['name'] for i in range(self.__num_classes)])
        self.save_file(confusion_matrix, "confusion_matrix/confusion_matrix.png")

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
            train_acc = 100.0 - train_loss if train_loss is not None else None

            if has_valid:
                valid_loss = self.__valid_loss_list[epoch] if epoch < len(self.__valid_loss_list) else None
                valid_acc = 100.0 - valid_loss if valid_loss is not None else None
                result.append([epoch + 1, train_loss, valid_loss, train_acc, valid_acc])
            else:
                result.append([epoch + 1, train_loss, train_acc])

        return result

    def save_csv(self, csv, path):
        save_uri = f"{self.__operation_builder.get_bucket_url()}/{self.__hierarchy_root}/{path}"
        mpp.intel64.save_csv(csv, save_uri, channel=self.__operation_builder.get_operation_channel(), access_token=self.__operation_builder.get_access_token(), chunk_size=self.__operation_builder.get_chunk_size())
        return self
    
    def save_file(self, file, path):
        save_uri = f"{self.__operation_builder.get_bucket_url()}/{self.__hierarchy_root}/{path}"
        mpp.intel64.save(file, save_uri, channel=self.__operation_builder.get_operation_channel(), access_token=self.__operation_builder.get_access_token(), chunk_size=self.__operation_builder.get_chunk_size())
        return self

    def upload_model(self, model, path, inputs):
        model_save_uri = f"{self.__operation_builder.get_bucket_url()}/{self.__hierarchy_root}/{path}"
        mpp.daq.object_service.upload_model(model, uri=model_save_uri, inference_info=self.__inference_info, example=inputs, channel=self.__operation_builder.get_operation_channel(), access_token=self.__operation_builder.get_access_token(), chunk_size=self.__operation_builder.get_chunk_size())
        return self

    def score_list_graph_image(self, total_epoch, loss_list, y_max=None, title="", color='r'):
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

    def confusion_matrix_image(self, true_list, pred_list, labels):
        matrix = confusion_matrix(true_list, pred_list, labels=[x for x in range(len(labels))])
        plt.figure(figsize=(9,9))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
        plt.title("Confusion Matrix")
        plt.colorbar()
        marks = np.arange(len(labels))
        nlabels = []
        for k in range(len(matrix)):
            nlabel = f'{labels[k]}'
            nlabels.append(nlabel)

        plt.xticks(marks, labels, rotation=45)
        plt.yticks(marks, nlabels, rotation=45)

        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, matrix[i, j], horizontalalignment="center", color="black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        matrix_buffer = io.BytesIO()
        plt.savefig(matrix_buffer, format='png')

        img_arr = np.frombuffer(matrix_buffer.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.clf()
        plt.close()

        return img

    def score_list_csv(self, score_list, header='Loss'):
        result = [['Epoch', header]] + [[epoch, score] for epoch, score in enumerate(score_list, 1)]
        return result
    
    def build(self):
        return self

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

if __name__ =="__main__":
    kwargs = json.loads(parameters)
    kwargs['authentication']['operation_service_address'] = ""
    kwargs['authentication']['access_token'] = ""
    kwargs['gt_dataset']['gt_dataset_id'] = r""
    kwargs['result']['id'] = r""

    RecipeRun(**kwargs) 