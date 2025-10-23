import json
import logging
from builders import (
    Operation_Builder,
    HyperparameterBuilder,
    Classification_Dataset_Builder,
    Torch_Model_Builder,
    Save_Builder
)

##!--{"Name":"hyperparameter","Type":"epoch","Key":"epoch","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"batch_size","Key":"batch_size","Value":"","Category":""}
##!--{"Name":"result","Type":"result","Key":"id","Value":"","Category":""}
##!--{"Name":"authentication","Type":"system_address","Key":"operation_service_address","Value":"","Category":""}
##!--{"Name":"authentication","Type":"access_token","Key":"access_token","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id","Value":"","Category":""}

##$--
parameters = '''{
    "hyperparameter":{
        "epoch" : 20,
        "save_epoch" : 1,
        "batch_size" : 16,
        "lr" : 1e-3,
        "optimizer_name" : "Adam",
        "input_size" : 384,
        "normalize_mean" : 0.5,
        "normalize_stdev" : 0.5,
        "using_gpu" : true,
        "using_amp" : true,
        "train_ratio" : 0.8,
        "validation_save_random" : false
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

class ConvNextModelBuilder(Torch_Model_Builder):
    def init_model(self, num_classes, device):
        """ConvNeXt Large 384 모델 초기화"""
        import torch
        from timm import create_model
        
        # ConvNeXt Large 384 모델 생성
        model = create_model(
            'convnext_large_384_in22ft1k', 
            pretrained=True,
            num_classes=num_classes
        )
        
        # 사전 학습된 가중치 로드 (있는 경우)
        try:
            weights_path = 'convnext_large_384_in22ft1k.pth'
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
        
        model.to(device)
        self.__model = model
        return self

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
        model_builder = ConvNextModelBuilder(logger)  # ConvNeXt 전용 모델 빌더 사용
        model_builder \
            .initialize(
                epoch_total=hyperparameter_builder.get_epoch(),
                save_epoch=hyperparameter_builder.get_save_epoch()
            ) \
            .init_model(
                num_classes=dataset_builder.get_num_classes(),
                device=operation_builder.get_device()
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

if __name__ =="__main__":
    kwargs = json.loads(parameters)
    kwargs['authentication']['operation_service_address'] = ""
    kwargs['authentication']['access_token'] = ""
    kwargs['gt_dataset']['gt_dataset_id'] = r""
    kwargs['result']['id'] = r""

    RecipeRun(**kwargs)
