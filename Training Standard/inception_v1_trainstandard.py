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
        "input_size" : 299,
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

import logging
logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

from Builder import Operation_Builder
from Builder import Json_HyperparameterBuilder
from Builder import DAQ_Classification_ClassCodeBuilder
from Builder import DAQ_Pytorch_ClassificatoinDatasetBuilder
from Builder import Pytorch_InceptionV1 as InceptionModel
from Builder import Pytorch_TrainingBuilder

from Builder import TrainHook
class SaveHook(TrainHook):
    def __init__(self, save_epoch, logger, bucket_url, operation_channel, access_token, chunk_size, label_info=None, etc=None):
        self.__save_epoch = save_epoch
        self.__loss = None

        from Builder import DAQ_MoritoringBuilder
        self.__monitoring_builder = DAQ_MoritoringBuilder(logger).builder()

        from Builder import DAQ_SaveBuilder
        self.__save_builder = DAQ_SaveBuilder(
                ).init_inference_info(
                    label_info = label_info,
                    etc=etc
                ).init_save_url(
                    bucket_url= bucket_url,
                    operation_channel= operation_channel,
                    access_token= access_token,
                    chunk_size= chunk_size,
                ).build()
    
    def on_epoch_end(self, total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time, model):
        self.__monitoring_builder.monitoring(total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time)
        self.__save_builder.append_metrics({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": 100-train_loss,
            "validation_accuracy": 100-validation_loss if validation_loss is not None else None
        })
        self.__save_builder.save_csv_metric()

        if epoch % self.__save_epoch == 0:
            self.__save_builder.save_model(f"{epoch}/model.pth", model=model)

        loss = validation_loss if validation_loss is not None else train_loss
        if self.__loss is None or loss < self.__loss:
            self.__loss = loss
            self.__save_builder.save_model(f"best/model.pth", model=model)

    def training_start(self):
        logger.info("Training Started.")
    def training_end(self):
        logger.info("Training Ended.")
        self.__save_builder.save_file_index()

def RecipeRun(**kwargs):
    operation_builder = Operation_Builder(**kwargs).initialize().build()
    
    classcode_builder = DAQ_Classification_ClassCodeBuilder().init_url_info(operation_channel=operation_builder.get_operation_channel(),access_token=operation_builder.get_access_token()
                        ).init_label_data(
                            gt_dataset_id=operation_builder.get_gt_dataset_id()
                        ).build()
    
    hyperparameter_builder = Json_HyperparameterBuilder(operation_builder.get_json_hyperparameter()).build()
    dataset_builder = DAQ_Pytorch_ClassificatoinDatasetBuilder(logger=logger).init_url_info(operation_channel=operation_builder.get_operation_channel(),access_token=operation_builder.get_access_token()
                        ).init_dataset_gts(
                            gt_dataset_id=operation_builder.get_gt_dataset_id()
                        ).init_transform(
                            input_size=hyperparameter_builder.get_input_size(),
                            normalize_mean=hyperparameter_builder.get_normalize_mean(),
                            normalize_stdev=hyperparameter_builder.get_normalize_stdev()
                        ).build()
    
    dataset_builder.create_train_dataset(
        train_ratio=hyperparameter_builder.get_train_ratio(),
        batch_size=hyperparameter_builder.get_batch_size(),
        validation_save_random=hyperparameter_builder.get_validation_save_random(),
        class_code_info=classcode_builder.get_class_code_info()
    )

    if dataset_builder.success() is False:
        if logger:logger.error(f"Dataset Build Failed")
        dataset_builder.temp_folder_delete()
        return None

    try:
        
        model = InceptionModel().init_device(hyperparameter_builder.get_device()
                ).init_model(
                    num_classes=classcode_builder.get_class_count()
                ).get_model()
        
        training_builder = Pytorch_TrainingBuilder(logger
                           ).initialize(
                                epoch_total=hyperparameter_builder.get_epoch(),
                                device= hyperparameter_builder.get_device(), # model.device
                                using_amp=hyperparameter_builder.get_using_amp(),
                           ).init_model(
                                model=model
                           ).init_optimizer(
                                optimizer_name=hyperparameter_builder.get_optimizer(),
                                lr=hyperparameter_builder.get_learning_rate()
                           ).init_criterion(
                                criterion_name=hyperparameter_builder.get_criterion()
                           ).builder()

        import json
        label_info = {'inference_info' : json.dumps({"input_size": hyperparameter_builder.get_input_size(), "label_info": classcode_builder.get_label_info()})}
        savehook = SaveHook(
                    save_epoch=hyperparameter_builder.get_save_epoch(),
                    logger=logger,
                    bucket_url=operation_builder.get_bucket_url(),
                    operation_channel=operation_builder.get_operation_channel(),
                    access_token=operation_builder.get_access_token(),
                    chunk_size=operation_builder.get_chunk_size(),
                    label_info=label_info
                   )

        training_builder.train(
            train_data_loader=dataset_builder.get_train_data_loader(),
            validation_data_loader=dataset_builder.get_validation_data_loader(),
            hook=savehook
        )

    except Exception as e:
        logger.error(f"Error Message : {e}")
        raise Exception(f"Train Failed, Error Message : {e}")
    finally:
        dataset_builder.temp_folder_delete()

    
if __name__ =="__main__":
    import json
    kwargs = json.loads(parameters)
    kwargs['authentication']['operation_service_address'] = "192.168.70.62:5022"
    kwargs['authentication']['access_token'] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJtaW5qaS5zb25nIiwibmFtZSI6IuyGoeuvvOyngCIsInJvbGUiOiJzdXBlcl9hZG1pbmlzdHJhdG9yIiwiZ3JvdXBzaWQiOiJtaXJlcm8iLCJsb2dfaW5fcHJvdmlkZXIiOiJkYXEiLCJwcml2aWxlZ2VzIjpbImFjY291bnRfZ3JvdXBfY3JlYXRlIiwiYWNjb3VudF9ncm91cF9yZWFkIiwiYWNjb3VudF9ncm91cF91cGRhdGUiLCJhY2NvdW50X2dyb3VwX2RlbGV0ZSIsImFjY291bnRfZ3JvdXBfcmVhZF9hbnkiLCJhY2NvdW50X2dyb3VwX3VwZGF0ZV9hbnkiLCJhY2NvdW50X2dyb3VwX2RlbGV0ZV9hbnkiLCJhY2NvdW50X3VzZXJfY3JlYXRlIiwiYWNjb3VudF91c2VyX3JlYWQiLCJhY2NvdW50X3VzZXJfdXBkYXRlIiwiYWNjb3VudF91c2VyX2RlbGV0ZSIsImFjY291bnRfdXNlcl9yZWFkX2FueSIsImFjY291bnRfdXNlcl91cGRhdGVfYW55IiwiYWNjb3VudF91c2VyX2RlbGV0ZV9hbnkiLCJkYXRhc2V0X3ZvbHVtZV9jcmVhdGUiLCJkYXRhc2V0X3ZvbHVtZV9yZWFkIiwiZGF0YXNldF92b2x1bWVfdXBkYXRlIiwiZGF0YXNldF92b2x1bWVfZGVsZXRlIiwiZGF0YXNldF92b2x1bWVfcmVhZF9hbnkiLCJkYXRhc2V0X3ZvbHVtZV91cGRhdGVfYW55IiwiZGF0YXNldF92b2x1bWVfZGVsZXRlX2FueSIsImRhdGFzZXRfY2xhc3NfY29kZV9jcmVhdGUiLCJkYXRhc2V0X2NsYXNzX2NvZGVfcmVhZCIsImRhdGFzZXRfY2xhc3NfY29kZV91cGRhdGUiLCJkYXRhc2V0X2NsYXNzX2NvZGVfZGVsZXRlIiwiZGF0YXNldF9jbGFzc19jb2RlX3JlYWRfYW55IiwiZGF0YXNldF9jbGFzc19jb2RlX3VwZGF0ZV9hbnkiLCJkYXRhc2V0X2NsYXNzX2NvZGVfZGVsZXRlX2FueSIsImRhdGFzZXRfZ3RfZGF0YXNldF9jcmVhdGUiLCJkYXRhc2V0X2d0X2RhdGFzZXRfcmVhZCIsImRhdGFzZXRfZ3RfZGF0YXNldF91cGRhdGUiLCJkYXRhc2V0X2d0X2RhdGFzZXRfZGVsZXRlIiwiZGF0YXNldF9ndF9kYXRhc2V0X3JlYWRfYW55IiwiZGF0YXNldF9ndF9kYXRhc2V0X3VwZGF0ZV9hbnkiLCJkYXRhc2V0X2d0X2RhdGFzZXRfZGVsZXRlX2FueSIsImRhdGFzZXRfY2xhc3NpZmljYXRpb25fZ3RfZGF0YXNldF9jcmVhdGUiLCJkYXRhc2V0X2NsYXNzaWZpY2F0aW9uX2d0X2RhdGFzZXRfcmVhZCIsImRhdGFzZXRfY2xhc3NpZmljYXRpb25fZ3RfZGF0YXNldF91cGRhdGUiLCJkYXRhc2V0X2NsYXNzaWZpY2F0aW9uX2d0X2RhdGFzZXRfZGVsZXRlIiwiZGF0YXNldF9jbGFzc2lmaWNhdGlvbl9ndF9kYXRhc2V0X3JlYWRfYW55IiwiZGF0YXNldF9jbGFzc2lmaWNhdGlvbl9ndF9kYXRhc2V0X3VwZGF0ZV9hbnkiLCJkYXRhc2V0X2NsYXNzaWZpY2F0aW9uX2d0X2RhdGFzZXRfZGVsZXRlX2FueSIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfY3JlYXRlIiwiZGF0YXNldF9zZWdtZW50YXRpb25fZ3RfZGF0YXNldF9yZWFkIiwiZGF0YXNldF9zZWdtZW50YXRpb25fZ3RfZGF0YXNldF91cGRhdGUiLCJkYXRhc2V0X3NlZ21lbnRhdGlvbl9ndF9kYXRhc2V0X2RlbGV0ZSIsImRhdGFzZXRfc2VnbWVudGF0aW9uX2d0X2RhdGFzZXRfcmVhZF9hbnkiLCJkYXRhc2V0X3NlZ21lbnRhdGlvbl9ndF9kYXRhc2V0X3VwZGF0ZV9hbnkiLCJkYXRhc2V0X3NlZ21lbnRhdGlvbl9ndF9kYXRhc2V0X2RlbGV0ZV9hbnkiLCJkYXRhc2V0X2ltYWdlX2RhdGFzZXRfY3JlYXRlIiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X3JlYWQiLCJkYXRhc2V0X2ltYWdlX2RhdGFzZXRfdXBkYXRlIiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X2RlbGV0ZSIsImRhdGFzZXRfaW1hZ2VfZGF0YXNldF9yZWFkX2FueSIsImRhdGFzZXRfaW1hZ2VfZGF0YXNldF91cGRhdGVfYW55IiwiZGF0YXNldF9pbWFnZV9kYXRhc2V0X2RlbGV0ZV9hbnkiLCJkZWZlY3RfZGVmZWN0X2RhdGFfY3JlYXRlIiwiZGVmZWN0X2RlZmVjdF9kYXRhX3JlYWQiLCJkZWZlY3RfZGVmZWN0X2RhdGFfdXBkYXRlIiwiZGVmZWN0X2RlZmVjdF9kYXRhX2RlbGV0ZSIsImRlZmVjdF9kZWZlY3RfZGF0YV9yZWFkX2FueSIsImRlZmVjdF9kZWZlY3RfZGF0YV91cGRhdGVfYW55IiwiZGVmZWN0X2RlZmVjdF9kYXRhX2RlbGV0ZV9hbnkiLCJkZWZlY3RfbWFzdGVyX2RhdGFfZmllbGRfY3JlYXRlIiwiZGVmZWN0X21hc3Rlcl9kYXRhX2ZpZWxkX3JlYWQiLCJkZWZlY3RfbWFzdGVyX2RhdGFfZmllbGRfdXBkYXRlIiwiZGVmZWN0X21hc3Rlcl9kYXRhX2ZpZWxkX2RlbGV0ZSIsImRlZmVjdF9tYXN0ZXJfZGF0YV9maWVsZF9yZWFkX2FueSIsImRlZmVjdF9tYXN0ZXJfZGF0YV9maWVsZF91cGRhdGVfYW55IiwiZGVmZWN0X21hc3Rlcl9kYXRhX2ZpZWxkX2RlbGV0ZV9hbnkiLCJkZWZlY3RfbWFzdGVyX2RhdGFfc2V0X2NyZWF0ZSIsImRlZmVjdF9tYXN0ZXJfZGF0YV9zZXRfcmVhZCIsImRlZmVjdF9tYXN0ZXJfZGF0YV9zZXRfdXBkYXRlIiwiZGVmZWN0X21hc3Rlcl9kYXRhX3NldF9kZWxldGUiLCJkZWZlY3RfbWFzdGVyX2RhdGFfc2V0X3JlYWRfYW55IiwiZGVmZWN0X21hc3Rlcl9kYXRhX3NldF91cGRhdGVfYW55IiwiZGVmZWN0X21hc3Rlcl9kYXRhX3NldF9kZWxldGVfYW55IiwiZGVmZWN0X21hc3Rlcl9kYXRhX2hpc3RvcnlfY3JlYXRlIiwiZGVmZWN0X21hc3Rlcl9kYXRhX2hpc3RvcnlfcmVhZCIsImRlZmVjdF9tYXN0ZXJfZGF0YV9oaXN0b3J5X3VwZGF0ZSIsImRlZmVjdF9tYXN0ZXJfZGF0YV9oaXN0b3J5X2RlbGV0ZSIsImRlZmVjdF9tYXN0ZXJfZGF0YV9oaXN0b3J5X3JlYWRfYW55IiwiZGVmZWN0X21hc3Rlcl9kYXRhX2hpc3RvcnlfdXBkYXRlX2FueSIsImRlZmVjdF9tYXN0ZXJfZGF0YV9oaXN0b3J5X2RlbGV0ZV9hbnkiLCJkZWZlY3RfbWFzdGVyX2RhdGFfZmllbGRfZ3JvdXBfaGlzdG9yeV9jcmVhdGUiLCJkZWZlY3RfbWFzdGVyX2RhdGFfZmllbGRfZ3JvdXBfaGlzdG9yeV9yZWFkIiwiZGVmZWN0X21hc3Rlcl9kYXRhX2ZpZWxkX2dyb3VwX2hpc3RvcnlfdXBkYXRlIiwiZGVmZWN0X21hc3Rlcl9kYXRhX2ZpZWxkX2dyb3VwX2hpc3RvcnlfZGVsZXRlIiwiZGVmZWN0X21hc3Rlcl9kYXRhX2ZpZWxkX2dyb3VwX2hpc3RvcnlfcmVhZF9hbnkiLCJkZWZlY3RfbWFzdGVyX2RhdGFfZmllbGRfZ3JvdXBfaGlzdG9yeV91cGRhdGVfYW55IiwiZGVmZWN0X21hc3Rlcl9kYXRhX2ZpZWxkX2dyb3VwX2hpc3RvcnlfZGVsZXRlX2FueSIsImdkc19jbGlwX2NyZWF0ZSIsImdkc19jbGlwX3JlYWQiLCJnZHNfY2xpcF91cGRhdGUiLCJnZHNfY2xpcF9kZWxldGUiLCJnZHNfY2xpcF9yZWFkX2FueSIsImdkc19jbGlwX3VwZGF0ZV9hbnkiLCJnZHNfY2xpcF9kZWxldGVfYW55IiwiZ2RzX2V4cG9ydF9jcmVhdGUiLCJnZHNfZXhwb3J0X3JlYWQiLCJnZHNfZXhwb3J0X3VwZGF0ZSIsImdkc19leHBvcnRfZGVsZXRlIiwiZ2RzX2V4cG9ydF9yZWFkX2FueSIsImdkc19leHBvcnRfdXBkYXRlX2FueSIsImdkc19leHBvcnRfZGVsZXRlX2FueSIsImdkc19nZHNfY3JlYXRlIiwiZ2RzX2dkc19yZWFkIiwiZ2RzX2dkc191cGRhdGUiLCJnZHNfZ2RzX2RlbGV0ZSIsImdkc19nZHNfcmVhZF9hbnkiLCJnZHNfZ2RzX3VwZGF0ZV9hbnkiLCJnZHNfZ2RzX2RlbGV0ZV9hbnkiLCJnZHNfc2VydmVyX2NyZWF0ZSIsImdkc19zZXJ2ZXJfcmVhZCIsImdkc19zZXJ2ZXJfdXBkYXRlIiwiZ2RzX3NlcnZlcl9kZWxldGUiLCJnZHNfc2VydmVyX3JlYWRfYW55IiwiZ2RzX3NlcnZlcl91cGRhdGVfYW55IiwiZ2RzX3NlcnZlcl9kZWxldGVfYW55IiwiZ2RzX3ZvbHVtZV9jcmVhdGUiLCJnZHNfdm9sdW1lX3JlYWQiLCJnZHNfdm9sdW1lX3VwZGF0ZSIsImdkc192b2x1bWVfZGVsZXRlIiwiZ2RzX3ZvbHVtZV9yZWFkX2FueSIsImdkc192b2x1bWVfdXBkYXRlX2FueSIsImdkc192b2x1bWVfZGVsZXRlX2FueSIsImluZmVyZW5jZV9pbmZlcmVuY2VfY3JlYXRlIiwiaW5mZXJlbmNlX2luZmVyZW5jZV9yZWFkIiwiaW5mZXJlbmNlX2luZmVyZW5jZV91cGRhdGUiLCJpbmZlcmVuY2VfaW5mZXJlbmNlX2RlbGV0ZSIsImluZmVyZW5jZV9pbmZlcmVuY2VfcmVhZF9hbnkiLCJpbmZlcmVuY2VfaW5mZXJlbmNlX3VwZGF0ZV9hbnkiLCJpbmZlcmVuY2VfaW5mZXJlbmNlX2RlbGV0ZV9hbnkiLCJpbmZlcmVuY2VfbW9kZWxfY3JlYXRlIiwiaW5mZXJlbmNlX21vZGVsX3JlYWQiLCJpbmZlcmVuY2VfbW9kZWxfdXBkYXRlIiwiaW5mZXJlbmNlX21vZGVsX2RlbGV0ZSIsImluZmVyZW5jZV9tb2RlbF9yZWFkX2FueSIsImluZmVyZW5jZV9tb2RlbF91cGRhdGVfYW55IiwiaW5mZXJlbmNlX21vZGVsX2RlbGV0ZV9hbnkiLCJpbmZlcmVuY2Vfc2VydmVyX2NyZWF0ZSIsImluZmVyZW5jZV9zZXJ2ZXJfcmVhZCIsImluZmVyZW5jZV9zZXJ2ZXJfdXBkYXRlIiwiaW5mZXJlbmNlX3NlcnZlcl9kZWxldGUiLCJpbmZlcmVuY2Vfc2VydmVyX3JlYWRfYW55IiwiaW5mZXJlbmNlX3NlcnZlcl91cGRhdGVfYW55IiwiaW5mZXJlbmNlX3NlcnZlcl9kZWxldGVfYW55IiwiaW5mZXJlbmNlX3ZvbHVtZV9jcmVhdGUiLCJpbmZlcmVuY2Vfdm9sdW1lX3JlYWQiLCJpbmZlcmVuY2Vfdm9sdW1lX3VwZGF0ZSIsImluZmVyZW5jZV92b2x1bWVfZGVsZXRlIiwiaW5mZXJlbmNlX3ZvbHVtZV9yZWFkX2FueSIsImluZmVyZW5jZV92b2x1bWVfdXBkYXRlX2FueSIsImluZmVyZW5jZV92b2x1bWVfZGVsZXRlX2FueSIsInVwZGF0ZV9tcHBfY3JlYXRlIiwidXBkYXRlX21wcF9yZWFkIiwidXBkYXRlX21wcF91cGRhdGUiLCJ1cGRhdGVfbXBwX2RlbGV0ZSIsInVwZGF0ZV9tcHBfcmVhZF9hbnkiLCJ1cGRhdGVfbXBwX3VwZGF0ZV9hbnkiLCJ1cGRhdGVfbXBwX2RlbGV0ZV9hbnkiLCJ1cGRhdGVfcmNfY3JlYXRlIiwidXBkYXRlX3JjX3JlYWQiLCJ1cGRhdGVfcmNfdXBkYXRlIiwidXBkYXRlX3JjX2RlbGV0ZSIsInVwZGF0ZV9yY19yZWFkX2FueSIsInVwZGF0ZV9yY191cGRhdGVfYW55IiwidXBkYXRlX3JjX2RlbGV0ZV9hbnkiLCJ3b3JrZmxvd19qb2JfY3JlYXRlIiwid29ya2Zsb3dfam9iX3JlYWQiLCJ3b3JrZmxvd19qb2JfdXBkYXRlIiwid29ya2Zsb3dfam9iX2RlbGV0ZSIsIndvcmtmbG93X2pvYl9yZWFkX2FueSIsIndvcmtmbG93X2pvYl91cGRhdGVfYW55Iiwid29ya2Zsb3dfam9iX2RlbGV0ZV9hbnkiLCJ3b3JrZmxvd19zZXJ2ZXJfY3JlYXRlIiwid29ya2Zsb3dfc2VydmVyX3JlYWQiLCJ3b3JrZmxvd19zZXJ2ZXJfdXBkYXRlIiwid29ya2Zsb3dfc2VydmVyX2RlbGV0ZSIsIndvcmtmbG93X3NlcnZlcl9yZWFkX2FueSIsIndvcmtmbG93X3NlcnZlcl91cGRhdGVfYW55Iiwid29ya2Zsb3dfc2VydmVyX2RlbGV0ZV9hbnkiLCJ3b3JrZmxvd192b2x1bWVfY3JlYXRlIiwid29ya2Zsb3dfdm9sdW1lX3JlYWQiLCJ3b3JrZmxvd192b2x1bWVfdXBkYXRlIiwid29ya2Zsb3dfdm9sdW1lX2RlbGV0ZSIsIndvcmtmbG93X3ZvbHVtZV9yZWFkX2FueSIsIndvcmtmbG93X3ZvbHVtZV91cGRhdGVfYW55Iiwid29ya2Zsb3dfdm9sdW1lX2RlbGV0ZV9hbnkiLCJ3b3JrZmxvd193b3JrZXJfY3JlYXRlIiwid29ya2Zsb3dfd29ya2VyX3JlYWQiLCJ3b3JrZmxvd193b3JrZXJfdXBkYXRlIiwid29ya2Zsb3dfd29ya2VyX2RlbGV0ZSIsIndvcmtmbG93X3dvcmtlcl9yZWFkX2FueSIsIndvcmtmbG93X3dvcmtlcl91cGRhdGVfYW55Iiwid29ya2Zsb3dfd29ya2VyX2RlbGV0ZV9hbnkiLCJ3b3JrZmxvd193b3JrZmxvd19jcmVhdGUiLCJ3b3JrZmxvd193b3JrZmxvd19yZWFkIiwid29ya2Zsb3dfd29ya2Zsb3dfdXBkYXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfZGVsZXRlIiwid29ya2Zsb3dfd29ya2Zsb3dfcmVhZF9hbnkiLCJ3b3JrZmxvd193b3JrZmxvd191cGRhdGVfYW55Iiwid29ya2Zsb3dfd29ya2Zsb3dfZGVsZXRlX2FueSIsIm9iamVjdF9vYmplY3RfY3JlYXRlIiwib2JqZWN0X29iamVjdF9yZWFkIiwib2JqZWN0X29iamVjdF91cGRhdGUiLCJvYmplY3Rfb2JqZWN0X2RlbGV0ZSIsIm9iamVjdF9vYmplY3RfcmVhZF9hbnkiLCJvYmplY3Rfb2JqZWN0X3VwZGF0ZV9hbnkiLCJvYmplY3Rfb2JqZWN0X2RlbGV0ZV9hbnkiLCJvYmplY3Rfdm9sdW1lX2NyZWF0ZSIsIm9iamVjdF92b2x1bWVfcmVhZCIsIm9iamVjdF92b2x1bWVfdXBkYXRlIiwib2JqZWN0X3ZvbHVtZV9kZWxldGUiLCJvYmplY3Rfdm9sdW1lX3JlYWRfYW55Iiwib2JqZWN0X3ZvbHVtZV91cGRhdGVfYW55Iiwib2JqZWN0X3ZvbHVtZV9kZWxldGVfYW55Iiwic2NoZWR1bGVyX2NvbmRpdGlvbl9jcmVhdGUiLCJzY2hlZHVsZXJfY29uZGl0aW9uX3JlYWQiLCJzY2hlZHVsZXJfY29uZGl0aW9uX3VwZGF0ZSIsInNjaGVkdWxlcl9jb25kaXRpb25fZGVsZXRlIiwic2NoZWR1bGVyX2NvbmRpdGlvbl9yZWFkX2FueSIsInNjaGVkdWxlcl9jb25kaXRpb25fdXBkYXRlX2FueSIsInNjaGVkdWxlcl9jb25kaXRpb25fZGVsZXRlX2FueSIsInNjaGVkdWxlcl9ldmVudF9oaXN0b3J5X2NyZWF0ZSIsInNjaGVkdWxlcl9ldmVudF9oaXN0b3J5X3JlYWQiLCJzY2hlZHVsZXJfZXZlbnRfaGlzdG9yeV91cGRhdGUiLCJzY2hlZHVsZXJfZXZlbnRfaGlzdG9yeV9kZWxldGUiLCJzY2hlZHVsZXJfZXZlbnRfaGlzdG9yeV9yZWFkX2FueSIsInNjaGVkdWxlcl9ldmVudF9oaXN0b3J5X3VwZGF0ZV9hbnkiLCJzY2hlZHVsZXJfZXZlbnRfaGlzdG9yeV9kZWxldGVfYW55IiwiZGF0YXNldF9jbGFzc19jb2RlX21hcF9jcmVhdGUiLCJkYXRhc2V0X2NsYXNzX2NvZGVfbWFwX3JlYWQiLCJkYXRhc2V0X2NsYXNzX2NvZGVfbWFwX3VwZGF0ZSIsImRhdGFzZXRfY2xhc3NfY29kZV9tYXBfZGVsZXRlIiwiZGF0YXNldF9jbGFzc19jb2RlX21hcF9yZWFkX2FueSIsImRhdGFzZXRfY2xhc3NfY29kZV9tYXBfdXBkYXRlX2FueSIsImRhdGFzZXRfY2xhc3NfY29kZV9tYXBfZGVsZXRlX2FueSIsInB1YmxpY19zZXR0aW5nc19jcmVhdGUiLCJwdWJsaWNfc2V0dGluZ3NfcmVhZCIsInB1YmxpY19zZXR0aW5nc191cGRhdGUiLCJwdWJsaWNfc2V0dGluZ3NfZGVsZXRlIiwicHVibGljX3NldHRpbmdzX3JlYWRfYW55IiwicHVibGljX3NldHRpbmdzX3VwZGF0ZV9hbnkiLCJwdWJsaWNfc2V0dGluZ3NfZGVsZXRlX2FueSIsImluZmVyZW5jZV9tb2RlbF9wb2xpY3lfY2hhbmdlIl0sImdyb3VwX3N5c3RlbXMiOlsibG9jYWwiLCJtaXJlcm8taWRjLWluLWxpbmUiLCJtaXJlcm8taWRjLW9mZi1saW5lIl0sImdyb3VwX2ZlYXR1cmVzIjpbInJjX3JlY2lwZV9kZXZlbG9wbWVudCIsInJjX3JlY2lwZV9pbWFnZV9hbmFseXplciIsInJjX3JlY2lwZV9tYW5hZ2VtZW50IiwicmNfZGF0YXNldF9tYW5hZ2VtZW50IiwicmNfd29ya2Zsb3dfbWFuYWdlbWVudCIsInJjX3dvcmtmbG93X21vbml0b3JpbmciLCJyY19tYWNoaW5lX2xlYXJuaW5nX2NsYXNzX2NvZGUiLCJyY19tYWNoaW5lX2xlYXJuaW5nX21vZGVsX21hbmFnZW1lbnQiLCJyY19nZHNfbWFuYWdlbWVudCIsInJjX2dkc19mbG9vcl9wbGFuX21hbmFnZW1lbnQiLCJyY19vcGVyYXRpb25fZGVmZWN0X2RhdGFfbWFuYWdlbWVudCIsInJjX29wZXJhdGlvbl9zY2hlZHVsZXJfbWFuYWdlbWVudCIsInJjX29wZXJhdGlvbl9zdGF0dXMiLCJyY19vcGVyYXRpb25fbW9uaXRvcmluZyIsInJjX21hc3RlcmRhdGFzZXRfbWFuYWdlbWVudCIsInJjX2RlZmVjdF9jb2xsZWN0aW9uIiwicmNfY2xhc3NfY29kZV9tYXBfbWFuYWdlbWVudCJdLCJqdGkiOiJmY2Q3MmY2MS02ZmMzLTQ4MzctYjUyMi0xYjUxMmYyYTVkNDEiLCJpYXQiOjE3NjA5MzcxODQsIm5iZiI6MTc2MDkzNzE4NCwiZXhwIjoxNzYyMTQ2Nzg0LCJpc3MiOiJkYXEiLCJhdWQiOiJkYXFfdXNlciJ9.NINdQnHTWFE41M7KUwBOEGShtlr9PDiRP57-X7KsVe0"
    kwargs['gt_dataset']['gt_dataset_id'] = r"20250930154837_labeled_dataset"
    import random
    kwargs['result']['id'] = f"test_result_volume_{random.randint(1000,9999)}"

    RecipeRun(**kwargs)

