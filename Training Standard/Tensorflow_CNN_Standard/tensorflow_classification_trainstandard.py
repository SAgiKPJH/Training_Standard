##UI{
##  "Step1": {
##    "Title": "Set Network",
##    "Label": "Step 1: Set Network and Using GPU",
##    "Parameters": {
##      "SetNetworkName": {
##         "Type": "select",
##         "Label": "Network Name",
##         "Prameter": "hyperparameter.network_name",
##         "Options": [
##             {"Label": "resnet50", "Value": "resnet50"},
##             {"Label": "resnet101", "Value": "resnet101"},
##             {"Label": "resnet152", "Value": "resnet152"},
##             {"Label": "efficientnet_b0", "Value": "efficientnet_b0"},
##             {"Label": "efficientnet_b1", "Value": "efficientnet_b1"},
##             {"Label": "efficientnet_b2", "Value": "efficientnet_b2"},
##             {"Label": "efficientnet_b3", "Value": "efficientnet_b3"},
##             {"Label": "efficientnet_b4", "Value": "efficientnet_b4"},
##             {"Label": "efficientnet_b5", "Value": "efficientnet_b5"},
##             {"Label": "efficientnet_b6", "Value": "efficientnet_b6"},
##             {"Label": "efficientnet_b7", "Value": "efficientnet_b7"},
##             {"Label": "efficientnet_v2_s", "Value": "efficientnet_v2_s"},
##             {"Label": "efficientnet_v2_m", "Value": "efficientnet_v2_m"},
##             {"Label": "efficientnet_v2_l", "Value": "efficientnet_v2_l"},
##             {"Label": "inceptionv3", "Value": "inceptionv3"},
##             {"Label": "mobilenet_v2", "Value": "mobilenet_v2"},
##             {"Label": "mobilenet_v3_small", "Value": "mobilenet_v3_small"},
##             {"Label": "mobilenet_v3_large", "Value": "mobilenet_v3_large"}
##         ]
##      },
##      "SetUsingGPU": {
##        "Type": "checkbox",
##        "Label": "Using GPU",
##        "Prameter": "hyperparameter.using_gpu"
##      }
##    }
##  },
##  "Step2": {
##    "Title": "Set Hyperparameters",
##    "Label": "Step 2: Set Hyperparameters",
##    "Parameters": {
##      "Epoch": { "Type": "integer", "Label": "Epoch", "Prameter": "hyperparameter.epoch"},
##      "SaveEpoch": { "Type": "integer", "Label": "Save Epoch", "Prameter": "hyperparameter.save_epoch"},
##      "BatchSize": { "Type": "integer", "Label": "Batch Size", "Prameter": "hyperparameter.batch_size"},
##      "LearningRate": { "Type": "float", "Label": "Learning Rate", "Prameter": "hyperparameter.lr"}
##    }
##  }
##}IU

##!--{"Name":"hyperparameter","Type":"epoch","Key":"epoch","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"batch_size","Key":"batch_size","Value":"","Category":""}
##!--{"Name":"result","Type":"result","Key":"id","Value":"","Category":""}
##!--{"Name":"authentication","Type":"system_address","Key":"operation_service_address","Value":"","Category":""}
##!--{"Name":"authentication","Type":"access_token","Key":"access_token","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id","Value":"","Category":""}
##$--

parameters = '''{
    "hyperparameter":{
        "network_name" : "efficientnet_b0",
        "epoch" : 20,
        "save_epoch" : 10,
        "batch_size" : 4,
        "lr" : 1e-3,
        "optimizer_name" : "Adam",
        "input_size" : 299,
        "normalize_mean" : 0.5,
        "normalize_stdev" : 0.5,
        "using_gpu" : false,
        "using_amp" : true,
        "train_ratio" : 0.8,
        "validation_save_random" : false,
        "debug" : false
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
from Builder import DAQ_Tensorflow_ClassificationDatasetBuilder
from Builder import Tensorflow_Classification_Models as Model
from Builder import Tensorflow_TrainingBuilder
from Builder import Tensorflow_TrainingBuilder_Debug

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
            self.__save_builder.save_model(f"{epoch}/model", model=model)

        loss = validation_loss if validation_loss is not None else train_loss
        if self.__loss is None or loss < self.__loss:
            self.__loss = loss
            self.__save_builder.save_model(f"best/model", model=model)

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
    dataset_builder = DAQ_Tensorflow_ClassificationDatasetBuilder(logger=logger).init_url_info(operation_channel=operation_builder.get_operation_channel(),access_token=operation_builder.get_access_token()
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

        model = Model().init_device(hyperparameter_builder.get_device()
                ).init_model(
                    num_classes=classcode_builder.get_class_count(),
                    input_size=hyperparameter_builder.get_input_size()
                ).get_model()

        TrainingBuilder = Tensorflow_TrainingBuilder_Debug if hyperparameter_builder.get_debug() else Tensorflow_TrainingBuilder
        training_builder = TrainingBuilder(logger
                           ).initialize(
                                epoch_total=hyperparameter_builder.get_epoch(),
                                device= hyperparameter_builder.get_device(),
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
    kwargs['authentication']['access_token'] = ""
    kwargs['gt_dataset']['gt_dataset_id'] = r"20250930154837_labeled_dataset"
    import random
    kwargs['result']['id'] = f"test_result_volume_{random.randint(1000,9999)}"

    RecipeRun(**kwargs)
