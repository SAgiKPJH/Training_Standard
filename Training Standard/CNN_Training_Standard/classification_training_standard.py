##UI{"$ref": "ui.json"}IU

##!--{"Name":"hyperparameter","Type":"epoch","Key":"epoch","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"batch_size","Key":"batch_size","Value":"","Category":""}
##!--{"Name":"result","Type":"result","Key":"id","Value":"","Category":""}
##!--{"Name":"authentication","Type":"system_address","Key":"operation_service_address","Value":"","Category":""}
##!--{"Name":"authentication","Type":"access_token","Key":"access_token","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id","Value":"","Category":""}
##$--

parameters = '''{
    "version" : "CNN_Training_Standard_v1.0.3",
    "hyperparameter":{
        "framework" : "pytorch",
        "network_name" : "efficientnet_b0",
        "epoch" : 20,
        "save_epoch" : 10,
        "batch_size" : 16,
        "lr" : 1e-3,
        "optimizer_name" : "Adam",
        "criterion" : "CrossEntropyLoss",
        "input_size" : 224,
        "normalize_mean" : 0.5,
        "normalize_stdev" : 0.5,
        "using_gpu" : true,
        "using_amp" : true,
        "train_ratio" : 0.8,
        "validation_save_random" : false,
        "debug" : false,
        "daq_old_path" : false,
        "loss_eps" : 0
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

import json
import logging
logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

from Builder import Json_HyperparameterBuilder
from Builder import get_framework_builders, get_daq_framework_builders
from Builder import TrainHook

class SaveHook(TrainHook):
    def __init__(self, save_epoch, logger, bucket_url, operation_channel, access_token, chunk_size, framework, daq_old_path=False, label_info=None, etc=None):
        self.__save_epoch = save_epoch
        self.__best_loss = None
        self.__daq_old_path = daq_old_path
        self.__framework = framework

        from Builder import DAQ_MoritoringBuilder
        self.__monitoring_builder = DAQ_MoritoringBuilder(logger).builder()

        if daq_old_path:
            from Builder import DAQ_SaveBuilder_DAQ_OLD
            self.__save_builder = DAQ_SaveBuilder_DAQ_OLD(
                    ).init_inference_info(
                        label_info = label_info,
                        etc=etc
                    ).init_save_url(
                        bucket_url= bucket_url,
                        operation_channel= operation_channel,
                        access_token= access_token,
                        chunk_size= chunk_size,
                    ).build()
        else:
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

        if self.__daq_old_path:
            self.__save_builder.save_csv_metric(epoch=epoch)
        else:
            self.__save_builder.save_csv_metric()

        ext = ".pth" if self.__framework == "pytorch" else ".h5"

        if (self.__save_epoch > 0 and epoch % self.__save_epoch == 0) or (self.__save_epoch == 0 and epoch == total_epoch):
            if self.__daq_old_path:
                self.__save_builder.save_model(epoch=epoch, model=model)
            else:
                self.__save_builder.save_model(f"{epoch}/model{ext}", model=model)

        # Best 모델 저장
        loss = validation_loss if validation_loss is not None else train_loss
        if self.__best_loss is None or loss < self.__best_loss:
            self.__best_loss = loss
            logger.info(f"  ★ Best model updated (loss: {loss:.6f})")
            if self.__daq_old_path:
                self.__save_builder.save_model(file_full_path=f"best/model/model{ext}", model=model)
            else:
                self.__save_builder.save_model(f"best/model{ext}", model=model)

    def training_start(self):
        logger.info("Training Started.")
    def training_end(self):
        logger.info("Training Ended.")
        self.__save_builder.save_file_index()


def RecipeRun(**kwargs):
    from Builder import Operation_Builder
    from Builder import DAQ_Classification_ClassCodeBuilder

    framework = kwargs['hyperparameter'].get('framework', 'pytorch')
    logger.info(f"Framework: {framework}")
    Model, _, TrainingBuilder, TrainingBuilderDebug = get_framework_builders(framework)
    DAQDatasetBuilder = get_daq_framework_builders(framework)

    operation_builder = Operation_Builder(**kwargs).initialize().build()

    classcode_builder = DAQ_Classification_ClassCodeBuilder().init_url_info(
        operation_channel=operation_builder.get_operation_channel(),
        access_token=operation_builder.get_access_token()
    ).init_label_data(
        gt_dataset_id=operation_builder.get_gt_dataset_id()
    ).build()

    hyperparameter_builder = Json_HyperparameterBuilder(operation_builder.get_json_hyperparameter()).build()

    dataset_builder = DAQDatasetBuilder(logger=logger).init_url_info(
        operation_channel=operation_builder.get_operation_channel(),
        access_token=operation_builder.get_access_token()
    ).init_dataset_gts(
        gt_dataset_id=operation_builder.get_gt_dataset_id()
    ).init_transform(
        input_size=hyperparameter_builder.get_input_size(),
        normalize_mean=hyperparameter_builder.get_normalize_mean(),
        normalize_stdev=hyperparameter_builder.get_normalize_stdev(),
        augmentation=hyperparameter_builder.get_augmentation()
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
        model_builder = Model().init_device(hyperparameter_builder.get_device())
        if framework == "tensorflow":
            model = model_builder.init_model(
                num_classes=classcode_builder.get_class_count(),
                network_name=hyperparameter_builder.get_network_name(),
                input_size=hyperparameter_builder.get_input_size()
            ).get_model()
        else:
            model = model_builder.init_model(
                num_classes=classcode_builder.get_class_count(),
                network_name=hyperparameter_builder.get_network_name()
            ).get_model()

        SelectedTrainingBuilder = TrainingBuilderDebug if hyperparameter_builder.get_debug() else TrainingBuilder
        training_builder = SelectedTrainingBuilder(logger
                           ).initialize(
                                epoch_total=hyperparameter_builder.get_epoch(),
                                device=hyperparameter_builder.get_device(),
                                using_amp=hyperparameter_builder.get_using_amp(),
                                loss_eps=hyperparameter_builder.get_loss_eps(),
                           ).init_model(
                                model=model
                           ).init_optimizer(
                                optimizer_name=hyperparameter_builder.get_optimizer(),
                                lr=hyperparameter_builder.get_learning_rate()
                           ).init_criterion(
                                criterion_name=hyperparameter_builder.get_criterion()
                           ).builder()

        label_info = {'inference_info' : json.dumps({"input_size": hyperparameter_builder.get_input_size(), "label_info": classcode_builder.get_label_info()})}
        savehook = SaveHook(
                    save_epoch=hyperparameter_builder.get_save_epoch(),
                    logger=logger,
                    bucket_url=operation_builder.get_bucket_url(),
                    operation_channel=operation_builder.get_operation_channel(),
                    access_token=operation_builder.get_access_token(),
                    chunk_size=operation_builder.get_chunk_size(),
                    framework=framework,
                    daq_old_path=hyperparameter_builder.get_daq_old_path(),
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
    kwargs = json.loads(parameters)
    kwargs['authentication']['operation_service_address'] = ""
    kwargs['authentication']['access_token'] = ""
    kwargs['gt_dataset']['gt_dataset_id'] = r""
    import random
    kwargs['result']['id'] = f"test_result_volume_{random.randint(1000,9999)}"

    RecipeRun(**kwargs)
