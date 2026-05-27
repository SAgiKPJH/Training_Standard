parameters = '''{
    "version" : "Seg_Training_Standard_v1.0.0",
    "hyperparameter":{
        "framework" : "pytorch",
        "network_name" : "deeplabv3_resnet50",
        "epoch" : 5,
        "save_epoch" : 1,
        "batch_size" : 2,
        "lr" : 1e-2,
        "weight_decay" : 1e-4,
        "optimizer_name" : "SGD",
        "criterion" : "CrossEntropyLoss",
        "input_size" : 512,
        "normalize_mean" : 0.5,
        "normalize_stdev" : 0.5,
        "using_gpu" : false,
        "using_amp" : false,
        "train_ratio" : 0.8,
        "validation_save_random" : false,
        "validation_save_count" : 5,
        "output_stride" : 16,
        "pretrained_backbone" : true,
        "debug" : false,
        "daq_old_path" : false,
        "loss_eps" : 0
    }
}'''
import json
hyperparameter = json.dumps(json.loads(parameters)['hyperparameter'])

import logging
logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

dataset = "D:\\Code\\Training_Standard\\create_dataset\\segmentation_dataset"
output = "D:\\Code\\Training_Standard\\create_dataset\\segmentation_output"

from Builder import Json_HyperparameterBuilder
hyperparameter_builder = Json_HyperparameterBuilder(hyperparameter).build()
framework = hyperparameter_builder.get_framework()
logger.info(f"Framework: {framework}")

from Builder import Local_Segmentation_ClassCodeBuilder
classcode_builder = Local_Segmentation_ClassCodeBuilder().init_label_data(dataset_path=dataset).build()

from Builder import get_framework_builders
Model, LocalDatasetBuilder, TrainingBuilder = get_framework_builders(framework)

dataset_builder = LocalDatasetBuilder(logger=logger).init_dataset_path(dataset_path=dataset
                  ).init_transform(
                      input_size=hyperparameter_builder.get_input_size(),
                      normalize_mean=hyperparameter_builder.get_normalize_mean(),
                      normalize_stdev=hyperparameter_builder.get_normalize_stdev(),
                      augmentation=hyperparameter_builder.get_augmentation()
                  ).create_train_dataset(
                      train_ratio=hyperparameter_builder.get_train_ratio(),
                      batch_size=hyperparameter_builder.get_batch_size(),
                      validation_save_random=hyperparameter_builder.get_validation_save_random(),
                      class_code_info=classcode_builder.get_class_code_info()
                  )

if dataset_builder.success() is False:
    if logger: logger.error("Dataset Build Failed")
    raise RuntimeError("Dataset Build Failed")


from Builder import TrainHook
class SaveHook(TrainHook):
    def __init__(self, save_epoch, logger, save_path, daq_old_path=False, label_info=None, etc=None):
        self.__save_epoch = save_epoch
        self.__save_path = save_path
        self.__best_loss = None
        self.__daq_old_path = daq_old_path

        from Builder import Local_MonitoringBuilder
        self.__monitoring_builder = Local_MonitoringBuilder(logger).builder()

        if daq_old_path:
            from Builder import Local_SaveBuilder_DAQ_OLD
            self.__save_builder = Local_SaveBuilder_DAQ_OLD(
                ).init_inference_info(label_info=label_info, etc=etc
                ).init_save_url(save_path=self.__save_path).build()
        else:
            from Builder import Local_SaveBuilder
            self.__save_builder = Local_SaveBuilder(
                ).init_inference_info(label_info=label_info, etc=etc
                ).init_save_url(save_path=self.__save_path).build()

    def on_epoch_end(self, total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time, model):
        self.__monitoring_builder.monitoring(total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time)
        self.__save_builder.append_metrics({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })

        if self.__daq_old_path:
            self.__save_builder.save_csv_metric(epoch=epoch)
        else:
            self.__save_builder.save_csv_metric()

        ext = ".pth" if framework == "pytorch" else ".h5"

        if (self.__save_epoch > 0 and epoch % self.__save_epoch == 0) or (self.__save_epoch == 0 and epoch == total_epoch):
            if self.__daq_old_path:
                self.__save_builder.save_model(epoch=epoch, model=model)
            else:
                self.__save_builder.save_model(f"{epoch}/model{ext}", model=model)

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


try:
    model_builder = Model().init_device(hyperparameter_builder.get_device())
    model = model_builder.init_model(
        num_classes=classcode_builder.get_class_count(),
        network_name=hyperparameter_builder.get_network_name(),
        input_size=hyperparameter_builder.get_input_size(),
        pretrained_backbone=hyperparameter_builder.get_pretrained_backbone(),
        output_stride=hyperparameter_builder.get_output_stride(),
    ).get_model()

    training_builder = TrainingBuilder(logger
                       ).initialize(
                            epoch_total=hyperparameter_builder.get_epoch(),
                            device=hyperparameter_builder.get_device(),
                            using_amp=hyperparameter_builder.get_using_amp(),
                            loss_eps=hyperparameter_builder.get_loss_eps(),
                            weight_decay=hyperparameter_builder.get_weight_decay(),
                       ).init_model(model=model
                       ).init_optimizer(
                            optimizer_name=hyperparameter_builder.get_optimizer(),
                            lr=hyperparameter_builder.get_learning_rate()
                       ).init_criterion(
                            criterion_name=hyperparameter_builder.get_criterion()
                       ).builder()

    label_info = {'inference_info': json.dumps({
        "input_size": hyperparameter_builder.get_input_size(),
        "label_info": classcode_builder.get_label_info()
    })}
    savehook = SaveHook(
        save_epoch=hyperparameter_builder.get_save_epoch(),
        logger=logger,
        save_path=output,
        daq_old_path=hyperparameter_builder.get_daq_old_path(),
        label_info=label_info,
    )

    training_builder.train(
        train_data_loader=dataset_builder.get_train_data_loader(),
        validation_data_loader=dataset_builder.get_validation_data_loader(),
        hook=savehook
    )

except Exception as e:
    logger.error(f"Error Message : {e}")
    raise Exception(f"Train Failed, Error Message : {e}")
