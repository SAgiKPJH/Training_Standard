parameters = '''{
    "hyperparameter":{
        "network_name" : "efficientnet_b0",
        "epoch" : 5,
        "save_epoch" : 2,
        "batch_size" : 2,
        "lr" : 1e-3,
        "optimizer_name" : "Adam",
        "input_size" : 299,
        "normalize_mean" : 0.5,
        "normalize_stdev" : 0.5,
        "using_gpu" : true,
        "using_amp" : true,
        "train_ratio" : 0.8,
        "validation_save_random" : false,
        "debug" : false,
        "daq_old_path" : false
    }
}'''
import json
hyperparameter = json.dumps(json.loads(parameters)['hyperparameter'])

import logging
logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

dataset = "D:\\test\\Dataset"
output = "D:\\test\\Output"

from Builder import Local_Classification_ClassCodeBuilder
classcode_builder = Local_Classification_ClassCodeBuilder().init_label_data(dataset_path=dataset).build()

from Builder import Json_HyperparameterBuilder
hyperparameter_builder = Json_HyperparameterBuilder(hyperparameter).build()

from Builder import Local_Pytorch_ClassificationDatasetBuilder
dataset_builder = Local_Pytorch_ClassificationDatasetBuilder(logger=logger).init_dataset_path(dataset_path=dataset
                  ).init_transform(
                      input_size=hyperparameter_builder.get_input_size(),
                      normalize_mean=hyperparameter_builder.get_normalize_mean(),
                      normalize_stdev=hyperparameter_builder.get_normalize_stdev()
                  ).create_train_dataset(
                      train_ratio=hyperparameter_builder.get_train_ratio(),
                      batch_size=hyperparameter_builder.get_batch_size(),
                      validation_save_random=hyperparameter_builder.get_validation_save_random(),
                      class_code_info=classcode_builder.get_class_code_info()
                  )

if dataset_builder.success() is False:
    if logger:logger.error(f"Dataset Build Failed")
    raise RuntimeError("Dataset Build Failed")

from Builder import TrainHook
class SaveHook(TrainHook):
    def __init__(self, save_epoch, logger, save_path, daq_old_path=False, label_info=None, etc=None):
        self.__save_epoch = save_epoch
        self.__save_path = save_path
        self.__loss = None
        self.__daq_old_path = daq_old_path

        from Builder import DAQ_MoritoringBuilder
        self.__monitoring_builder = DAQ_MoritoringBuilder(logger).builder()

        if daq_old_path:
            from Builder import Local_SaveBuilder_DAQ_OLD
            self.__save_builder = Local_SaveBuilder_DAQ_OLD(
                    ).init_inference_info(
                        label_info = label_info,
                        etc=etc
                    ).init_save_url(
                        save_path=self.__save_path
                    ).build()
        else:
            from Builder import Local_SaveBuilder
            self.__save_builder = Local_SaveBuilder(
                    ).init_inference_info(
                        label_info = label_info,
                        etc=etc
                    ).init_save_url(
                        save_path=self.__save_path
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

        if epoch % self.__save_epoch == 0:
            if self.__daq_old_path:
                self.__save_builder.save_model(epoch=epoch, model=model)
            else:
                self.__save_builder.save_model(f"{epoch}/model.pth", model=model)

        loss = validation_loss if validation_loss is not None else train_loss
        if self.__loss is None or loss < self.__loss:
            self.__loss = loss
            if self.__daq_old_path:
                self.__save_builder.save_model(file_full_path="best/model/model.h5", model=model)
            else:
                self.__save_builder.save_model(f"best/model.pth", model=model)

    def training_start(self):
        logger.info("Training Started.")
    def training_end(self):
        logger.info("Training Ended.")
        self.__save_builder.save_file_index()


from Builder import Pytorch_Classification_Models as Model
from Builder import Pytorch_TrainingBuilder
from Builder import Pytorch_TrainingBuilder_Debug
try:
    model = Model().init_device(hyperparameter_builder.get_device()
        ).init_model(
            num_classes=classcode_builder.get_class_count(),
            network_name=hyperparameter_builder.get_network_name()
        ).get_model()

    TrainingBuilder = Pytorch_TrainingBuilder_Debug if hyperparameter_builder.get_debug() else Pytorch_TrainingBuilder
    training_builder = TrainingBuilder(logger
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

    label_info = {'inference_info' : json.dumps({"input_size": hyperparameter_builder.get_input_size(), "label_info": classcode_builder.get_label_info()})}
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