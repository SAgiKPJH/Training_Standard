parameters = '''{
    "version" : "Pytorch_CNN_Standard_v1.0.2",
    "hyperparameter":{
        "network_name" : "inceptionv3",
        "epoch" : 20,
        "save_epoch" : 2,
        "batch_size" : 2,
        "lr" : 1e-3,
        "optimizer_name" : "Adam",
        "criterion" : "CrossEntropyLoss",
        "input_size" : 299,
        "normalize_mean" : 0.5,
        "normalize_stdev" : 0.5,
        "using_gpu" : false,
        "using_amp" : true,
        "train_ratio" : 0.8,
        "validation_save_random" : false,
        "debug" : false,
        "daq_old_path" : false,
        "resume_path" : ""
    }
}'''
import json
hyperparameter = json.dumps(json.loads(parameters)['hyperparameter'])

import logging
logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

dataset = "D:\\Code\\Training_Standard\\create_dataset\\dataset"
output = "D:\\Code\\Training_Standard\\create_dataset\\output"

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
        self.__best_loss = None
        self.__best_acc = None
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

        if (self.__save_epoch > 0 and epoch % self.__save_epoch == 0) or (self.__save_epoch == 0 and epoch == total_epoch):
            if self.__daq_old_path:
                self.__save_builder.save_model(epoch=epoch, model=model)
            else:
                self.__save_builder.save_model(f"{epoch}/model.pth", model=model)

        # Best loss 모델 저장
        loss = validation_loss if validation_loss is not None else train_loss
        if self.__best_loss is None or loss < self.__best_loss:
            self.__best_loss = loss
            logger.info(f"  ★ Best loss model updated (loss: {loss:.6f})")
            if self.__daq_old_path:
                self.__save_builder.save_model(file_full_path="best_loss/model/model.pth", model=model)
            else:
                self.__save_builder.save_model(f"best_loss/model.pth", model=model)

        # Best accuracy 모델 저장
        acc = (100 - validation_loss) if validation_loss is not None else (100 - train_loss)
        if self.__best_acc is None or acc > self.__best_acc:
            self.__best_acc = acc
            logger.info(f"  ★ Best acc model updated (acc: {acc:.2f}%)")
            if self.__daq_old_path:
                self.__save_builder.save_model(file_full_path="best_acc/model/model.pth", model=model)
            else:
                self.__save_builder.save_model(f"best_acc/model.pth", model=model)

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

    import os, torch
    resume_path = hyperparameter_builder.get_resume_path()
    if resume_path:
        resume_path = os.path.abspath(resume_path) if not os.path.isabs(resume_path) else resume_path
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume path not found: {resume_path}")
        device = hyperparameter_builder.get_device()
        try:
            loaded = torch.jit.load(resume_path, map_location=device)
            model.load_state_dict(loaded.state_dict())
        except Exception:
            state_dict = torch.load(resume_path, map_location=device, weights_only=False)
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict.state_dict())
        logger.info(f"Resumed from: {resume_path}")

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