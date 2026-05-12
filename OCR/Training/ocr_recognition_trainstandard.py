##UI{"$ref": "ui.json"}IU

##!--{"Name":"hyperparameter","Type":"epoch_num","Key":"epoch_num","Value":"","Category":""}
##!--{"Name":"hyperparameter","Type":"batch_size","Key":"batch_size","Value":"","Category":""}
##!--{"Name":"result","Type":"result","Key":"id","Value":"","Category":""}
##!--{"Name":"authentication","Type":"system_address","Key":"operation_service_address","Value":"","Category":""}
##!--{"Name":"authentication","Type":"access_token","Key":"access_token","Value":"","Category":""}
##!--{"Name":"gt_dataset","Type":"gt_dataset","Key":"gt_dataset_id","Value":"","Category":""}
##$--

parameters = '''{
    "version": "OCR_Training_Standard_v1.0.0",
    "hyperparameter": {
        "pretrained_model":    "data/pretrained/en_PP-OCRv3_rec_train/best_accuracy",
        "resume_path":         "",
        "rec_char_dict_path":  "data/en_dict.txt",
        "save_dir":            "output/",
        "image_shape":         [3, 48, 320],
        "max_text_length":     50,
        "epoch_num":           100,
        "save_epoch_step":     10,
        "batch_size":          64,
        "learning_rate":       0.0005,
        "use_space_char":      false,
        "use_guided_training": false,
        "device":              "gpu",
        "gpu_id":              "0",
        "paddleocr_home":      ""
    },
    "authentication": {
        "operation_service_address": "",
        "access_token": ""
    },
    "result": {
        "id": "",
        "volume_id": "default"
    },
    "gt_dataset": {
        "gt_dataset_id": ""
    },
    "chunk_size": 100000
}'''
##$--

import json
import logging

logger = globals().get('JOB_LOGGER', logging.getLogger())
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

from Builder import Json_HyperparameterBuilder, TrainHook


def RecipeRun(**kwargs):
    from Builder import Operation_Builder
    from Builder import DAQ_OCR_DatasetBuilder
    from Builder import PaddleOCR_TrainingBuilder

    operation_builder = Operation_Builder(**kwargs).initialize().build()
    hp_builder = Json_HyperparameterBuilder(
        operation_builder.get_json_hyperparameter()
    ).build()

    # ClassCode 불필요 (rec task는 클래스 없음)
    dataset_builder = DAQ_OCR_DatasetBuilder(logger=logger
    ).init_url_info(
        operation_channel=operation_builder.get_operation_channel(),
        access_token=operation_builder.get_access_token()
    ).init_dataset_gts(
        gt_dataset_id=operation_builder.get_gt_dataset_id()
    ).init_char_dict(
        rec_char_dict_path=hp_builder.get_rec_char_dict_path()
    ).build()

    if not dataset_builder.success():
        logger.error("Dataset Build Failed")
        dataset_builder.temp_folder_delete()
        return None

    training_builder = PaddleOCR_TrainingBuilder(logger
    ).initialize(
        hp_builder=hp_builder,
        base_dir=dataset_builder.get_base_dir()
    ).builder()

    class SaveHook(TrainHook):
        def training_start(self):
            logger.info("Training Started.")

        def training_end(self):
            logger.info("Training Ended.")
            from Builder import DAQ_OCR_SaveBuilder
            save_builder = DAQ_OCR_SaveBuilder(
            ).init_save_url(
                bucket_url=operation_builder.get_bucket_url(),
                operation_channel=operation_builder.get_operation_channel(),
                access_token=operation_builder.get_access_token(),
                chunk_size=operation_builder.get_chunk_size()
            ).build()
            save_builder.save_model_dir(dataset_builder.get_save_dir())
            save_builder.save_file_index()

    try:
        training_builder.train(
            dataset_builder=dataset_builder,
            hook=SaveHook()
        )
    except Exception as e:
        logger.error(f"Error Message : {e}")
        raise Exception(f"Train Failed, Error Message : {e}")
    finally:
        dataset_builder.temp_folder_delete()


if __name__ == "__main__":
    kwargs = json.loads(parameters)
    kwargs['authentication']['operation_service_address'] = ""
    kwargs['authentication']['access_token'] = ""
    kwargs['gt_dataset']['gt_dataset_id'] = r""
    import random
    kwargs['result']['id'] = f"test_result_volume_{random.randint(1000, 9999)}"

    RecipeRun(**kwargs)
