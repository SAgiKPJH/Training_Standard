import json


class Json_HyperparameterBuilder:
    def __init__(self, hyperparameter: str):
        self.__p = json.loads(hyperparameter) if isinstance(hyperparameter, str) else hyperparameter

    def build(self) -> 'Json_HyperparameterBuilder':
        return self

    def get_pretrained_model(self) -> str:
        return self.__p.get('pretrained_model', '')

    def get_resume_path(self) -> str:
        return self.__p.get('resume_path', '')

    def get_rec_char_dict_path(self) -> str:
        return self.__p.get('rec_char_dict_path', 'data/en_dict.txt')

    def get_save_dir(self) -> str:
        return self.__p.get('save_dir', 'output/')

    def get_train_data_dir(self) -> str:
        return self.__p.get('train_data_dir', 'data/')

    def get_train_label_file(self) -> str:
        return self.__p.get('train_label_file', 'data/train_labels.txt')

    def get_val_data_dir(self) -> str:
        return self.__p.get('val_data_dir', 'data/')

    def get_val_label_file(self) -> str:
        return self.__p.get('val_label_file', 'data/val_labels.txt')

    def get_image_shape(self) -> list:
        return self.__p.get('image_shape', [3, 48, 320])

    def get_max_text_length(self) -> int:
        return int(self.__p.get('max_text_length', 50))

    def get_epoch_num(self) -> int:
        return int(self.__p.get('epoch_num', 100))

    def get_save_epoch_step(self) -> int:
        return int(self.__p.get('save_epoch_step', 10))

    def get_batch_size(self) -> int:
        return int(self.__p.get('batch_size', 64))

    def get_learning_rate(self) -> float:
        return float(self.__p.get('learning_rate', 0.0005))

    def get_use_space_char(self) -> bool:
        return bool(self.__p.get('use_space_char', False))

    def get_use_guided_training(self) -> bool:
        return bool(self.__p.get('use_guided_training', False))

    def get_device(self) -> str:
        return self.__p.get('device', 'gpu').lower()

    def get_gpu_id(self) -> str:
        return str(self.__p.get('gpu_id', '0'))

    def get_paddleocr_home(self) -> str:
        return self.__p.get('paddleocr_home', '')
