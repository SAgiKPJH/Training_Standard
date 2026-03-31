import os

class Local_Classification_ClassCodeBuilder:
    """Local implementation for building classification class codes from local directory structure."""

    def __init__(self):
        self.__label_info = None
        self.__class_code_info = None
        self.__num_classes = None

    def get_label_info(self):
        return self.__label_info

    def get_class_code_info(self):
        return self.__class_code_info

    def get_class_count(self):
        return self.__num_classes

    def init_label_data(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        if not os.path.isdir(dataset_path):
            raise ValueError(f"Dataset path is not a directory: {dataset_path}")

        class_dirs = [d for d in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, d))]
        class_dirs.sort()

        num_classes = len(class_dirs)
        label_info = {"label_count": num_classes}
        class_code_info = {}

        for i, class_name in enumerate(class_dirs):
            label_info[f'label_{i}'] = {"code": i, "name": class_name}
            class_code_info[i] = i

        self.__label_info = label_info
        self.__class_code_info = class_code_info
        self.__num_classes = num_classes

        return self

    def build(self):
        return self
