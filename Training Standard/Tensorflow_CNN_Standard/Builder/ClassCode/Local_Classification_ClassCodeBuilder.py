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
        """
        Initialize label data from local directory structure.
        Expects directory structure: dataset_path/class_name/images...

        Args:
            dataset_path: Path to dataset directory containing class folders
        """
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        if not os.path.isdir(dataset_path):
            raise ValueError(f"Dataset path is not a directory: {dataset_path}")

        # Get class directories
        class_dirs = [d for d in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, d))]

        # 숫자 기준 정렬 (폴더명 prefix가 숫자면 int로, 아니면 문자열)
        def _sort_key(name):
            prefix = name.split('_')[0] if '_' in name else name
            try:
                return (0, int(prefix), name)
            except ValueError:
                return (1, 0, name)

        class_dirs.sort(key=_sort_key)

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
