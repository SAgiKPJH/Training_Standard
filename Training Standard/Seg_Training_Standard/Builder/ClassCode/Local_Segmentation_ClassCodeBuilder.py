import os
import json


class Local_Segmentation_ClassCodeBuilder:
    """Local Segmentation ClassCode Builder.

    데이터셋 구조:
        dataset_path/
        ├── image/        # 원본 이미지
        │   ├── 0.png
        │   └── ...
        ├── mask/         # 클래스 인덱스 마스크 (grayscale, 픽셀값 = class id)
        │   ├── 0.png
        │   └── ...
        └── label.json    # (optional) {"0": "background", "1": "object1", ...}

    label.json이 없으면 mask 디렉터리의 모든 픽셀값을 스캔하여 라벨 정보를 구성합니다.
    """

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

        label_json_path = os.path.join(dataset_path, 'label.json')
        if os.path.exists(label_json_path):
            with open(label_json_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            items = sorted(((int(k), v) for k, v in raw.items()), key=lambda x: x[0])
        else:
            items = self._scan_mask_labels(dataset_path)

        num_classes = len(items)
        label_info = {"label_count": num_classes}
        class_code_info = {}
        for i, (label_id, label_name) in enumerate(items):
            label_info[f'label_{i}'] = {"code": int(label_id), "name": str(label_name)}
            class_code_info[int(label_id)] = i

        self.__label_info = label_info
        self.__class_code_info = class_code_info
        self.__num_classes = num_classes
        return self

    def _scan_mask_labels(self, dataset_path):
        import numpy as np
        import cv2
        mask_dir = os.path.join(dataset_path, 'mask')
        if not os.path.isdir(mask_dir):
            raise ValueError(f"mask directory not found: {mask_dir}")

        unique_values = set()
        for filename in os.listdir(mask_dir):
            if not filename.lower().endswith(('.png', '.bmp', '.tiff', '.tif', '.jpg', '.jpeg')):
                continue
            mask = cv2.imread(os.path.join(mask_dir, filename), 0)
            if mask is None:
                continue
            unique_values.update(np.unique(mask).tolist())

        return [(int(v), f"class_{v}") for v in sorted(unique_values)]

    def build(self):
        return self
