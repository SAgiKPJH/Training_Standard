import os
import uuid
from torch.utils.data import Dataset, DataLoader
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ClassificationDataset(Dataset):
    def __init__(self, train_dataset, transform):
        self.dataset = train_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]['image']
        label = self.dataset[index]['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class Classification_Dataset_Builder:
    def __init__(self, keyword_arguments):
        self.__gt_dataset_id = keyword_arguments['gt_dataset_id']
        self.__local_download_path = None
        self.__train_data_loader = None
        self.__validation_data_loader = None
        self.__classification_gts = None
        self.__label_info = None
        self.__class_code_info = None
        self.__num_classes = None

    def initialize(self):
        self.__local_download_path = os.path.join(f"/temp/{uuid.uuid4()}")
        return self

    def init_label_data(self, operation_channel, access_token):
        if operation_channel:
            # 실제 서비스에서 라벨 정보를 가져오는 로직
            label_info = {}  # 서비스에서 가져온 라벨 정보
            class_code_info = {}  # 서비스에서 가져온 클래스 코드 정보
            num_classes = 0  # 서비스에서 가져온 클래스 수
        else: 
            # 테스트용 더미 데이터
            label_info = {"label_0": {"name": "class1"}, "label_1": {"name": "class2"}}
            class_code_info = {"class1": 0, "class2": 1}
            num_classes = 2
        
        self.__label_info = label_info
        self.__class_code_info = class_code_info
        self.__num_classes = num_classes
        return self
    
    def init_dataset_gts(self, operation_channel, access_token):
        if operation_channel:
            # 실제 서비스에서 데이터셋 정보를 가져오는 로직
            self.__classification_gts = []  # 서비스에서 가져온 데이터셋
        else:
            # 테스트용 더미 데이터
            self.__classification_gts = [
                {"image": np.random.rand(224, 224, 3), "label": 0},
                {"image": np.random.rand(224, 224, 3), "label": 1}
            ]
        return self

    def create_train_dataset(self, operation_channel, access_token, train_ratio, transform, batch_size, validation_save_random):
        logger.info("Create Train Dataset")
        try:
            # 데이터 다운로드 및 전처리
            train_data, valid_data = self.data_download(
                train_ratio,
                self.__local_download_path,
                self.__classification_gts,
                self.__class_code_info,
                operation_channel,
                access_token
            )

            # DataLoader 생성
            train_dataset = ClassificationDataset(train_data, transform)
            valid_dataset = ClassificationDataset(valid_data, transform)

            self.__train_data_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

            self.__validation_data_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

            return self

        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            return False
        
    def get_label_info(self):
        return self.__label_info
    
    def get_class_code_info(self):
        return self.__class_code_info

    def get_num_classes(self):
        return self.__num_classes
    
    def data_download(self, train_ratio, local_download_path, classification_gts, class_code_info, operation_channel, access_token):
        train_ratio = min(1, train_ratio)
        total_len = len(classification_gts)
        train_len = int(total_len * train_ratio)
        validation_len = total_len - train_len

        # 데이터 셔플
        indices = np.random.permutation(total_len)
        train_indices = indices[:train_len]
        valid_indices = indices[train_len:]

        train_data = [classification_gts[i] for i in train_indices]
        valid_data = [classification_gts[i] for i in valid_indices]

        return train_data, valid_data

    def get_train_loader(self):
        return self.__train_data_loader
    
    def get_validation_loader(self):
        return self.__validation_data_loader
    
    def temp_folder_delete(self):
        if os.path.exists(self.__local_download_path):
            import shutil
            shutil.rmtree(self.__local_download_path)

    def build(self):
        return self.__train_data_loader is not None and self.__validation_data_loader is not None
