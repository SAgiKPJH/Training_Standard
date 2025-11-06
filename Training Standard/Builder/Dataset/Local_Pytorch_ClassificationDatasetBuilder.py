import os
import random
import mpp
import torch
from torch.utils.data import Dataset

class Local_Pytorch_ClassificationDatasetBuilder:
    """Local implementation for building PyTorch classification datasets from local files."""

    def __init__(self, logger=None):
        self.__logger = logger
        self.__dataset_path = None
        self.__train_data_loader = None
        self.__validation_data_loader = None
        self.__transform = None

    def get_train_data_loader(self):
        return self.__train_data_loader

    def get_validation_data_loader(self):
        return self.__validation_data_loader

    def init_dataset_path(self, dataset_path):
        """
        Initialize dataset path.

        Args:
            dataset_path: Path to local dataset directory
        """
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        self.__dataset_path = dataset_path
        return self

    def init_transform(self, input_size, normalize_mean, normalize_stdev):
        """
        Initialize image transformation pipeline.

        Args:
            input_size: Target image size
            normalize_mean: Mean for normalization
            normalize_stdev: Standard deviation for normalization
        """
        import torchvision.transforms as transforms
        from PIL import Image

        transform = transforms.Compose([
            transforms.Lambda(lambda img: Image.fromarray(img).convert("RGB")),
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Normalize(
                (normalize_mean, normalize_mean, normalize_mean),
                (normalize_stdev, normalize_stdev, normalize_stdev)
            )
        ])
        self.__transform = transform
        return self

    def create_train_dataset(self, train_ratio, batch_size, validation_save_random, class_code_info):
        """
        Create train and validation datasets from local files.

        Args:
            train_ratio: Ratio of training data (0.0 to 1.0)
            batch_size: Batch size for training
            validation_save_random: Whether to shuffle validation data
            class_code_info: Dictionary mapping class codes to indices
        """
        if self.__logger:
            self.__logger.info("Create Train Dataset from Local Files")

        try:
            train_data_info, validation_data_info = self._load_data(
                train_ratio, self.__dataset_path, class_code_info
            )

            train_dataset = ClassificationDataset(train_data_info, self.__transform)
            train_data_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )

            valid_flag = False
            if len(validation_data_info[0]) > 0:
                valid_flag = True
                validation_dataset = ClassificationDataset(validation_data_info, self.__transform)
                validation_data_loader = torch.utils.data.DataLoader(
                    validation_dataset,
                    batch_size=1,
                    shuffle=validation_save_random,
                    num_workers=0,
                    drop_last=False
                )

            self.__train_data_loader = train_data_loader
            self.__validation_data_loader = validation_data_loader if valid_flag else None

        except Exception as e:
            if self.__logger:
                self.__logger.error(f"Error Message: {e}")
            raise Exception(f"Create Train Dataset Failed, Error Message: {e}")

        return self

    def _load_data(self, train_ratio, dataset_path, class_code_info):
        """
        Load image paths and labels from local directory structure.

        Args:
            train_ratio: Ratio of training data
            dataset_path: Path to dataset directory
            class_code_info: Dictionary mapping class codes to indices

        Returns:
            Tuple of (train_data_info, validation_data_info)
        """
        train_ratio = min(1, max(0, train_ratio))

        train_uri_list = []
        train_label_list = []
        validation_uri_list = []
        validation_label_list = []

        # Get class directories
        class_dirs = [d for d in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, d))]
        class_dirs.sort()

        # Load all image paths
        for class_index, class_name in enumerate(class_dirs):
            class_path = os.path.join(dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]

            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                train_uri_list.append(image_path)
                train_label_list.append(class_code_info[class_index])

        # Split into train and validation
        validation_len = int(len(train_uri_list) * (1 - train_ratio))

        for _ in range(validation_len):
            if len(train_uri_list) == 0:
                break

            random_index = random.randrange(len(train_uri_list))
            valid_uri = train_uri_list.pop(random_index)
            valid_label = train_label_list.pop(random_index)

            validation_uri_list.append(valid_uri)
            validation_label_list.append(valid_label)

        return (train_uri_list, train_label_list), (validation_uri_list, validation_label_list)

    def success(self):
        """Check if datasets were created successfully."""
        return self.__train_data_loader is not None

    def build(self):
        return self


class ClassificationDataset(Dataset):
    """PyTorch Dataset for classification from local files."""

    def __init__(self, data_info, transform):
        self.__uri_list = data_info[0]
        self.__label_list = data_info[1]
        self.__transform = transform

    def __len__(self):
        return len(self.__uri_list)

    def __getitem__(self, index):
        uri = self.__uri_list[index]
        image = mpp.intel64.load(uri, False)
        image = self.__transform(image)
        label = self.__label_list[index]

        return image, label