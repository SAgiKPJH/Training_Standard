import os
import random
import cv2
import numpy as np
import tensorflow as tf

class Local_Tensorflow_ClassificationDatasetBuilder:
    """Local implementation for building TensorFlow classification datasets from local files."""

    def __init__(self, logger=None):
        self.__logger = logger
        self.__dataset_path = None
        self.__train_data_loader = None
        self.__validation_data_loader = None
        self.__input_size = None
        self.__normalize_mean = None
        self.__normalize_stdev = None

    def get_train_data_loader(self):
        return self.__train_data_loader

    def get_validation_data_loader(self):
        return self.__validation_data_loader

    def init_dataset_path(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        self.__dataset_path = dataset_path
        return self

    def init_transform(self, input_size, normalize_mean, normalize_stdev, augmentation=None):
        self.__input_size = input_size
        self.__normalize_mean = normalize_mean
        self.__normalize_stdev = normalize_stdev
        self.__augmentation = augmentation or {}
        return self

    def _preprocess_image(self, image_path, label):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.__input_size, self.__input_size))
        image = image.astype(np.float32) / 255.0
        return image, label

    def _create_tf_dataset(self, uri_list, label_list, batch_size, shuffle):
        from .AugmentationBuilder import build_augmentation_fn

        images = []
        labels = []
        for uri, label in zip(uri_list, label_list):
            img, lbl = self._preprocess_image(uri, label)
            images.append(img)
            labels.append(lbl)

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # Augmentation (normalize 전, 0~1 범위에서 적용)
        augment_fn = build_augmentation_fn(self.__augmentation)
        if augment_fn is not None:
            dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

        # Normalize
        mean = self.__normalize_mean
        std = self.__normalize_stdev
        dataset = dataset.map(lambda img, lbl: ((img - mean) / std, lbl), num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(uri_list))
        dataset = dataset.batch(batch_size, drop_remainder=(batch_size > 1))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def create_train_dataset(self, train_ratio, batch_size, validation_save_random, class_code_info):
        if self.__logger:
            self.__logger.info("Create Train Dataset from Local Files")

        try:
            train_data_info, validation_data_info = self._load_data(
                train_ratio, self.__dataset_path, class_code_info
            )

            self.__train_data_loader = self._create_tf_dataset(
                train_data_info[0], train_data_info[1], batch_size, shuffle=True
            )

            if len(validation_data_info[0]) > 0:
                self.__validation_data_loader = self._create_tf_dataset(
                    validation_data_info[0], validation_data_info[1], batch_size=1, shuffle=validation_save_random
                )
            else:
                self.__validation_data_loader = None

        except Exception as e:
            if self.__logger:
                self.__logger.error(f"Error Message: {e}")
            raise Exception(f"Create Train Dataset Failed, Error Message: {e}")

        return self

    def _load_data(self, train_ratio, dataset_path, class_code_info):
        train_ratio = min(1, max(0, train_ratio))

        train_uri_list = []
        train_label_list = []
        validation_uri_list = []
        validation_label_list = []

        class_dirs = [d for d in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, d))]
        class_dirs.sort()

        for class_index, class_name in enumerate(class_dirs):
            class_path = os.path.join(dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]

            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                train_uri_list.append(image_path)
                train_label_list.append(class_code_info[class_index])

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
        return self.__train_data_loader is not None

    def build(self):
        return self
