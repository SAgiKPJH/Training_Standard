import os
import uuid
import shutil
import random

import tensorflow as tf
import mpp
from mpp.daq import protos

from google.protobuf.wrappers_pb2 import StringValue


class DAQ_Tensorflow_ClassificationDatasetBuilder:
    def __init__(self, logger=None):
        self.__operation_channel = None
        self.__access_token = None
        self.__logger = logger

        import sys
        _main_dir = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
        self.__local_download_path = os.path.join(_main_dir, "temp", str(uuid.uuid4()))

        self.__classification_gts = None
        self.__train_data_loader = None
        self.__validation_data_loader = None

        self.__input_size = None
        self.__normalize_mean = None
        self.__normalize_stdev = None

    def get_train_data_loader(self):
        return self.__train_data_loader

    def get_validation_data_loader(self):
        return self.__validation_data_loader

    def init_url_info(self, operation_channel, access_token):
        self.__operation_channel = operation_channel
        self.__access_token = access_token
        return self

    def init_dataset_gts(self, gt_dataset_id):
        if not self.__operation_channel:
            self.__classification_gts = gt_dataset_id
            return self

        stub = protos.daq_dataset_classification_gt_dataset_api_v1_pb2_grpc.ClassificationGtDatasetServiceStub(self.__operation_channel)
        query_parameter = protos.daq_common_pb2.QueryParameter(
            page_index=0,
            page_size=-1,
            where=StringValue(value=f"GtDatasetId=\"{gt_dataset_id}\""),
            order_by=None)

        response = stub.ListClassificationGts(
            request=protos.daq_dataset_classification_gt_dataset_api_v1_pb2.ListClassificationGtsRequest(
                query_parameter=query_parameter, with_image=False),
            metadata=[('authorization', f'Bearer {self.__access_token}')])

        self.__classification_gts = response.classification_gts
        return self

    def init_transform(self, input_size, normalize_mean, normalize_stdev, augmentation=None):
        self.__input_size = input_size
        self.__normalize_mean = normalize_mean
        self.__normalize_stdev = normalize_stdev
        self.__augmentation = augmentation or {}
        return self

    def _load_image(self, path, label):
        """경로 텐서를 받아 디스크에서 1장 디코딩하고 resize + [0,1] 스케일링.
        tf.data.map에서 호출되어 graph mode로 실행됨 (lazy load)."""
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [self.__input_size, self.__input_size])
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def _create_tf_dataset(self, uri_list, label_list, batch_size, shuffle):
        """Lazy load 기반 tf.data 파이프라인.
        경로 문자열만 메모리에 두고, .map에서 batch 단위로 디스크에서 1장씩 로드한다.
        이미지 N장 전체를 한 번에 RAM에 올리지 않으므로 대용량 데이터셋도 안전.
        """
        from .TF_AugmentationBuilder import build_augmentation_fn

        if len(uri_list) == 0:
            raise ValueError("uri_list is empty")

        paths = tf.constant(uri_list, dtype=tf.string)
        labels = tf.constant(label_list, dtype=tf.int64)
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

        if shuffle:
            # paths/labels만 셔플 (이미지 아님 → 메모리 부담 없음)
            dataset = dataset.shuffle(buffer_size=len(uri_list), reshuffle_each_iteration=True)

        # Lazy load: 실제 이미지는 여기서 batch 직전에 1장씩 로드
        dataset = dataset.map(self._load_image, num_parallel_calls=tf.data.AUTOTUNE)

        augment_fn = build_augmentation_fn(self.__augmentation)
        if augment_fn is not None:
            dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

        mean = self.__normalize_mean
        std = self.__normalize_stdev
        dataset = dataset.map(lambda img, lbl: ((img - mean) / std, lbl), num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(batch_size, drop_remainder=(batch_size > 1))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def create_train_dataset(self, train_ratio, batch_size, validation_save_random, class_code_info):
        if self.__logger:
            self.__logger.info("Create Train Dataset")
        try:
            train_data_info, validation_data_info = self.data_download(
                train_ratio,
                self.__local_download_path,
                self.__classification_gts,
                class_code_info
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
                self.__logger.error(f"Error Message : {e}")
            raise Exception(f"Create Train Dataset Failed, Error Message : {e}")

        return self

    def data_download(self, train_ratio, local_download_path, classification_gts, class_code_info):
        train_ratio = min(1, train_ratio)

        train_uri_list = list()
        train_label_list = list()
        validation_uri_list = list()
        validation_label_list = list()

        if self.__operation_channel:
            validation_len = int(len(classification_gts) * (1 - train_ratio))
            for i in range(len(classification_gts)):
                image_id = classification_gts[i].image_id
                class_code = classification_gts[i].class_code.value

                uri = f"dataset:///?image_id={image_id}"
                image = mpp.daq.intel64.load(uri, False, channel=self.__operation_channel, access_token=self.__access_token)
                if image is None or (hasattr(image, 'size') and image.size == 0):
                    if self.__logger: self.__logger.warning(f"[{i}] Image load failed, skipping: image_id={image_id}")
                    continue
                download_path = os.path.join(local_download_path, f"{image_id}.png")
                mpp.intel64.save(image, download_path)

                train_uri_list.append(download_path)
                train_label_list.append(class_code_info[class_code])
        else:
            validation_len = int(len(train_uri_list) * (1 - train_ratio))
            class_code_list = os.listdir(classification_gts)
            for index in range(len(class_code_list)):
                file_list = os.listdir(os.path.join(classification_gts, class_code_list[index]))
                for filename in file_list:
                    image_path = os.path.join(classification_gts, class_code_list[index], filename)
                    train_uri_list.append(image_path)
                    train_label_list.append(class_code_info[index])

        for _ in range(validation_len):
            random_index = random.randrange(len(train_uri_list))
            valid_uri = train_uri_list.pop(random_index)
            valid_label = train_label_list.pop(random_index)
            validation_uri_list.append(valid_uri)
            validation_label_list.append(valid_label)

        return (train_uri_list, train_label_list), (validation_uri_list, validation_label_list)

    def temp_folder_delete(self):
        if os.path.exists(self.__local_download_path):
            shutil.rmtree(self.__local_download_path)
            if self.__logger:
                self.__logger.info("Temp Folder Delete")

    def success(self):
        return self.__train_data_loader is not None and self.__validation_data_loader is not None

    def build(self):
        return self
