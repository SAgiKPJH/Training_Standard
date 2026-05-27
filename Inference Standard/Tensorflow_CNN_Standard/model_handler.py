import json
import os
import cv2
import pickle
import numpy as np
import tensorflow as tf


class ModelHandler:
    def __init__(self, data, context):
        model_config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
        with open(model_config_file_path, "r", encoding='utf-8') as config_file:
            __config = json.load(config_file)

        self.__device = __config['device']

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), __config.get('model_dir', 'data/model'))
        self.__model = tf.keras.models.load_model(model_path)

        # 모델 내장 inference_info에서 꺼내기
        inference_info_raw = json.loads(self.__model.inference_info.numpy().decode('utf-8'))
        # inference_info 구조: {'label_info': '{...}', 'inference_info': '{...}'}
        # label_info 키의 값이 'inference_info' 키를 가진 JSON 문자열
        label_info_data = inference_info_raw.get('label_info', '{}')
        if isinstance(label_info_data, str):
            label_info_data = json.loads(label_info_data)

        # label_info_data 내에 'inference_info' 키가 있으면 그 안에 input_size, label_info가 있음
        if 'inference_info' in label_info_data:
            inner = label_info_data['inference_info']
            if isinstance(inner, str):
                inner = json.loads(inner)
            self.__input_size = inner['input_size']
            self.__label_info = inner['label_info']
        else:
            self.__input_size = label_info_data.get('input_size', 299)
            self.__label_info = label_info_data.get('label_info', {})

    @property
    def input_size(self):
        return self.__input_size

    def __preprocess(self, image):
        """이미지 전처리: resize + normalize"""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.__input_size, self.__input_size))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.expand_dims(image, axis=0)
        return image

    def __call__(self, data, context):
        data = pickle.loads(data)

        input_data = self.__preprocess(data)

        with tf.device(f'/{self.__device}:0' if self.__device != 'cpu' else '/cpu:0'):
            predict = self.__model(input_data, training=False)
            predict = tf.nn.softmax(predict, axis=1).numpy()

        output = list()
        for i in range(self.__label_info['label_count']):
            current_label = self.__label_info[f'label_{i}']
            current_label['score'] = float(predict[0][i])
            output.append(current_label)

        output = sorted(output, key=lambda x: x['score'], reverse=True)

        return pickle.dumps(output), context


if __name__ == "__main__":
    import cv2
    import numpy as np

    handler = ModelHandler(None, None)

    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image.png")
    test_data = cv2.imread(test_data_path, 1) if os.path.exists(test_data_path) else None

    if test_data is None:
        size = handler.input_size
        print(f"[WARN] Test image not found at {test_data_path}. Generating random image ({size}x{size}).")
        test_data = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)

    test_data_pickle = pickle.dumps(test_data)

    # =================================================================== #

    inference_result, context = handler(test_data_pickle, None)

    # =================================================================== #

    inference_result = pickle.loads(inference_result)
    print(*inference_result, sep="\n")
