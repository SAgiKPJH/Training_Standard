import cv2
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import load_img, img_to_array

# GPU 메모리 제한
memory_limit = 3  # GB 기준
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit * 1024)]
            )
    except RuntimeError as e:
        print(e)
        pass

class ModelHandler:
    def __init__(self, data=None, context=None):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "model.h5")
        self.model = keras.models.load_model(model_path)

    def test(self):
        input_shape = self.model.input_shape[1:]  # (224, 224, 3) expected
        dummy = np.zeros(input_shape)
        dummy = np.expand_dims(dummy, axis=0)
        self.model.predict(dummy, batch_size=1)

    def __call__(self, data, context):
        # 역직렬화 및 전처리
        image_np = pickle.loads(data)  # HxWx3 형태의 np.uint8
        image = cv2.resize(image_np, (224, 224))
        image_tensor = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)

        # 추론
        prediction = self.model.predict(image_tensor)
        pred_indices = np.argmax(prediction[0], axis=-1)  # (224, 224)

        # 결과 패킹
        result = {
            'label_mask': pred_indices.tolist()  # JSON 직렬화 호환
        }

        return pickle.dumps(result), context

if __name__ == '__main__':
    handler = ModelHandler()
    handler.test()  # Warm-up