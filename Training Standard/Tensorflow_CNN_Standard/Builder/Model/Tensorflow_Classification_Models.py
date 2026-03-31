import tensorflow as tf

class Tensorflow_Classification_Models:
    def __init__(self):
        self.__device = '/cpu:0'
        self.__model = None
        self.__network_map = {
            "resnet50": tf.keras.applications.ResNet50,
            "resnet101": tf.keras.applications.ResNet101,
            "resnet152": tf.keras.applications.ResNet152,
            "efficientnet_b0": tf.keras.applications.EfficientNetB0,
            "efficientnet_b1": tf.keras.applications.EfficientNetB1,
            "efficientnet_b2": tf.keras.applications.EfficientNetB2,
            "efficientnet_b3": tf.keras.applications.EfficientNetB3,
            "efficientnet_b4": tf.keras.applications.EfficientNetB4,
            "efficientnet_b5": tf.keras.applications.EfficientNetB5,
            "efficientnet_b6": tf.keras.applications.EfficientNetB6,
            "efficientnet_b7": tf.keras.applications.EfficientNetB7,
            "efficientnet_v2_s": tf.keras.applications.EfficientNetV2S,
            "efficientnet_v2_m": tf.keras.applications.EfficientNetV2M,
            "efficientnet_v2_l": tf.keras.applications.EfficientNetV2L,
            "inceptionv3": tf.keras.applications.InceptionV3,
            "mobilenet_v2": tf.keras.applications.MobileNetV2,
            "mobilenet_v3_small": tf.keras.applications.MobileNetV3Small,
            "mobilenet_v3_large": tf.keras.applications.MobileNetV3Large,
        }

    def init_device(self, device):
        if device == 'cuda' or device == '/gpu:0':
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                raise Exception("GPU is not available")
            self.__device = '/gpu:0'
        else:
            self.__device = '/cpu:0'
        return self

    def init_model(self, num_classes, network_name='efficientnet_b0', input_size=299):
        if network_name not in self.__network_map:
            raise ValueError(
                f"Unsupported network name: {network_name}. "
                f"Supported names are: {list(self.__network_map.keys())}"
            )

        with tf.device(self.__device):
            inputs = tf.keras.layers.Input(shape=(input_size, input_size, 3))
            base_model = self.__network_map[network_name](
                include_top=False,
                weights=None,
                input_shape=(input_size, input_size, 3),
            )
            x = base_model(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(num_classes)(x)
            self.__model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return self

    def get_model(self):
        return self.__model
