import tensorflow as tf

# tf.keras.applications에서 가져올 모델 이름 매핑
_TF_MODELS = {
    # ResNet
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
    "resnet152": "ResNet152",
    "resnet50_v2": "ResNet50V2",
    "resnet101_v2": "ResNet101V2",
    "resnet152_v2": "ResNet152V2",
    # Inception
    "inceptionv3": "InceptionV3",
    "inception_resnet_v2": "InceptionResNetV2",
    # EfficientNet
    "efficientnet_b0": "EfficientNetB0",
    "efficientnet_b1": "EfficientNetB1",
    "efficientnet_b2": "EfficientNetB2",
    "efficientnet_b3": "EfficientNetB3",
    "efficientnet_b4": "EfficientNetB4",
    "efficientnet_b5": "EfficientNetB5",
    "efficientnet_b6": "EfficientNetB6",
    "efficientnet_b7": "EfficientNetB7",
    "efficientnet_v2_s": "EfficientNetV2S",
    "efficientnet_v2_m": "EfficientNetV2M",
    "efficientnet_v2_l": "EfficientNetV2L",
    # VGG
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    # DenseNet
    "densenet121": "DenseNet121",
    "densenet169": "DenseNet169",
    "densenet201": "DenseNet201",
    # MobileNet
    "mobilenet": "MobileNet",
    "mobilenet_v2": "MobileNetV2",
    "mobilenet_v3_small": "MobileNetV3Small",
    "mobilenet_v3_large": "MobileNetV3Large",
    # NASNet
    "nasnet_mobile": "NASNetMobile",
    "nasnet_large": "NASNetLarge",
    # Xception
    "xception": "Xception",
    # ConvNeXt
    "convnext_tiny": "ConvNeXtTiny",
    "convnext_small": "ConvNeXtSmall",
    "convnext_base": "ConvNeXtBase",
    "convnext_large": "ConvNeXtLarge",
    "convnext_xlarge": "ConvNeXtXLarge",
}


def _resolve_tf_model(network_name: str):
    """network_name에 해당하는 tf.keras.applications 모델을 반환. 없으면 None."""
    if network_name not in _TF_MODELS:
        return None
    attr_name = _TF_MODELS[network_name]
    return getattr(tf.keras.applications, attr_name, None)


class Tensorflow_Classification_Models:
    def __init__(self):
        self.__device = '/cpu:0'
        self.__model = None

    def init_device(self, device):
        if device == 'cuda' or device == '/gpu:0':
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                raise Exception("GPU is not available")
            self.__device = '/gpu:0'
        else:
            self.__device = '/cpu:0'
        return self

    def init_model(self, num_classes, network_name=None, input_size=299):
        if network_name is None:
            raise ValueError("network_name is required")

        model_fn = _resolve_tf_model(network_name)
        if model_fn is None:
            available = [n for n in _TF_MODELS if _resolve_tf_model(n) is not None]
            raise ValueError(
                f"Unsupported network: '{network_name}'. "
                f"This TensorFlow (v{tf.__version__}) supports: {available}"
            )

        with tf.device(self.__device):
            inputs = tf.keras.layers.Input(shape=(input_size, input_size, 3))
            base_model = model_fn(
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
