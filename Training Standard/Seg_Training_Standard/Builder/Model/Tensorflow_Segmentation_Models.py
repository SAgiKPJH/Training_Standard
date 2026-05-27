import tensorflow as tf


# tf.keras.applications backbone 매핑 (CNN과 동일)
_TF_BACKBONES = {
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
    "resnet152": "ResNet152",
    "mobilenet_v2": "MobileNetV2",
    "mobilenet_v3_small": "MobileNetV3Small",
    "mobilenet_v3_large": "MobileNetV3Large",
    "efficientnet_b0": "EfficientNetB0",
    "efficientnet_b1": "EfficientNetB1",
    "efficientnet_b2": "EfficientNetB2",
    "efficientnet_b3": "EfficientNetB3",
    "xception": "Xception",
}


def _resolve_backbone(name: str):
    if name not in _TF_BACKBONES:
        return None
    return getattr(tf.keras.applications, _TF_BACKBONES[name], None)


def _build_aspp(x, filters=256, dilations=(6, 12, 18)):
    """Atrous Spatial Pyramid Pooling."""
    h, w = x.shape[1], x.shape[2]
    layers = [tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)]
    for d in dilations:
        layers.append(tf.keras.layers.Conv2D(filters, 3, padding='same', dilation_rate=d, use_bias=False)(x))

    pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    pool = tf.keras.layers.Reshape((1, 1, x.shape[-1]))(pool)
    pool = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(pool)
    pool = tf.keras.layers.UpSampling2D(size=(h, w), interpolation='bilinear')(pool)
    layers.append(pool)

    x = tf.keras.layers.Concatenate()(layers)
    x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def _build_deeplabv3(num_classes, input_size, backbone_name, pretrained_backbone=True):
    weights = 'imagenet' if pretrained_backbone else None
    backbone_fn = _resolve_backbone(backbone_name)
    if backbone_fn is None:
        raise ValueError(f"Unsupported backbone: {backbone_name}. Available: {list(_TF_BACKBONES.keys())}")

    inputs = tf.keras.layers.Input(shape=(input_size, input_size, 3))
    backbone = backbone_fn(include_top=False, weights=weights, input_tensor=inputs)
    x = backbone.output

    x = _build_aspp(x)
    x = tf.keras.layers.Conv2D(num_classes, 1, padding='same')(x)
    x = tf.keras.layers.UpSampling2D(size=(input_size // x.shape[1], input_size // x.shape[2]), interpolation='bilinear')(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=f"deeplabv3_{backbone_name}")


def _build_unet(num_classes, input_size):
    """Simple U-Net (no backbone, light weight)."""
    inputs = tf.keras.layers.Input(shape=(input_size, input_size, 3))

    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        return x

    c1 = conv_block(inputs, 64)
    p1 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 128)
    p2 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 256)
    p3 = tf.keras.layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 512)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    b = conv_block(p4, 1024)

    u4 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, padding='same')(b)
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    u4 = conv_block(u4, 512)
    u3 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same')(u4)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    u3 = conv_block(u3, 256)
    u2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(u3)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    u2 = conv_block(u2, 128)
    u1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(u2)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    u1 = conv_block(u1, 64)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same')(u1)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="unet")


class Tensorflow_Segmentation_Models:
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

    def init_model(self, num_classes, network_name=None, input_size=512, pretrained_backbone=True, output_stride=16, **kwargs):
        if network_name is None:
            raise ValueError("network_name is required")

        with tf.device(self.__device):
            if network_name == "unet":
                self.__model = _build_unet(num_classes, input_size)
            elif network_name.startswith("deeplabv3_"):
                backbone = network_name[len("deeplabv3_"):]
                self.__model = _build_deeplabv3(num_classes, input_size, backbone, pretrained_backbone=pretrained_backbone)
            else:
                available = ["unet"] + [f"deeplabv3_{b}" for b in _TF_BACKBONES]
                raise ValueError(f"Unsupported network: '{network_name}'. Available: {available}")
        return self

    def get_model(self):
        return self.__model
