import tensorflow as tf


def build_augmentation_fn(augmentation: dict):
    """augmentation dict에서 tf.data.Dataset용 augmentation 함수를 생성합니다.

    Args:
        augmentation: augmentation 옵션 dict

    Returns:
        augment_fn: (image, label) -> (image, label) 함수. augmentation이 없으면 None.
    """
    if not augmentation:
        return None

    ops = []

    # Flip
    if augmentation.get('horizontal_flip', False):
        ops.append(lambda img: tf.image.random_flip_left_right(img))
    if augmentation.get('vertical_flip', False):
        ops.append(lambda img: tf.image.random_flip_up_down(img))

    # Brightness
    brightness = augmentation.get('brightness', 0)
    if brightness > 0:
        ops.append(lambda img, b=brightness: tf.image.random_brightness(img, max_delta=b))

    # Contrast
    contrast = augmentation.get('contrast', 0)
    if contrast > 0:
        ops.append(lambda img, c=contrast: tf.image.random_contrast(img, lower=1-c, upper=1+c))

    # Saturation
    saturation = augmentation.get('saturation', 0)
    if saturation > 0:
        ops.append(lambda img, s=saturation: tf.image.random_saturation(img, lower=1-s, upper=1+s))

    # Hue
    hue = augmentation.get('hue', 0)
    if hue > 0:
        ops.append(lambda img, h=hue: tf.image.random_hue(img, max_delta=h))

    # Rotation
    rotation = augmentation.get('rotation', 0)
    if rotation > 0:
        import math
        max_rad = rotation * math.pi / 180.0
        ops.append(lambda img, r=max_rad: tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32)) if r >= 90
                   else _random_rotate(img, r))

    # Random Grayscale
    random_grayscale = augmentation.get('random_grayscale', 0)
    if random_grayscale > 0:
        ops.append(lambda img, p=random_grayscale: _random_grayscale(img, p))

    # Gaussian Blur (approximate with gaussian filter)
    gaussian_blur = augmentation.get('gaussian_blur', 0)
    if gaussian_blur > 0:
        ops.append(lambda img, s=gaussian_blur: _gaussian_blur(img, s))

    if not ops:
        return None

    def augment_fn(image, label):
        for op in ops:
            image = op(image)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    return augment_fn


def _random_rotate(image, max_radians):
    """랜덤 회전 (임의 각도)"""
    angle = tf.random.uniform([], -max_radians, max_radians)
    return _rotate_image(image, angle)


def _rotate_image(image, angle):
    """이미지 회전"""
    # tf.keras.preprocessing 또는 tfa 없이 구현
    cos_angle = tf.math.cos(angle)
    sin_angle = tf.math.sin(angle)
    # affine transform matrix [cos, -sin, 0, sin, cos, 0, 0, 0]
    transform = [cos_angle, -sin_angle, 0.0, sin_angle, cos_angle, 0.0, 0.0, 0.0]
    image = tf.expand_dims(image, 0)
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=[transform],
        output_shape=tf.shape(image)[1:3],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0
    )
    return tf.squeeze(image, 0)


def _random_grayscale(image, probability):
    """랜덤 그레이스케일 변환"""
    if tf.random.uniform([]) < probability:
        gray = tf.image.rgb_to_grayscale(image)
        return tf.image.grayscale_to_rgb(gray)
    return image


def _gaussian_blur(image, sigma):
    """가우시안 블러 (근사)"""
    kernel_size = 3
    # 가우시안 커널 생성
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    g = tf.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / tf.reduce_sum(g)
    g_2d = tf.tensordot(g, g, axes=0)
    g_2d = tf.reshape(g_2d, [kernel_size, kernel_size, 1, 1])
    g_2d = tf.tile(g_2d, [1, 1, 3, 1])  # 3채널

    image = tf.expand_dims(image, 0)
    image = tf.nn.depthwise_conv2d(image, g_2d, strides=[1, 1, 1, 1], padding='SAME')
    return tf.squeeze(image, 0)
