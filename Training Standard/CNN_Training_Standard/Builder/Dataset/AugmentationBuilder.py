import torchvision.transforms as transforms


def build_augmentation_transforms(augmentation: dict) -> list:
    """augmentation dict에서 PyTorch transforms 리스트를 생성합니다.

    Args:
        augmentation: augmentation 옵션 dict

    Returns:
        list of torchvision transforms (ToTensor 전에 적용할 것들)
    """
    if not augmentation:
        return [], []

    pre_tensor = []   # ToTensor 전 (PIL Image 대상)
    post_tensor = []  # ToTensor 후 (Tensor 대상)

    # Flip
    if augmentation.get('horizontal_flip', False):
        pre_tensor.append(transforms.RandomHorizontalFlip(p=0.5))
    if augmentation.get('vertical_flip', False):
        pre_tensor.append(transforms.RandomVerticalFlip(p=0.5))

    # Rotation
    if augmentation.get('rotation', 0) > 0:
        pre_tensor.append(transforms.RandomRotation(degrees=augmentation['rotation']))

    # Color Jitter
    brightness = augmentation.get('brightness', 0)
    contrast = augmentation.get('contrast', 0)
    saturation = augmentation.get('saturation', 0)
    hue = augmentation.get('hue', 0)
    if any([brightness, contrast, saturation, hue]):
        pre_tensor.append(transforms.ColorJitter(
            brightness=brightness or 0,
            contrast=contrast or 0,
            saturation=saturation or 0,
            hue=hue or 0
        ))

    # Gaussian Blur
    if augmentation.get('gaussian_blur', 0) > 0:
        pre_tensor.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, augmentation['gaussian_blur'])))

    # Affine
    if augmentation.get('random_affine', 0) > 0:
        pre_tensor.append(transforms.RandomAffine(degrees=augmentation['random_affine']))

    # Perspective
    if augmentation.get('random_perspective', 0) > 0:
        pre_tensor.append(transforms.RandomPerspective(distortion_scale=augmentation['random_perspective'], p=0.5))

    # Grayscale
    if augmentation.get('random_grayscale', 0) > 0:
        pre_tensor.append(transforms.RandomGrayscale(p=augmentation['random_grayscale']))

    # Random Erasing (post-tensor)
    if augmentation.get('random_erasing', 0) > 0:
        post_tensor.append(transforms.RandomErasing(p=augmentation['random_erasing']))

    return pre_tensor, post_tensor
