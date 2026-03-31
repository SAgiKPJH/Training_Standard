import numpy as np
import cv2
from .base import IMAGE_SIZE, draw_pcb_background, draw_rounded_rect, random_pad_params, pad_fill_color, pad_border_color


def generate_normal(return_bounds=False):
    """정상 패드 이미지"""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    draw_pcb_background(img)
    bg_b, bg_g, bg_r = int(img[0, 0, 0]), int(img[0, 0, 1]), int(img[0, 0, 2])

    cx, cy, w, h, radius = random_pad_params()
    fill = pad_fill_color(bg_b, bg_g, bg_r)
    border = pad_border_color()

    draw_rounded_rect(img, cx, cy, w, h, radius, fill, thickness=-1)
    draw_rounded_rect(img, cx, cy, w, h, radius, border, thickness=2)

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    draw_rounded_rect(mask, cx, cy, w, h, radius, 255, thickness=-1)
    highlight = np.zeros_like(img)
    cv2.circle(highlight, (cx - w // 6, cy - h // 6), w // 3, (20, 25, 30), -1)
    highlight = cv2.GaussianBlur(highlight, (51, 51), 0)
    img = np.clip(img.astype(np.int16) + (highlight * (mask[:, :, None] / 255.0)).astype(np.int16), 0, 255).astype(np.uint8)

    if return_bounds:
        margin = radius // 2 + 4
        x1 = cx - w // 2 + margin
        y1 = cy - h // 2 + margin
        x2 = cx + w // 2 - margin
        y2 = cy + h // 2 - margin
        return img, (x1, y1, x2, y2)
    return img
