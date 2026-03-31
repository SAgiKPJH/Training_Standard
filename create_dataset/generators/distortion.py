import random
import numpy as np
import cv2
from .base import IMAGE_SIZE, draw_pcb_background, draw_rounded_rect, random_pad_params, pad_fill_color, pad_border_color


def _make_clip_mask(cx, cy, w, h):
    """패드를 깍는 랜덤 도형 마스크 생성 (흰색=깍이는 영역)"""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    shape_type = random.choice(["rect", "polygon"])

    edge = random.choice(["top", "bottom", "left", "right", "tl", "tr", "bl", "br"])
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = cx + w // 2
    y2 = cy + h // 2

    offsets = {
        "top": (cx + random.randint(-w // 4, w // 4), y1 + random.randint(-10, h // 4)),
        "bottom": (cx + random.randint(-w // 4, w // 4), y2 - random.randint(-10, h // 4)),
        "left": (x1 + random.randint(-10, w // 4), cy + random.randint(-h // 4, h // 4)),
        "right": (x2 - random.randint(-10, w // 4), cy + random.randint(-h // 4, h // 4)),
        "tl": (x1 + random.randint(0, w // 4), y1 + random.randint(0, h // 4)),
        "tr": (x2 - random.randint(0, w // 4), y1 + random.randint(0, h // 4)),
        "bl": (x1 + random.randint(0, w // 4), y2 - random.randint(0, h // 4)),
        "br": (x2 - random.randint(0, w // 4), y2 - random.randint(0, h // 4)),
    }
    ox, oy = offsets[edge]

    if shape_type == "rect":
        rw = random.randint(w // 3, w)
        rh = random.randint(h // 3, h)
        angle = random.randint(-30, 30)
        box = cv2.boxPoints(((ox, oy), (rw, rh), angle))
        box = np.int32(box)
        cv2.fillPoly(mask, [box], 255)

    else:  # polygon
        n_pts = random.randint(3, 6)
        pts = []
        for i in range(n_pts):
            a = 2 * np.pi * i / n_pts + random.uniform(-0.4, 0.4)
            dist = random.randint(w // 4, w // 2 + 10)
            px = ox + int(dist * np.cos(a))
            py = oy + int(dist * np.sin(a))
            pts.append([px, py])
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)

    return mask


def generate_distortion():
    """일그러짐 결함: 마스크 빼기 방식 (사각형 - 랜덤도형 = 깍인 패드)"""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    draw_pcb_background(img)
    bg_b, bg_g, bg_r = int(img[0, 0, 0]), int(img[0, 0, 1]), int(img[0, 0, 2])

    cx, cy, w, h, radius = random_pad_params()
    fill = pad_fill_color(bg_b, bg_g, bg_r)
    border = pad_border_color()

    pad_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    draw_rounded_rect(pad_mask, cx, cy, w, h, radius, 255, thickness=-1)

    clip_mask = _make_clip_mask(cx, cy, w, h)

    result_mask = cv2.subtract(pad_mask, clip_mask)

    contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(img, [largest], -1, fill, thickness=-1, lineType=cv2.LINE_AA)
        cv2.drawContours(img, [largest], -1, border, thickness=2, lineType=cv2.LINE_AA)

    return img
