import random
import numpy as np
import cv2

IMAGE_SIZE = 299


def draw_pcb_background(img):
    """PCB 기판 배경: 단색 랜덤 (노란색 계열)"""
    base_b = random.randint(80, 120)
    base_g = random.randint(190, 220)
    base_r = random.randint(200, 235)
    img[:] = (base_b, base_g, base_r)


def draw_rounded_rect(img, center_x, center_y, w, h, radius, color, thickness=-1):
    """꼭짓점이 둥근 사각형 그리기"""
    x1 = center_x - w // 2
    y1 = center_y - h // 2
    x2 = center_x + w // 2
    y2 = center_y + h // 2
    r = min(radius, w // 2, h // 2)

    pts_outer = []
    corners = [
        (x1 + r, y1 + r, 180, 270),
        (x2 - r, y1 + r, 270, 360),
        (x2 - r, y2 - r, 0, 90),
        (x1 + r, y2 - r, 90, 180),
    ]

    for cx, cy, start_angle, end_angle in corners:
        for angle in range(start_angle, end_angle + 1, 5):
            rad = np.radians(angle)
            px = int(cx + r * np.cos(rad))
            py = int(cy + r * np.sin(rad))
            pts_outer.append([px, py])

    pts = np.array(pts_outer, dtype=np.int32)

    if thickness == -1:
        cv2.fillPoly(img, [pts], color)
    else:
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    return x1, y1, x2, y2


def random_pad_params():
    """패드 위치/크기 랜덤 생성 (거의 동일 크기, 거의 정사각형)"""
    size = random.randint(110, 130)
    w = size + random.randint(-5, 5)
    h = size + random.randint(-5, 5)
    cx = IMAGE_SIZE // 2 + random.randint(-20, 20)
    cy = IMAGE_SIZE // 2 + random.randint(-20, 20)
    radius = random.randint(12, 22)
    return cx, cy, w, h, radius


def pad_fill_color(bg_b, bg_g, bg_r):
    """패드 내부: 배경보다 살짝 밝고 노란색 계열 (BGR)"""
    r = min(255, bg_r + random.randint(30, 60))
    g = min(255, bg_g + random.randint(25, 50))
    b = max(0, bg_b - random.randint(10, 30))
    return (b, g, r)


def pad_border_color():
    """패드 테두리: 거의 검은색 (BGR)"""
    v = random.randint(10, 35)
    return (v, v, v)
