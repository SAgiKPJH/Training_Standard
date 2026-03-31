import random
import cv2
from .normal import generate_normal


def generate_dot():
    """점 결함: 패드 사각형 안에 검은색 점"""
    img, (x1, y1, x2, y2) = generate_normal(return_bounds=True)

    for _ in range(random.randint(1, 3)):
        px = random.randint(x1, x2)
        py = random.randint(y1, y2)
        dot_r = random.randint(2, 8)
        dot_color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
        cv2.circle(img, (px, py), dot_r, dot_color, -1)

    return img
