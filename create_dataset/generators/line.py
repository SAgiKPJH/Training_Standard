import random
import numpy as np
import cv2
from .normal import generate_normal


def generate_line():
    """선 결함: 패드 사각형 안에 리얼한 스크래치"""
    img, (bx1, by1, bx2, by2) = generate_normal(return_bounds=True)

    for _ in range(random.randint(1, 2)):
        angle = random.uniform(0, np.pi)
        cx = (bx1 + bx2) // 2 + random.randint(-20, 20)
        cy = (by1 + by2) // 2 + random.randint(-20, 20)
        half_len = random.randint(25, 60)

        num_segments = random.randint(8, 16)
        pts = []
        for i in range(num_segments + 1):
            t = (i / num_segments) - 0.5
            px = cx + int(half_len * 2 * t * np.cos(angle)) + random.randint(-2, 2)
            py = cy + int(half_len * 2 * t * np.sin(angle)) + random.randint(-2, 2)
            px = max(bx1, min(bx2, px))
            py = max(by1, min(by2, py))
            pts.append((px, py))

        base_v = random.randint(30, 80)
        scratch_color = (base_v, base_v, base_v)
        thickness = random.choice([1, 1, 1, 2])
        for i in range(len(pts) - 1):
            seg_thick = thickness if random.random() > 0.2 else max(1, thickness - 1)
            cv2.line(img, pts[i], pts[i + 1], scratch_color, seg_thick, cv2.LINE_AA)

        if random.random() > 0.3:
            highlight_color = (
                min(255, scratch_color[0] + random.randint(80, 140)),
                min(255, scratch_color[1] + random.randint(80, 140)),
                min(255, scratch_color[2] + random.randint(60, 100)),
            )
            offset_x = int(np.sin(angle) * 1.5)
            offset_y = int(-np.cos(angle) * 1.5)
            for i in range(len(pts) - 1):
                p1 = (pts[i][0] + offset_x, pts[i][1] + offset_y)
                p2 = (pts[i + 1][0] + offset_x, pts[i + 1][1] + offset_y)
                cv2.line(img, p1, p2, highlight_color, 1, cv2.LINE_AA)

    return img
