"""
OCR 인식(rec) 학습/추론용 샘플 데이터 생성기.

학습 데이터: OCR/Training/data/
  images/train/*.png  +  train_labels.txt
  images/val/*.png    +  val_labels.txt
추론 테스트: OCR/Inference/data/test_image.png

라벨 형식 (PaddleOCR SimpleDataSet):
  images/train/00000.png\tHello123

Usage:
    python create_sample_data.py [--train N] [--val N]
"""

import argparse
import os
import random
import string

from PIL import Image, ImageDraw, ImageFont

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "Training", "data", "images", "train")
VAL_IMG_DIR   = os.path.join(BASE_DIR, "Training", "data", "images", "val")
DATA_DIR      = os.path.join(BASE_DIR, "Training", "data")
INF_DIR       = os.path.join(BASE_DIR, "Inference", "data")

CHARS  = string.digits + string.ascii_letters + "!@#$%&*()-_=+[]{}|;:,.<>?"
HEIGHT = 48


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _get_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _make_image(text: str, height: int = HEIGHT) -> Image.Image:
    font = _get_font(int(height * 0.72))

    tmp  = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = tmp.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad_x = random.randint(6, 14)
    width = max(tw + pad_x * 2, height * 2)

    bg   = tuple(random.randint(220, 255) for _ in range(3))
    fg   = tuple(random.randint(0, 40) for _ in range(3))
    img  = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x, (height - th) // 2), text, fill=fg, font=font)
    return img


def _random_text(min_len: int = 3, max_len: int = 12) -> str:
    return "".join(random.choices(CHARS, k=random.randint(min_len, max_len)))


# ── 생성 함수 ─────────────────────────────────────────────────────────────────

def generate_char_dict(dict_path: str):
    """PaddleOCR 형식 문자 사전 생성 (한 줄에 한 문자)."""
    os.makedirs(os.path.dirname(dict_path), exist_ok=True)
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(CHARS))
    print(f"  문자 사전 ({len(CHARS)}자) → {dict_path}")


def generate_train_set(img_dir: str, label_path: str, count: int):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    lines = []
    for i in range(count):
        text  = _random_text()
        fname = f"{i:05d}.png"
        fpath = os.path.join(img_dir, fname)
        _make_image(text).save(fpath)
        rel = os.path.relpath(fpath, DATA_DIR).replace("\\", "/")
        lines.append(f"{rel}\t{text}")

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  {count}개 이미지 → {img_dir}")
    print(f"  라벨            → {label_path}")


def generate_inference_sample():
    os.makedirs(INF_DIR, exist_ok=True)
    samples = ["Hello123", "ABC-007", "2024/05/08", "Test_99%", "ID:X0042"]
    for text in samples:
        fname = f"test_{text.replace('/', '_').replace(':', '_')}.png"
        _make_image(text, height=64).save(os.path.join(INF_DIR, fname))
    _make_image("Hello123", height=64).save(os.path.join(INF_DIR, "test_image.png"))
    print(f"  추론 이미지 {len(samples) + 1}개 → {INF_DIR}")


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=200, help="학습 이미지 수")
    parser.add_argument("--val",   type=int, default=50,  help="검증 이미지 수")
    args = parser.parse_args()

    print("[Training] 문자 사전 생성...")
    generate_char_dict(os.path.join(DATA_DIR, "en_dict.txt"))

    print(f"[Training] train set ({args.train}개) 생성...")
    generate_train_set(TRAIN_IMG_DIR, os.path.join(DATA_DIR, "train_labels.txt"), args.train)

    print(f"[Training] val set ({args.val}개) 생성...")
    generate_train_set(VAL_IMG_DIR, os.path.join(DATA_DIR, "val_labels.txt"), args.val)

    print("[Inference] 테스트 이미지 생성...")
    generate_inference_sample()

    print("완료.")
