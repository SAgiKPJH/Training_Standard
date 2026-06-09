"""
EVA 학습 테스트
===============
timm eva_large_patch14_196 을 로컬 데이터셋으로 간단 파인튜닝하고
추론에 필요한 정보를 포함한 체크포인트(model.pth)를 저장한다.

실행:
    python train_eva.py
"""
import os
import time
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import eva_common as ec

# ── 설정 ─────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", "create_dataset", "dataset"))
OUTPUT_PATH = os.path.join(_HERE, "output", "model.pth")

EPOCH = 1
BATCH_SIZE = 2
LR = 1e-4
TRAIN_RATIO = 0.8
USING_GPU = torch.cuda.is_available()
MAX_ITER = 10  # 간단 테스트용 epoch 당 최대 iteration (None 이면 전체)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def split(samples, train_ratio):
    samples = samples[:]
    random.shuffle(samples)
    n_val = int(len(samples) * (1 - train_ratio))
    return samples[n_val:], samples[:n_val]


def main():
    device = "cuda" if USING_GPU else "cpu"
    logger.info(f"Device: {device}")

    classes = ec.list_classes(DATASET_PATH)
    samples = ec.build_samples(DATASET_PATH, classes)
    train_samples, val_samples = split(samples, TRAIN_RATIO)
    logger.info(f"Classes: {classes} / train: {len(train_samples)}, val: {len(val_samples)}")

    train_loader = DataLoader(ec.BGRClassificationDataset(train_samples),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(ec.BGRClassificationDataset(val_samples),
                            batch_size=1, shuffle=False, num_workers=0) if val_samples else None

    logger.info(f"Loading pretrained model: {ec.MODEL_NAME} ...")
    model = ec.build_model(len(classes), pretrained=True, device=device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_loss = None
    for epoch in range(1, EPOCH + 1):
        model.train()
        epoch_start = time.time()
        running, n_iter = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_iter += 1
            logger.info(f"Epoch {epoch}/{EPOCH}  iter {n_iter}  loss {loss.item():.4f}")
            if MAX_ITER and n_iter >= MAX_ITER:
                break
        train_loss = running / max(1, n_iter)

        val_loss = None
        if val_loader:
            model.eval()
            v_running, v_n = 0.0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    v_running += criterion(model(inputs), labels).item()
                    v_n += 1
                    if MAX_ITER and v_n >= MAX_ITER:
                        break
            val_loss = v_running / max(1, v_n)

        logger.info(f"[Epoch {epoch}] train_loss={train_loss:.4f} "
                    f"val_loss={val_loss if val_loss is None else round(val_loss,4)} "
                    f"time={time.time()-epoch_start:.1f}s")

        cur = val_loss if val_loss is not None else train_loss
        if best_loss is None or cur < best_loss:
            best_loss = cur
            ec.save_checkpoint(OUTPUT_PATH, model, classes)
            logger.info(f"  ★ Saved checkpoint -> {OUTPUT_PATH} (loss {cur:.4f})")

    logger.info("Training done.")


if __name__ == "__main__":
    main()
