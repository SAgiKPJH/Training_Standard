"""
PCB 패드 결함 분류용 샘플 데이터셋 생성기.
4개 클래스의 299x299 이미지를 concurrent하게 생성합니다:
  - 0_normal      : 정상 패드 (꼭짓점이 둥근 사각형)
  - 1_dot         : 점 결함 (패드 위에 작은 점)
  - 2_line        : 선 결함 (패드 위에 선/스크래치)
  - 3_distortion  : 일그러짐 결함 (패드 형태가 비틀어짐)

Usage:
    python create_sample_dataset.py --output D:\\test\\Dataset --count 50
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2

from generators import GENERATORS


def _generate_single(args):
    """단일 이미지 생성 (워커 프로세스에서 실행)"""
    class_name, index, output_dir, generator_name = args

    # 워커에서 generator 함수를 다시 import (pickle 호환)
    from generators import GENERATORS as gens
    generator = gens[class_name]

    img = generator()
    filename = f"{class_name}_{index:04d}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, img)
    return class_name, filename


def create_dataset(output_path, num_images, workers):
    print(f"Creating dataset at: {output_path}")
    print(f"Images per class: {num_images}")
    print(f"Classes: {list(GENERATORS.keys())}")
    print(f"Workers: {workers}")
    print()

    # 디렉토리 생성
    for class_name in GENERATORS:
        os.makedirs(os.path.join(output_path, class_name), exist_ok=True)

    # 작업 목록 생성
    tasks = []
    for class_name in GENERATORS:
        class_dir = os.path.join(output_path, class_name)
        for i in range(num_images):
            tasks.append((class_name, i, class_dir, class_name))

    # concurrent 실행
    completed = {name: 0 for name in GENERATORS}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_generate_single, task): task for task in tasks}

        for future in as_completed(futures):
            class_name, filename = future.result()
            completed[class_name] += 1

    for class_name, count in completed.items():
        print(f"  [{class_name}] {count} images created")

    total = sum(completed.values())
    print(f"\nDone. Total {total} images in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB 패드 결함 분류용 샘플 데이터셋 생성")
    parser.add_argument("--output", "-o", type=str, default=r"D:\test\Dataset", help="데이터셋 저장 경로")
    parser.add_argument("--count", "-n", type=int, default=50, help="클래스당 이미지 개수")
    parser.add_argument("--workers", "-w", type=int, default=4, help="동시 실행 프로세스 수")
    args = parser.parse_args()

    create_dataset(args.output, args.count, args.workers)
