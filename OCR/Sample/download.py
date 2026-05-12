"""
인터넷이 되는 PC에서 실행해 PaddleOCR 한국어 사전학습 모델을 다운로드합니다.
다운로드 완료 후 data/ 폴더를 내부망 PC의 같은 위치에 복사하세요.

다운로드 대상:
  - Multilingual_PP-OCRv3_det_infer  (det 모델,  ~4 MB) -> data/det_model_Multilingual_PP-OCRv3/
  - ch_ppocr_mobile_v2.0_cls_infer   (cls 모델,  ~2 MB) -> data/cls_model_ch_ppocr_mobile_v2.0/
  - korean_PP-OCRv4_rec_infer        (rec 모델, ~24 MB) -> data/rec_model_korean_PP-OCRv4/
  - korean_dict.txt (한국어 문자 사전)                  -> data/korean_dict.txt
"""

import os
import tarfile
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    {
        'name': 'det (Multilingual_PP-OCRv3_det_infer)',
        'url':  'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar',
        'tar':  'Multilingual_PP-OCRv3_det_infer.tar',
        'dest': 'data/det_model_Multilingual_PP-OCRv3',
    },
    {
        'name': 'cls (ch_ppocr_mobile_v2.0_cls_infer)',
        'url':  'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
        'tar':  'ch_ppocr_mobile_v2.0_cls_infer.tar',
        'dest': 'data/cls_model_ch_ppocr_mobile_v2.0',
    },
    {
        'name': 'rec (korean_PP-OCRv4_rec_infer)',
        'url':  'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar',
        'tar':  'korean_PP-OCRv4_rec_infer.tar',
        'dest': 'data/rec_model_korean_PP-OCRv4',
    },
]


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    pct = downloaded * 100 // total_size
    bar = '#' * (pct // 5) + '-' * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%  {downloaded / 1_048_576:.1f} MB", end='', flush=True)


def _download(url: str, save_path: str):
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, save_path, reporthook=_progress)
    print()


def _extract(tar_path: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        top = members[0].name.split('/')[0] if members else ''
        for m in members:
            if m.isdir():
                continue
            m.name = os.path.relpath(m.name, top)
            tar.extract(m, dest_dir)
    print(f"  Extracted to: {dest_dir}")


def _copy_korean_dict():
    dest = os.path.join(BASE_DIR, 'data', 'korean_dict.txt')
    if os.path.exists(dest):
        print("[skip] Already exists: data/korean_dict.txt")
        return
    try:
        import paddleocr
        pkg_dict = os.path.join(os.path.dirname(paddleocr.__file__),
                                'ppocr', 'utils', 'dict', 'korean_dict.txt')
        if not os.path.exists(pkg_dict):
            print("[warn] paddleocr 패키지에서 korean_dict.txt 를 찾지 못했습니다.")
            return
        import shutil
        shutil.copy2(pkg_dict, dest)
        print(f"  Copied: {pkg_dict} -> {dest}")
    except ImportError:
        print("[warn] paddleocr 미설치 상태 — korean_dict.txt 복사를 건너뜁니다.")


def main():
    for info in MODELS:
        dest_dir = os.path.join(BASE_DIR, info['dest'])
        if os.path.isdir(dest_dir) and os.listdir(dest_dir):
            print(f"[skip] Already exists: {dest_dir}")
            continue

        tar_path = os.path.join(BASE_DIR, info['tar'])
        print(f"\n[{info['name']}]")
        try:
            _download(info['url'], tar_path)
            _extract(tar_path, dest_dir)
        finally:
            if os.path.exists(tar_path):
                os.remove(tar_path)
        print(f"  Done: {dest_dir}")

    print("\n[korean_dict.txt]")
    _copy_korean_dict()

    print("\n--- 완료 ---")
    print("data/ 폴더를 내부망 PC의 같은 위치에 복사하세요.")


if __name__ == "__main__":
    main()
