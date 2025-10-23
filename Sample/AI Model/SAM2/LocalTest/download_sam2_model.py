import os
import requests
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def download_file(url: str, filepath: str, chunk_size: int = 8192):
    """파일 다운로드 함수"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"다운로드 완료: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"다운로드 실패: {e}")
        return False

def main():
    """SAM2 모델 다운로드"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # SAM2.1 모델 URL (Meta AI에서 제공하는 공식 모델 - 더 나은 성능)
    # https://github.com/facebookresearch/sam2 에서 제공하는 공식 URL
    sam2_urls = {
        "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    }
    
    logger.info("SAM2 모델 다운로드를 시작합니다...")
    
    for model_name, url in sam2_urls.items():
        filepath = os.path.join(base_path, model_name)
        
        if os.path.exists(filepath):
            logger.info(f"모델이 이미 존재합니다: {model_name}")
            continue
        
        logger.info(f"다운로드 중: {model_name}")
        success = download_file(url, filepath)
        
        if success:
            logger.info(f"성공: {model_name}")
        else:
            logger.error(f"실패: {model_name}")
    
    logger.info("다운로드 완료!")

if __name__ == "__main__":
    main()




