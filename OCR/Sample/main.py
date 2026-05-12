import os
from paddleocr import PaddleOCR

BASE = os.path.dirname(os.path.abspath(__file__))

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='korean',
    use_gpu=True,
    det_model_dir=os.path.join(BASE, 'data/det_model_Multilingual_PP-OCRv3'),
    cls_model_dir=os.path.join(BASE, 'data/cls_model_ch_ppocr_mobile_v2.0'),
    rec_model_dir=os.path.join(BASE, 'data/rec_model_korean_PP-OCRv4'),
    rec_char_dict_path=os.path.join(BASE, 'data/korean_dict.txt'),
)

img_path = os.path.join(BASE, 'image.png')
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    print(result[idx])
