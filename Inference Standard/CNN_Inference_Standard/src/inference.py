import json
import numpy as np


def parse_inference_info(raw: dict) -> tuple:
    """
    Training Standard 저장 구조에서 (input_size, label_info) 를 추출한다.

    저장 구조:
        {
          "label_info":     '{"inference_info": "{\"input_size\":224,\"label_info\":{...}}"}',
          "inference_info": '{}'
        }
    """
    val = raw.get('label_info', '{}')
    if isinstance(val, bytes):
        val = val.decode('utf-8')
    if isinstance(val, str):
        val = json.loads(val)

    inner = val.get('inference_info', '{}')
    if isinstance(inner, str):
        inner = json.loads(inner)

    return inner.get('input_size', 224), inner.get('label_info', {})


class Inference:
    """
    CNN 분류 추론 공통 인터페이스.

    서브클래스는 _run(image) 만 구현하면 된다.
    출력 빌드·정렬 등 공통 후처리는 infer() 가 담당한다.
    """

    def infer(self, image: np.ndarray) -> list:
        """
        BGR ndarray → [{'code': ..., 'name': ..., 'score': ...}, ...] (score 내림차순)
        """
        scores, label_info = self._run(image)

        output = []
        for i in range(label_info['label_count']):
            label = dict(label_info[f'label_{i}'])
            label['score'] = scores[i]
            output.append(label)

        output.sort(key=lambda x: x['score'], reverse=True)
        return output

    def _run(self, image: np.ndarray) -> tuple:
        """
        (scores: list[float], label_info: dict) 반환.
        서브클래스에서 반드시 구현해야 한다.
        """
        raise NotImplementedError
