import numpy as np
from abc import abstractmethod


class OcrInference:
    def infer(self, image: np.ndarray) -> list:
        """Returns [{'text': str, 'score': float, 'box': [[x1,y1],...,[x4,y4]]}, ...]"""
        return self._run(image)

    @abstractmethod
    def _run(self, image: np.ndarray) -> list:
        raise NotImplementedError
