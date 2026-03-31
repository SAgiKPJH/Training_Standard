from .normal import generate_normal
from .dot import generate_dot
from .line import generate_line
from .distortion import generate_distortion

GENERATORS = {
    "0_normal": generate_normal,
    "1_dot": generate_dot,
    "2_line": generate_line,
    "3_distortion": generate_distortion,
}

__all__ = ['GENERATORS', 'generate_normal', 'generate_dot', 'generate_line', 'generate_distortion']
