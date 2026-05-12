parameters = '''{
    "version": "OCR_Training_Standard_v1.0.0",
    "hyperparameter": {
        "pretrained_model":    "data/pretrained/en_PP-OCRv3_rec_train/best_accuracy",
        "resume_path":         "",
        "rec_char_dict_path":  "data/en_dict.txt",
        "save_dir":            "output/",
        "train_data_dir":      "data/",
        "train_label_file":    "data/train_labels.txt",
        "val_data_dir":        "data/",
        "val_label_file":      "data/val_labels.txt",
        "image_shape":         [3, 48, 320],
        "max_text_length":     50,
        "epoch_num":           100,
        "save_epoch_step":     10,
        "batch_size":          64,
        "learning_rate":       0.0005,
        "use_space_char":      false,
        "use_guided_training": false,
        "device":              "gpu",
        "gpu_id":              "0",
        "paddleocr_home":      ""
    }
}'''

import json, logging, os, shutil, subprocess, sys
import yaml

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(message)s')


def _abs(base, p):
    return os.path.join(base, p) if p and not os.path.isabs(p) else (p or None)


def _validate(hp, base):
    for key in ('train_data_dir', 'train_label_file', 'val_data_dir', 'val_label_file', 'rec_char_dict_path'):
        path = _abs(base, hp[key])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{key} not found: {path}")
    tc = sum(1 for _ in open(_abs(base, hp['train_label_file']), encoding='utf-8'))
    vc = sum(1 for _ in open(_abs(base, hp['val_label_file']),   encoding='utf-8'))
    cc = sum(1 for _ in open(_abs(base, hp['rec_char_dict_path']), encoding='utf-8'))
    logger.info(f"Dataset  train={tc}  val={vc}  chars={cc}")


def _build_yaml(hp, base):
    epoch     = int(hp['epoch_num'])
    save_step = int(hp.get('save_epoch_step', 10)) or epoch
    guided    = bool(hp.get('use_guided_training', False))
    shape     = hp.get('image_shape', [3, 48, 320])
    max_len   = int(hp['max_text_length'])
    batch     = int(hp['batch_size'])

    label_enc = ({'MultiLabelEncode': {}} if guided
                 else {'CTCLabelEncode': {'max_text_length': max_len}})
    keys      = (['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio'] if guided
                 else ['image', 'label', 'length'])

    def transforms(is_train):
        t = [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
        if is_train:
            t.append({'RecAug': {}})
        return t + [label_enc, {'RecResizeImg': {'image_shape': shape}},
                    {'KeepKeys': {'keep_keys': keys}}]

    def dataset(data_dir, label_file, is_train):
        return {'name': 'SimpleDataSet',
                'data_dir': _abs(base, data_dir),
                'label_file_list': [_abs(base, label_file)],
                'transforms': transforms(is_train)}

    return {
        'Global': {
            'use_gpu':               hp.get('device', 'gpu').lower() == 'gpu',
            'epoch_num':             epoch,
            'log_smooth_window':     20,
            'print_batch_step':      10,
            'save_model_dir':        _abs(base, hp.get('save_dir', 'output/')),
            'save_epoch_step':       save_step,
            'eval_batch_step':       [0, 2000],
            'cal_metric_during_train': True,
            'pretrained_model':      _abs(base, hp.get('pretrained_model', '')),
            'checkpoints':           _abs(base, hp.get('resume_path', '')) or None,
            'save_inference_dir':    None,
            'use_visualdl':          False,
            'character_dict_path':   _abs(base, hp['rec_char_dict_path']),
            'max_text_length':       max_len,
            'infer_mode':            False,
            'use_space_char':        bool(hp.get('use_space_char', False)),
            'distributed':           False,
        },
        'Optimizer': {
            'name': 'Adam', 'beta1': 0.9, 'beta2': 0.999,
            'lr': {'name': 'Cosine', 'learning_rate': float(hp['learning_rate']), 'warmup_epoch': 5},
            'regularizer': {'name': 'L2', 'factor': 3.0e-5},
        },
        'Architecture': {
            'model_type': 'rec', 'algorithm': 'SVTR_LCNet', 'Transform': None,
            'Backbone': {
                'name': 'MobileNetV1Enhance', 'scale': 0.5,
                'last_conv_stride': [1, 2], 'last_pool_type': 'avg', 'last_pool_kernel_size': [2, 4],
            },
            'Neck': {
                'name': 'SequenceEncoder', 'encoder_type': 'svtr',
                'dims': 64, 'depth': 2, 'hidden_dims': 120, 'use_guide': guided,
            },
            'Head': {'name': 'CTCHead', 'fc_decay': 1.0e-5},
        },
        'Loss':        {'name': 'MultiLoss' if guided else 'CTCLoss'},
        'PostProcess': {'name': 'CTCLabelDecode'},
        'Metric':      {'name': 'RecMetric', 'main_indicator': 'acc', 'ignore_space': False},
        'Train': {
            'dataset': dataset(hp.get('train_data_dir', 'data/'), hp.get('train_label_file', 'data/train_labels.txt'), True),
            'loader': {'shuffle': True, 'batch_size_per_card': batch, 'drop_last': True, 'num_workers': 4},
        },
        'Eval': {
            'dataset': dataset(hp.get('val_data_dir', 'data/'), hp.get('val_label_file', 'data/val_labels.txt'), False),
            'loader': {'shuffle': False, 'drop_last': False, 'batch_size_per_card': batch * 2, 'num_workers': 4},
        },
    }


def _find_train_py(hp, base):
    home = hp.get('paddleocr_home', '') or os.environ.get('PADDLEOCR_HOME', '')
    if not home:
        for rel in ('PaddleOCR', os.path.join('..', 'PaddleOCR')):
            c = os.path.normpath(os.path.join(base, rel))
            if os.path.isdir(c):
                home = c; break
    if not home:
        raise FileNotFoundError(
            "PaddleOCR source not found.\n"
            "Clone: git clone https://github.com/PaddlePaddle/PaddleOCR.git -b release/2.7\n"
            "Set 'paddleocr_home' in hyperparameters or PADDLEOCR_HOME env var."
        )
    train_py = os.path.join(home, 'tools', 'train.py')
    if not os.path.exists(train_py):
        raise FileNotFoundError(f"tools/train.py not found in: {home}")
    return train_py


if __name__ == "__main__":
    hp   = json.loads(parameters)['hyperparameter']
    base = os.path.dirname(os.path.abspath(__file__))
    temp = os.path.join(base, "temp")

    _validate(hp, base)
    train_py = _find_train_py(hp, base)

    os.makedirs(temp, exist_ok=True)
    yml = os.path.join(temp, "train_config.yml")
    try:
        with open(yml, 'w', encoding='utf-8') as f:
            yaml.dump(_build_yaml(hp, base), f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info("Training Started.")
        device = hp.get('device', 'gpu').lower()
        cmd = ([sys.executable, "-m", "paddle.distributed.launch", "--gpus", hp.get('gpu_id', '0'), train_py, "-c", yml]
               if device == 'gpu' else
               [sys.executable, train_py, "-c", yml])
        logger.info(f"CMD: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=base)
        logger.info(f"Training Ended. Output: {_abs(base, hp.get('save_dir', 'output/'))}")
    except Exception as e:
        logger.error(f"Error Message : {e}")
        raise Exception(f"Train Failed, Error Message : {e}")
    finally:
        shutil.rmtree(temp, ignore_errors=True)
