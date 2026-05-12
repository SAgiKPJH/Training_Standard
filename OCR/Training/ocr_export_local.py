parameters = '''{
    "version": "OCR_Training_Standard_v1.0.0",
    "hyperparameter": {
        "checkpoint":          "data/pretrained/en_PP-OCRv3_rec_train/best_accuracy",
        "rec_char_dict_path":  "data/en_dict.txt",
        "save_inference_dir":  "output/inference-pretrained",
        "image_shape":         [3, 48, 320],
        "max_text_length":     50,
        "use_space_char":      false,
        "use_guided_training": false,
        "device":              "gpu",
        "paddleocr_home":      ""
    }
}'''
# output/iter_epoch_100

import json, logging, os, shutil, subprocess, sys
import yaml

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(message)s')


def _abs(base, p):
    return os.path.join(base, p) if p and not os.path.isabs(p) else (p or None)


def _build_yaml(hp, base):
    guided  = bool(hp.get('use_guided_training', False))
    shape   = hp.get('image_shape', [3, 48, 320])
    max_len = int(hp['max_text_length'])

    label_enc = ({'MultiLabelEncode': {}} if guided
                 else {'CTCLabelEncode': {'max_text_length': max_len}})
    keys      = (['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio'] if guided
                 else ['image', 'label', 'length'])

    def transforms():
        return [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                label_enc,
                {'RecResizeImg': {'image_shape': shape}},
                {'KeepKeys': {'keep_keys': keys}}]

    return {
        'Global': {
            'use_gpu':             hp.get('device', 'gpu').lower() == 'gpu',
            'epoch_num':           1,
            'save_model_dir':      _abs(base, 'output/'),
            'save_epoch_step':     1,
            'eval_batch_step':     [0, 2000],
            'cal_metric_during_train': False,
            'pretrained_model':    None,
            'checkpoints':         None,
            'save_inference_dir':  _abs(base, hp.get('save_inference_dir', 'output/inference')),
            'use_visualdl':        False,
            'character_dict_path': _abs(base, hp['rec_char_dict_path']),
            'max_text_length':     max_len,
            'infer_mode':          False,
            'use_space_char':      bool(hp.get('use_space_char', False)),
            'distributed':         False,
        },
        'Optimizer': {
            'name': 'Adam', 'beta1': 0.9, 'beta2': 0.999,
            'lr': {'name': 'Cosine', 'learning_rate': 0.001},
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
            'dataset': {
                'name': 'SimpleDataSet', 'data_dir': _abs(base, 'data/'),
                'label_file_list': [_abs(base, 'data/train_labels.txt')],
                'transforms': transforms(),
            },
            'loader': {'shuffle': False, 'batch_size_per_card': 1, 'drop_last': False, 'num_workers': 0},
        },
        'Eval': {
            'dataset': {
                'name': 'SimpleDataSet', 'data_dir': _abs(base, 'data/'),
                'label_file_list': [_abs(base, 'data/val_labels.txt')],
                'transforms': transforms(),
            },
            'loader': {'shuffle': False, 'drop_last': False, 'batch_size_per_card': 1, 'num_workers': 0},
        },
    }


def _find_export_py(hp, base):
    home = hp.get('paddleocr_home', '') or os.environ.get('PADDLEOCR_HOME', '')
    if not home:
        for rel in ('PaddleOCR', os.path.join('..', 'PaddleOCR')):
            c = os.path.normpath(os.path.join(base, rel))
            if os.path.isdir(c):
                home = c; break
    if not home:
        raise FileNotFoundError(
            "PaddleOCR source not found.\n"
            "Clone: git clone https://github.com/PaddlePaddle/PaddleOCR.git -b release/2.7"
        )
    export_py = os.path.join(home, 'tools', 'export_model.py')
    if not os.path.exists(export_py):
        raise FileNotFoundError(f"tools/export_model.py not found in: {home}")
    return export_py


if __name__ == "__main__":
    hp   = json.loads(parameters)['hyperparameter']
    base = os.path.dirname(os.path.abspath(__file__))
    temp = os.path.join(base, "temp")

    checkpoint = _abs(base, hp['checkpoint'])
    if not os.path.exists(checkpoint + '.pdparams'):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}.pdparams")

    export_py = _find_export_py(hp, base)
    save_dir  = _abs(base, hp.get('save_inference_dir', 'output/inference'))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(temp, exist_ok=True)

    yml = os.path.join(temp, "export_config.yml")
    try:
        with open(yml, 'w', encoding='utf-8') as f:
            yaml.dump(_build_yaml(hp, base), f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        cmd = [
            sys.executable, export_py,
            "-c", yml,
            "-o", f"Global.pretrained_model={checkpoint}",
            "-o", f"Global.save_inference_dir={save_dir}",
        ]
        logger.info(f"Exporting: {checkpoint}.pdparams")
        logger.info(f"CMD: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=base)
        logger.info(f"Export Done. Inference model saved to: {save_dir}")
        logger.info(f"  {save_dir}/inference.pdmodel")
        logger.info(f"  {save_dir}/inference.pdiparams")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        shutil.rmtree(temp, ignore_errors=True)
