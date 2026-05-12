import os
import yaml


def build_yaml_config(config: dict, base_dir: str) -> dict:
    image_shape     = config.get('image_shape', [3, 48, 320])
    max_text_length = config.get('max_text_length', 50)
    epoch_num       = config.get('epoch_num', 100)
    use_guided      = config.get('use_guided_training', False)

    save_epoch_step = config.get('save_epoch_step', 10)
    if save_epoch_step == 0:
        save_epoch_step = epoch_num

    def abs_path(p):
        return os.path.join(base_dir, p) if p else None

    global_cfg = {
        'use_gpu':               config.get('device', 'gpu').lower() == 'gpu',
        'epoch_num':             epoch_num,
        'log_smooth_window':     20,
        'print_batch_step':      10,
        'save_model_dir':        abs_path(config.get('save_dir', 'output/')),
        'save_epoch_step':       save_epoch_step,
        'eval_batch_step':       [0, 2000],
        'cal_metric_during_train': True,
        'pretrained_model':      abs_path(config.get('pretrained_model', '')),
        'checkpoints':           abs_path(config.get('resume_path', '')) or None,
        'save_inference_dir':    None,
        'use_visualdl':          False,
        'character_dict_path':   abs_path(config['rec_char_dict_path']),
        'max_text_length':       max_text_length,
        'infer_mode':            False,
        'use_space_char':        config.get('use_space_char', False),
        'distributed':           False,
    }

    optimizer_cfg = {
        'name': 'Adam', 'beta1': 0.9, 'beta2': 0.999,
        'lr': {
            'name': 'Cosine',
            'learning_rate': config.get('learning_rate', 0.0005),
            'warmup_epoch': 5,
        },
        'regularizer': {'name': 'L2', 'factor': 3.0e-05},
    }

    arch_cfg = {
        'model_type': 'rec', 'algorithm': 'SVTR_LCNet', 'Transform': None,
        'Backbone': {
            'name': 'MobileNetV1Enhance', 'scale': 0.5,
            'last_conv_stride': [1, 2], 'last_pool_type': 'avg',
            'last_pool_kernel_size': [2, 4],
        },
        'Neck': {
            'name': 'SequenceEncoder', 'encoder_type': 'svtr',
            'dims': 64, 'depth': 2, 'hidden_dims': 120, 'use_guide': True,
        },
        'Head': {'name': 'CTCHead', 'fc_decay': 0.00001},
    }

    if use_guided:
        label_encode = {'MultiLabelEncode': {}}
        keep_keys    = ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']
    else:
        label_encode = {'CTCLabelEncode': {'max_text_length': max_text_length}}
        keep_keys    = ['image', 'label', 'length']

    def make_transforms(is_train: bool) -> list:
        t = [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
        if is_train:
            t.append({'RecAug': {}})
        t.append(label_encode)
        t.append({'RecResizeImg': {'image_shape': image_shape}})
        t.append({'KeepKeys': {'keep_keys': keep_keys}})
        return t

    batch = config.get('batch_size', 64)
    train_cfg = {
        'dataset': {
            'name': 'SimpleDataSet',
            'data_dir':        abs_path(config.get('train_data_dir', 'data/')),
            'label_file_list': [abs_path(config.get('train_label_file', 'data/train_labels.txt'))],
            'transforms':      make_transforms(True),
        },
        'loader': {'shuffle': True, 'batch_size_per_card': batch, 'drop_last': True, 'num_workers': 4},
    }
    eval_cfg = {
        'dataset': {
            'name': 'SimpleDataSet',
            'data_dir':        abs_path(config.get('val_data_dir', 'data/')),
            'label_file_list': [abs_path(config.get('val_label_file', 'data/val_labels.txt'))],
            'transforms':      make_transforms(False),
        },
        'loader': {'shuffle': False, 'drop_last': False, 'batch_size_per_card': batch * 2, 'num_workers': 4},
    }

    return {
        'Global':       global_cfg,
        'Optimizer':    optimizer_cfg,
        'Architecture': arch_cfg,
        'Loss':         {'name': 'CTCLoss' if not use_guided else 'MultiLoss'},
        'PostProcess':  {'name': 'CTCLabelDecode'},
        'Metric':       {'name': 'RecMetric', 'main_indicator': 'acc', 'ignore_space': False},
        'Train':        train_cfg,
        'Eval':         eval_cfg,
    }


def save_yaml_config(config: dict, base_dir: str, output_path: str):
    data = build_yaml_config(config, base_dir)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
