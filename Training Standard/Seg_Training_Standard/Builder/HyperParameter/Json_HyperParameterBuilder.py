import json

parameters = '''{
    "hyperparameter":{
        "framework" : "pytorch",
        "network_name" : "deeplabv3_resnet50",
        "epoch" : 20,
        "save_epoch" : 1,
        "batch_size" : 4,
        "lr" : 1e-2,
        "weight_decay" : 1e-4,
        "optimizer_name" : "SGD",
        "criterion" : "CrossEntropyLoss",
        "input_size" : 512,
        "normalize_mean" : 0.5,
        "normalize_stdev" : 0.5,
        "using_gpu" : true,
        "using_amp" : false,
        "train_ratio" : 0.8,
        "validation_save_random" : false,
        "validation_save_count" : 10,
        "output_stride" : 16,
        "pretrained_backbone" : true
    }
}'''

class Json_HyperparameterBuilder:
    def __init__(self, json_parameter=parameters):
        self.__hyperparams = json.loads(json_parameter)

    def get_framework(self):
        return self.__hyperparams.get('framework', 'pytorch')

    def get_train_ratio(self):
        return self.__hyperparams['train_ratio']

    def get_batch_size(self) -> int:
        return int(self.__hyperparams['batch_size'])

    def get_validation_save_random(self):
        return self.__hyperparams.get('validation_save_random', False)

    def get_validation_save_count(self) -> int:
        return int(self.__hyperparams.get('validation_save_count', 10))

    def get_optimizer(self):
        for key in ['optimizer', 'optimizer_name']:
            if key in self.__hyperparams:
                return self.__hyperparams[key]
        return 'SGD'

    def get_learning_rate(self):
        for key in ['lr', 'learningRate', 'LearningRate']:
            if key in self.__hyperparams:
                return self.__hyperparams[key]
        raise KeyError("Learning rate not found")

    def get_weight_decay(self):
        return float(self.__hyperparams.get('weight_decay', 1e-4))

    def get_input_size(self):
        return int(self.__hyperparams.get('input_size', 512))

    def get_epoch(self) -> int:
        return int(self.__hyperparams['epoch'])

    def get_save_epoch(self) -> int:
        return int(self.__hyperparams['save_epoch'])

    def get_using_amp(self):
        return self.__hyperparams['using_amp']

    def get_network_name(self):
        return self.__hyperparams['network_name']

    def get_criterion(self):
        for key in ['criterion', 'loss']:
            if key in self.__hyperparams:
                return self.__hyperparams[key]
        return 'CrossEntropyLoss'

    def get_normalize_mean(self):
        return self.__hyperparams['normalize_mean']

    def get_normalize_stdev(self):
        return self.__hyperparams['normalize_stdev']

    def get_using_gpu(self):
        return self.__hyperparams['using_gpu']

    def get_debug(self):
        return self.__hyperparams.get('debug', False)

    def get_daq_old_path(self):
        return self.__hyperparams.get('daq_old_path', False)

    def get_augmentation(self):
        return self.__hyperparams.get('augmentation', {})

    def get_loss_eps(self):
        return float(self.__hyperparams.get('loss_eps', 0))

    def get_output_stride(self) -> int:
        return int(self.__hyperparams.get('output_stride', 16))

    def get_pretrained_backbone(self) -> bool:
        return bool(self.__hyperparams.get('pretrained_backbone', True))

    def get_device(self):
        fw = self.get_framework()
        if fw == 'pytorch':
            return 'cuda' if self.get_using_gpu() else 'cpu'
        else:
            return '/gpu:0' if self.get_using_gpu() else '/cpu:0'

    def build(self):
        return self
