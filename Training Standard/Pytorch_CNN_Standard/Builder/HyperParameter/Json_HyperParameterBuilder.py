import json

parameters = '''{
    "hyperparameter":{
        "network_name" : "efficientnet_b0",
        "epoch" : 20,
        "save_epoch" : 1,
        "batch_size" : 16,
        "lr" : 1e-3,
        "optimizer_name" : "Adam",
        "criterion" : "CrossEntropyLoss",
        "input_size" : 224,
        "normalize_mean" : 0.5,
        "normalize_stdev" : 0.5,
        "using_gpu" : true,
        "using_amp" : true,
        "train_ratio" : 0.8,
        "validation_save_random" : false
    }
}'''

class Json_HyperparameterBuilder:
    def __init__(self, json_parameter = parameters):
        self.__hyperparams = json.loads(json_parameter)

    def get_train_ratio(self):
        return self.__hyperparams['train_ratio']
    
    def get_batch_size(self)->int:
        return int(self.__hyperparams['batch_size'])
    
    def get_validation_save_random(self):
        return self.__hyperparams.get('validation_save_random', False)
    
    def get_optimizer(self):
        keys = ['optimizer', 'optimizer_name']
        for key in keys:
            if key in self.__hyperparams:
                return self.__hyperparams[key]

        raise KeyError(f"Optimizer not found. Tried keys: {', '.join(keys)}")
    
    def get_learning_rate(self):
        keys = ['lr', 'learningRate', 'LearningRate', 'Learningrate']
        for key in keys:
            if key in self.__hyperparams:
                return self.__hyperparams[key]

        raise KeyError(f"Learning rate not found. Tried keys: {', '.join(keys)}")
    
    def get_input_size(self):
        return self.__hyperparams['input_size']
    
    def get_epoch(self)->int:
        return int(self.__hyperparams['epoch'])
    
    def get_save_epoch(self)->int:
        return int(self.__hyperparams['save_epoch'])
    
    def get_using_amp(self):
        return self.__hyperparams['using_amp']
    
    def get_early_stop_patience(self):
        keys = ['earlyStopPatience']
        for key in keys:
            if key in self.__hyperparams:
                return self.__hyperparams[key]
        return 0
    
    def get_network_name(self):
        return self.__hyperparams['network_name']
    
    def get_reduce_learning_rate_patience(self):
        keys = ['reduceLRPatience']
        for key in keys:
            if key in self.__hyperparams:
                return self.__hyperparams[key]
        return 0

    def get_criterion(self):
        keys = ['criterion', 'loss']
        for key in keys:
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

    def get_resume_path(self):
        return self.__hyperparams.get('resume_path', '')

    def get_augmentation(self):
        return self.__hyperparams.get('augmentation', {})

    def get_device(self):
        if self.get_using_gpu():
            return 'cuda'
        else:
            return 'cpu'

    def build(self):
        return self