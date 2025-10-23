from torchvision import transforms
from PIL import Image

class HyperparameterBuilder:
    def __init__(self, keyword_arguments):
        self.__hyperparams = keyword_arguments

    def initialize(self):
        self.__hyperparams['epoch'] = int(self.__hyperparams['epoch'])
        self.__hyperparams['batch_size'] = int(self.__hyperparams['batch_size'])
        return self

    def get_train_ratio(self):
        return self.__hyperparams['train_ratio']
    
    def get_transform(self):
        input_size = self.__hyperparams['input_size']
        mean = self.__hyperparams['normalize_mean']
        stdev = self.__hyperparams['normalize_stdev']

        transform = transforms.Compose([    
        transforms.Lambda(lambda img: Image.fromarray(img).convert("RGB")),
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Normalize((mean, mean, mean), (stdev, stdev, stdev))
        ])
        return transform        

    def get_batch_size(self):
        return self.__hyperparams['batch_size']
    
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
    
    def get_epoch(self):
        return self.__hyperparams['epoch']
    
    def get_save_epoch(self):
        return self.__hyperparams['save_epoch']
    
    def get_using_amp(self):
        return self.__hyperparams['using_amp']
    
    def get_early_stop_patience(self):
        keys = ['earlyStopPatience']
        for key in keys:
            if key in self.__hyperparams:
                return self.__hyperparams[key]
        return 0
    
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

    def build(self):
        return self 