# Local (mpp 불필요)
from .Save.Local_SaveBuilder import Local_SaveBuilder
from .Save.Local_SaveBuilder_DAQ_OLD import Local_SaveBuilder_DAQ_OLD
from .HyperParameter.Json_HyperParameterBuilder import Json_HyperparameterBuilder
from .ClassCode.Local_Classification_ClassCodeBuilder import Local_Classification_ClassCodeBuilder
from .Dataset.Local_Tensorflow_ClassificationDatasetBuilder import Local_Tensorflow_ClassificationDatasetBuilder
from .Model.Tensorflow_Classification_Models import Tensorflow_Classification_Models
from .Train.Tensorflow_TrainingBuilder import Tensorflow_TrainingBuilder
from .Train.Tensorflow_TrainingBuilder_Debug import Tensorflow_TrainingBuilder_Debug
from .Train.Tensorflow_TrainingBuilder import TrainHook
from .Monitoring.Local_MonitoringBuilder import Local_MonitoringBuilder

# DAQ (mpp 필요 - lazy import)
def __getattr__(name):
    if name == 'DAQ_SaveBuilder':
        from .Save.DAQ_Savebuilder import DAQ_SaveBuilder
        return DAQ_SaveBuilder
    if name == 'DAQ_SaveBuilder_DAQ_OLD':
        from .Save.DAQ_Savebuilder_DAQ_OLD import DAQ_SaveBuilder_DAQ_OLD
        return DAQ_SaveBuilder_DAQ_OLD
    if name == 'DAQ_Classification_ClassCodeBuilder':
        from .ClassCode.DAQ_Classification_ClassCodeBuilder import DAQ_Classification_ClassCodeBuilder
        return DAQ_Classification_ClassCodeBuilder
    if name == 'DAQ_Tensorflow_ClassificationDatasetBuilder':
        from .Dataset.DAQ_Tensorflow_ClassificationDatasetBuilder import DAQ_Tensorflow_ClassificationDatasetBuilder
        return DAQ_Tensorflow_ClassificationDatasetBuilder
    if name == 'Operation_Builder':
        from .Operation.DAQ_OperationBuilder import Operation_Builder
        return Operation_Builder
    if name == 'DAQ_MoritoringBuilder':
        from .Monitoring.DAQ_MonitoringBuilder import DAQ_MoritoringBuilder
        return DAQ_MoritoringBuilder
    raise AttributeError(f"module 'Builder' has no attribute '{name}'")

__all__ = [
    'DAQ_SaveBuilder',
    'DAQ_SaveBuilder_DAQ_OLD',
    'Local_SaveBuilder',
    'Local_SaveBuilder_DAQ_OLD',
    'Json_HyperparameterBuilder',
    'DAQ_Classification_ClassCodeBuilder',
    'Local_Classification_ClassCodeBuilder',
    'DAQ_Tensorflow_ClassificationDatasetBuilder',
    'Local_Tensorflow_ClassificationDatasetBuilder',
    'Tensorflow_Classification_Models',
    'Tensorflow_TrainingBuilder',
    'Tensorflow_TrainingBuilder_Debug',
    'TrainHook',
    'Operation_Builder',
    'DAQ_MoritoringBuilder',
    'Local_MonitoringBuilder',
]
