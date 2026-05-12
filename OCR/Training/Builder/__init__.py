# Local (항상 로드)
from .HyperParameter.Json_HyperParameterBuilder import Json_HyperparameterBuilder
from .Dataset.Local_OCR_DatasetBuilder import Local_OCR_DatasetBuilder
from .Train.TrainHook import TrainHook
from .Train.PaddleOCR_TrainingBuilder import PaddleOCR_TrainingBuilder
from .Config.PaddleOCR_ConfigBuilder import save_yaml_config, build_yaml_config


# DAQ (mpp 필요 - lazy import)
def __getattr__(name):
    if name == 'Operation_Builder':
        from .Operation.DAQ_OperationBuilder import Operation_Builder
        return Operation_Builder
    if name == 'DAQ_OCR_DatasetBuilder':
        from .Dataset.DAQ_OCR_DatasetBuilder import DAQ_OCR_DatasetBuilder
        return DAQ_OCR_DatasetBuilder
    if name == 'DAQ_OCR_SaveBuilder':
        from .Save.DAQ_OCR_SaveBuilder import DAQ_OCR_SaveBuilder
        return DAQ_OCR_SaveBuilder
    if name == 'DAQ_MoritoringBuilder':
        from .Monitoring.DAQ_MonitoringBuilder import DAQ_MoritoringBuilder
        return DAQ_MoritoringBuilder
    raise AttributeError(f"module 'Builder' has no attribute '{name}'")
