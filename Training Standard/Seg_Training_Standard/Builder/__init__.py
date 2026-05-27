# 공통 (framework 무관)
from .Save.Local_SaveBuilder import Local_SaveBuilder
from .Save.Local_SaveBuilder_DAQ_OLD import Local_SaveBuilder_DAQ_OLD
from .HyperParameter.Json_HyperParameterBuilder import Json_HyperparameterBuilder
from .ClassCode.Local_Segmentation_ClassCodeBuilder import Local_Segmentation_ClassCodeBuilder
from .Train.TrainHook import TrainHook
from .Monitoring.Local_MonitoringBuilder import Local_MonitoringBuilder


def get_framework_builders(framework: str):
    """framework에 따라 Model, LocalDatasetBuilder, TrainingBuilder를 반환"""
    if framework == 'pytorch':
        from .Model.Pytorch_Segmentation_Models import Pytorch_Segmentation_Models as SegmentationModels
        from .Dataset.Local_Pytorch_SegmentationDatasetBuilder import Local_Pytorch_SegmentationDatasetBuilder as LocalDatasetBuilder
        from .Train.Pytorch_TrainingBuilder import Pytorch_TrainingBuilder as TrainingBuilder
    elif framework == 'tensorflow':
        from .Model.Tensorflow_Segmentation_Models import Tensorflow_Segmentation_Models as SegmentationModels
        from .Dataset.Local_Tensorflow_SegmentationDatasetBuilder import Local_Tensorflow_SegmentationDatasetBuilder as LocalDatasetBuilder
        from .Train.Tensorflow_TrainingBuilder import Tensorflow_TrainingBuilder as TrainingBuilder
    else:
        raise ValueError(f"Unsupported framework: {framework}. Use 'pytorch' or 'tensorflow'")

    return SegmentationModels, LocalDatasetBuilder, TrainingBuilder


def get_daq_framework_builders(framework: str):
    """framework에 따라 DAQ용 Dataset Builder를 반환"""
    if framework == 'pytorch':
        from .Dataset.DAQ_Pytorch_SegmentationDatasetBuilder import DAQ_Pytorch_SegmentationDatasetBuilder as DAQDatasetBuilder
    elif framework == 'tensorflow':
        from .Dataset.DAQ_Tensorflow_SegmentationDatasetBuilder import DAQ_Tensorflow_SegmentationDatasetBuilder as DAQDatasetBuilder
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    return DAQDatasetBuilder


# DAQ (mpp 필요 - lazy import)
def __getattr__(name):
    if name == 'DAQ_SaveBuilder':
        from .Save.DAQ_Savebuilder import DAQ_SaveBuilder
        return DAQ_SaveBuilder
    if name == 'DAQ_SaveBuilder_DAQ_OLD':
        from .Save.DAQ_Savebuilder_DAQ_OLD import DAQ_SaveBuilder_DAQ_OLD
        return DAQ_SaveBuilder_DAQ_OLD
    if name == 'DAQ_Segmentation_ClassCodeBuilder':
        from .ClassCode.DAQ_Segmentation_ClassCodeBuilder import DAQ_Segmentation_ClassCodeBuilder
        return DAQ_Segmentation_ClassCodeBuilder
    if name == 'Operation_Builder':
        from .Operation.DAQ_OperationBuilder import Operation_Builder
        return Operation_Builder
    if name == 'DAQ_MoritoringBuilder':
        from .Monitoring.DAQ_MonitoringBuilder import DAQ_MoritoringBuilder
        return DAQ_MoritoringBuilder
    raise AttributeError(f"module 'Builder' has no attribute '{name}'")
