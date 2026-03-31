
class DAQ_MoritoringBuilder:
    def __init__(self, logger = None):
        self.__logger = logger

    def monitoring(self, total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time):
        remaining_epochs = total_epoch - epoch
        estimated_time_per_epoch = epoch_elapsed_time if epoch > 1 else 0
        estimated_remaining_time = remaining_epochs * estimated_time_per_epoch
        if self.__logger: self.__logger.info("MonitoringData:"
            f"Epoch:[{epoch:4d}/{total_epoch:4d}], "
            f"Train Loss: {train_loss:4.4f}, "
            f"Valid Loss : {validation_loss:4.4f}, "
            f"Time: {epoch_elapsed_time:4.2f}s, "
            f"Estimated Remaining Time: {estimated_remaining_time / 60:.2f} minutes")

    def builder(self):
        return self
