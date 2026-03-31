class TrainHook:
    def training_start(self): pass
    def on_epoch_end(self, total_epoch, epoch, train_loss, validation_loss, epoch_elapsed_time, model): pass
    def training_end(self): pass