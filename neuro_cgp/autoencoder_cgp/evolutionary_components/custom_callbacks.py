import tensorflow as tf
import numpy as np

from autoencoder_cgp.evolutionary_components.logger import ConvModelInfoLogger

callbacks = tf.keras.callbacks
K = tf.keras.backend



class CustomEarlyStopping(callbacks.Callback):
    """Stop training when a monitored quantity has stopped improving.

  Arguments:
      monitor: Quantity to be monitored.
      min_delta: Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: Number of epochs with no improvement
          after which training will be stopped.
      verbose: verbosity mode.
      mode: One of `{"auto", "min", "max"}`. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `max`
          mode it will stop when the quantity
          monitored has stopped increasing; in `auto`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
      baseline: Baseline value for the monitored quantity.
          Training will stop if the model doesn't show improvement over the
          baseline.
      restore_best_weights: Whether to restore model weights from
          the epoch with the best value of the monitored quantity.
          If False, the model weights obtained at the last step of
          training are used.

  Example:

  ```python
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
  # This callback will stop the training when there is no improvement in
  # the validation loss for three consecutive epochs.
  model.fit(data, labels, epochs=100, callbacks=[callback],
      validation_data=(val_data, val_labels))
  ```
  """

    def __init__(self,
                 monitor='loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(CustomEarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            print('EarlyStopping mode %s is unknown, '
                  'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    else:
                        print("Restoring model weights failed because the model was at no point better than the baseline specified by EarlyStopping.")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        return monitor_value


class WandbSaveModelCallback(callbacks.Callback):

    def __init__(self, model_info_logger: "ConvModelInfoLogger"):
        super(WandbSaveModelCallback, self).__init__()
        self.model_info_logger = model_info_logger

    def on_train_end(self, logs=None):
        self.model : tf.keras.models.Model
        self.model_info_logger.save_model(self.model)



class ReduceLRonPerformanceDrop(callbacks.Callback):
    """Stop training when a monitored quantity has stopped improving.

  Arguments:
      monitor: Quantity to be monitored.
      min_delta: Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: Number of epochs with no improvement
          after which training will be stopped.
      verbose: verbosity mode.
      mode: One of `{"auto", "min", "max"}`. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `max`
          mode it will stop when the quantity
          monitored has stopped increasing; in `auto`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
      baseline: Baseline value for the monitored quantity.
          Training will stop if the model doesn't show improvement over the
          baseline.
      restore_best_weights: Whether to restore model weights from
          the epoch with the best value of the monitored quantity.
          If False, the model weights obtained at the last step of
          training are used.

  Example:

  ```python
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
  # This callback will stop the training when there is no improvement in
  # the validation loss for three consecutive epochs.
  model.fit(data, labels, epochs=100, callbacks=[callback],
      validation_data=(val_data, val_labels))
  ```
  """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 min_lr=0,
                 factor=0.1,
                 patience=0,
                 verbose=0,
                 mode='auto'):

        super(ReduceLRonPerformanceDrop, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.best_weights = None
        self.best_epoch = 0
        self.current_monitor_value = None
        self.min_lr = min_lr
        self.factor = factor

        if mode not in ['auto', 'min', 'max']:
            print('EarlyStopping mode %s is unknown, '
                  'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        self.current_monitor_value = current

        # Check if performance increased compared to previous epochs
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            # Check if model has not improved for the a period of epochs specified by parameter patience
            if self.wait >= self.patience:
                self.wait = 0
                self.model.set_weights(self.best_weights)
                old_lr = float(K.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    K.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        print(f'Restoring model weights from the end of the best epoch and reducing the learning rate '
                              f'to {new_lr}.')


    def on_train_end(self, logs=None):

        current = self.current_monitor_value
        if current is None:
            print("Could get the monitor value on training end. Therefore it was not possible to restore the best weights.")
            return

        # reset the model weights, if there was a better loss at some point prior to the end of training
        if self.monitor_op(current - self.min_delta, self.best):
            self.model.set_weights(self.best_weights)
            print(f"There was a better loss at some point prior to the end of training. Resetting weights to weights of epoch {self.best_epoch +1}.")


    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        return monitor_value
