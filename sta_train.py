"""
    Implementation of the Separable STA from the "Toyota SmartHome" ICCV 2019 paper.
"""
import sta_config as cfg
from toyota_generator import ToyotaGeneratorTrain
import sys
sys.path.insert(0, '%s/ntu-i3d' % cfg.code_path)
sys.path.insert(0, '%s/LSTM_action_recognition' % cfg.code_path)

from separable_sta import separable_sta, separable_sta_loss
import i3d_config as i3d_cfg
from keras.callbacks import CSVLogger, Callback
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

_classes = 35
_dropout_prob = 0.5
_ver = 'sep_sta_rot_cw'
_batch_size = 4
_model_name = 'sep_sta_rot_cw'
_epochs = 50


lambda1 = 0.00001
lambda2 = 0.00001
# lr = 0.001
lr = 0.0001
adam = Adam(lr=lr)

train_generator = ToyotaGeneratorTrain('%s/splits_i3d/train_CS.txt' % i3d_cfg.dataset_dir, _ver, batch_size=_batch_size)
val_generator = ToyotaGeneratorTrain('%s/splits_i3d/validation_CS.txt' % i3d_cfg.dataset_dir, _ver, batch_size=_batch_size)

if cfg.gpus > 1:
    psta = multi_gpu_model(separable_sta, cfg.gpus)
else:
    psta = separable_sta

separable_sta.compile(optimizer=adam, loss=separable_sta_loss(lambda1, lambda2), metrics=['accuracy'])
if cfg.gpus > 1:
    psta.compile(optimizer=adam, loss=separable_sta_loss(lambda1, lambda2), metrics=['accuracy'])


class CustomModelCheckpoint(Callback):

    def __init__(self, model_parallel, path):
        super(CustomModelCheckpoint, self).__init__()

        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')


model_checkpoint = CustomModelCheckpoint(psta, './weights_' + _model_name + '/epoch_')
csv_logger = CSVLogger('sta_logging_' + _model_name + '.csv')

class_weights = {0: 0, 1: 11.48, 2: 10.29, 3: 26.44, 4: 7.47, 5: 37.46,
                 6: 107.88, 7: 0, 8: 13.42, 9: 13.69, 10: 1.90, 11: 47.32,
                 12: 7.93, 13: 17.29, 14: 9.17, 15: 5.58, 16: 24.74, 17: 9.08,
                 18: 0, 19: 69.15, 20: 58.63, 21: 61.30, 22: 74.92, 23: 16.45,
                 24: 79.32, 25: 0, 26: 35.96, 27: 4.68, 28: 4.20, 29: 13.76,
                 30: 13.29, 31: 84.28, 32: 9.17, 33: 1.00, 34: 5.86}

psta.fit_generator(
    generator=train_generator,
    validation_data=val_generator,
    class_weight=class_weights,
    epochs=_epochs,
    callbacks=[csv_logger, model_checkpoint],
    max_queue_size=48,
    # workers=cpu_count() - 2,
    # use_multiprocessing=True,
)
