"""
    Implementation of a data generator for both skeletons + RGB*t stacks.
"""
import sta_config as cfg

import sys
sys.path.insert(0, '%s/ntu-i3d' % cfg.code_path)
sys.path.insert(0, '%s/LSTM_action_recognition' % cfg.code_path)

import Smarthome_Loader as i3d_shl
import readers.smarthome_skeleton_fromjson_sampling as skel
import config as lstm_cfg
import i3d_config as i3d_cfg
import sta_config as sta_cfg
from keras.utils import Sequence, to_categorical
from random import sample, randint, shuffle
import numpy as np
import os.path

_skel_dims = 39
_step = 30


class ToyotaGenerator(Sequence):
    def __init__(self, split_path, version, batch_size = 4, is_test=True):
        self.batch_size = batch_size
        self.version = version
        self.crops_path = i3d_cfg.crops_dir
        self.skeletons_path = lstm_cfg.dataset_dir + '/%s/' % lstm_cfg.variant_dir
        self.files = [i.strip() for i in open(split_path).readlines()]
        self.stack_size = 64
        self.num_classes = sta_cfg.classes
        self.stride = 2
        self.is_test = is_test

    def __len__(self):
        return int(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.files[idx * self.batch_size: (idx + 1) * self.batch_size]
        i3d_batch = [os.path.splitext(i)[0] for i in batch]
        skel_x_train, skel_y_train = skel.generate_data(batch, self.skeletons_path,
                                                           self.batch_size, _step, _skel_dims, self.is_test)

        x_train = [i3d_shl.get_video(i, self.crops_path, self.stride, self.stack_size, self.is_test) for i in i3d_batch]
        x_train = np.array(x_train, np.float32)
        x_train /= 127.5
        crops_x_train = x_train - 1

        if cfg.experiment == 'crosssubject':
            y_train = np.array([i3d_shl.name_to_int(i.split('_')[0]) for i in batch]) - 1
        elif cfg.experiment == 'crossview':
            y_train = np.array([i3d_shl.name_to_int_CV(i.split('_')[0]) for i in batch]) - 1
        else:
            print('Error: No "experiment" defined in config file.')
            sys.exit(-1)
        # assert y_train == skel_y_train

        labels = y_train
        y_train = to_categorical(y_train, num_classes=self.num_classes)

        if cfg.use_sample_weights:
            w = np.zeros((self.batch_size,))
            for i in range(len(batch)):
                if cfg.experiment == 'crosssubject':
                    w[i] = i3d_shl.weights_CS[labels[i]]
                else:
                    w[i] = i3d_shl.weights_CV[labels[i]]
            return [skel_x_train, crops_x_train], y_train, w

        return [skel_x_train, crops_x_train], y_train

    def on_epoch_end(self):
        shuffle(self.files)
