"""
    Implementation of the Separable STA from the "Toyota SmartHome" ICCV 2019 paper.
"""
import sta_config as cfg
from toyota_generator import ToyotaGeneratorTrain
import sys
sys.path.insert(0, '%s/ntu-i3d' % cfg.code_path)
sys.path.insert(0, '%s/LSTM_action_recognition' % cfg.code_path)

import i3d_config as i3d_cfg
from multiprocessing import cpu_count
from keras.callbacks import CSVLogger, Callback
from keras.layers import Dense, Dropout, AveragePooling3D, Conv3D, Activation
from keras.layers import Multiply, Reshape, Lambda, Concatenate
from keras.models import Model, load_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import keras.backend as K

path_to_lstm = '%s/LSTM_action_recognition/lstm.hdf5' % cfg.code_path
full_lstm = load_model(path_to_lstm)
full_lstm.summary()
lstm = Model(inputs=full_lstm.input, outputs=full_lstm.get_layer(index=2).output)
lstm.trainable = False

path_to_i3d = '%s/ntu-i3d/i3d.hdf5' % cfg.code_path
full_i3d = load_model(path_to_i3d)
full_i3d.summary()
# check size of mixed_5c (index=-6), which should be (b) x t x m x n x c (None, 8, 7, 7, feats)
i3d = Model(inputs=full_i3d.input, outputs=full_i3d.get_layer(index=-6).output)
i3d.trainable = False
_, t, m, n, c = i3d.output.get_shape().as_list()
_classes = 35
_dropout_prob = 0.5
_ver = 'sep_sta'
_batch_size = 4
_model_name = 'sep_sta'
_epochs = 100


def reshape_spatial_attention(x):
    x = K.reshape(x, shape=(-1, 1, m, n, 1))
    x = K.repeat_elements(x, c, axis=-1)
    x = K.repeat_elements(x, t, axis=1)
    return x


def reshape_temporal_attention(x):
    x = K.reshape(x, shape=(-1, t, 1, 1, 1))
    x = K.repeat_elements(x, m, axis=2)
    x = K.repeat_elements(x, n, axis=3)
    x = K.repeat_elements(x, c, axis=4)
    return x


sa = Dense(256, activation='tanh')(lstm.output)
alpha = Dense(m*n, activation='sigmoid')(sa)
sa = Reshape(target_shape=(m, n))(alpha)
sa = Lambda(reshape_spatial_attention)(sa)
gs = Multiply()([sa, i3d.output])

ta = Dense(256, activation='tanh')(lstm.output)
beta = Dense(t, activation='softmax')(ta)
ta = Lambda(reshape_temporal_attention)(beta)
gt = Multiply()([ta, i3d.output])


gs = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool_gs')(gs)
gt = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool_gt')(gt)

x = Concatenate()([gs, gt])
x = Dropout(_dropout_prob)(x)
x = Conv3D(filters=_classes, kernel_size=(1, 1, 1), padding='same', use_bias=True)(x)

num_frames_remaining = int(x.shape[1])
x = Reshape((num_frames_remaining, _classes))(x)

# logits (raw scores for each class)
x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
           output_shape=lambda s: (s[0], s[2]))(x)

x = Activation('softmax', name='prediction')(x)


separable_sta = Model(inputs=[lstm.input, i3d.input], outputs=x)


def separable_sta_loss(lambda1, lambda2):
    def sep_loss(y_actual, y_pred):
        Lc = categorical_crossentropy(y_actual, y_pred)
        spatial_loss = lambda1 * K.sqrt(K.sum(K.square(alpha)))
        temporal_loss = lambda2 * K.sum(K.square(1 - beta))
        return Lc + spatial_loss + temporal_loss
    return sep_loss