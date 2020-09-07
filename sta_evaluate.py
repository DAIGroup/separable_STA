"""
    Test load_model and obtain results (per class accuracies)
"""
import numpy as np
import sta_config as cfg
import sys
sys.path.insert(0, '%s/ntu-i3d' % cfg.code_path)
sys.path.insert(0, '%s/LSTM_action_recognition' % cfg.code_path)

from separable_sta import separable_sta
from keras.models import Model, load_model
from toyota_generator import *
from tqdm import tqdm

_ver = 'sep_sta'

print('Loading model ...')
separable_sta.load_weights('./weights_sep_sta_rot_cw/epoch_37.hdf5')
print('Done.')

test_generator = ToyotaGeneratorTrain('%s/splits_i3d/test_CS.txt' % i3d_cfg.dataset_dir, _ver, batch_size=1)

num_tests = len(test_generator)
print('Testing %d samples.' % num_tests)

nc = test_generator.num_classes
conf_mat = np.zeros((nc, nc))

for i in tqdm(range(num_tests)):
    sample = test_generator[i]
    x, y = sample
    pred = separable_sta.predict(x)
    p = np.argmax(pred)
    t = np.argmax(y)
    conf_mat[t, p] += 1

np.savetxt("confusion_matrix_rot_cw_37.csv", conf_mat, delimiter=";")
print('FINISHED.')
