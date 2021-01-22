"""
    Configuration parameters for Separable STA
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
tf.device('/gpu:0')
# code_path = '/home/paal/deploy'
code_path = '/home/pau/harnets'
gpus = 1

experiment_name = 'sep_sta_rot_cw_2'
use_sample_weights = False

experiment = 'crossview'
if experiment == 'crossview':
    classes = 19
elif experiment == 'crosssubject':
    classes = 35
