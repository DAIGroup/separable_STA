"""
    Configuration parameters for Separable STA
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
tf.device('/gpu:0')
code_path = '/home/paal/deploy'
gpus = 1
