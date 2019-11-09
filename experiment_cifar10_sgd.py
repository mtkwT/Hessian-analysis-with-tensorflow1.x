import math
import os
import sys
from datetime import datetime
from typing import List

import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from scipy import linalg
from tqdm import tqdm

sys.path.append('src/')
from layers import get_params, f_props
from layers import Conv, Dense, Flatten, Pooling
from optimizer import SGD
from prepare_data import PrepareTrainValidData

SEED = np.random.randint(1000)
tf.compat.v1.set_random_seed(SEED)

OUTPUT_DIR = './results/'
dt_now = datetime.now()
dt_now = dt_now.strftime('CIFAR10_SGD_SEED{}_%Y-%m-%d-%H-%M-%S/'.format(SEED))
if not os.path.exists(OUTPUT_DIR+dt_now):
    OUTPUT_DIR += dt_now
    os.mkdir(OUTPUT_DIR)

def tf_log(x):
    return tf.math.log(tf.clip_by_value(x, 1e-10, x))

class TrainCifar10CNN_SGD(object):
    def __init__(
                self,
                learning_rate,
                momentum,
                batch_size,
                epoch_size,
                per_process_gpu_memory_fraction,
            ) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        self.x_train, self.x_valid, self.t_train, self.t_valid = PrepareTrainValidData().cifar10()
        self.x, self.t, self.is_training, self.y, self.cost, self.params = self.build_model()
    
    @staticmethod
    def build_model():
        tf.compat.v1.reset_default_graph()
        x = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3]) 
        t = tf.compat.v1.placeholder(tf.float32, [None, 10]) 
        is_training = tf.compat.v1.placeholder(tf.bool)
        layers = [
            Conv((3, 3, 3, 64), tf.nn.relu),
            Conv((3, 3, 64, 64), tf.nn.relu),
            Pooling((1, 2, 2, 1)),
            Conv((3, 3, 64, 128), tf.nn.relu),
            Conv((3, 3, 128, 128), tf.nn.relu),
            Pooling((1, 2, 2, 1)),
            Flatten(),
            Dense(3200, 256, tf.nn.relu),
            Dense(256, 256, tf.nn.relu),
            Dense(256, 10, tf.nn.softmax)
        ]
        y = f_props(layers, x)
        
        params = get_params(layers)
        cost = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1))

        return x, t, is_training, y, cost, params
    
    def save_result(self, save_list, reslut_name):
        filename = \
        '{rn}_cifar10_SGD_lr{lr}_batchsize{bs}_epochsize{es}_seed{seed}.npy'.format(
            rn=reslut_name,
            lr=self.learning_rate, 
            bs=self.batch_size, 
            es=self.epoch_size,
            seed=SEED)
        np.save(OUTPUT_DIR+filename, save_list)
        print('save {} done!'.format(filename))

    def extract_gradient_norm(self, all_grad, mode, feed):
        grad = self.sess.run(all_grad[-1], feed_dict=feed)
        norm = np.linalg.norm(grad, ord=2)
        print("{} grad norm: ".format(mode), norm)
        return norm
    
    def extract_cost_and_acc(self, mode, feed):
        y_pred, cost_ = self.sess.run([self.y, self.cost], feed_dict=feed)
        if mode == 'Train':
            acc_ = accuracy_score(self.t_train[:20000].argmax(axis=1), y_pred[:20000].argmax(axis=1))
        else:
            acc_ = accuracy_score(self.t_valid.argmax(axis=1), y_pred.argmax(axis=1))
        print('EPOCH: {}, {} Cost: {:.3f}, {} Accuracy: {:.3f}'.format(self.epoch+1, mode, cost_, mode, acc_))
        return cost_, acc_

    def calculate_loss_surface(self, optimizer, hesse_matrix, feed):
        eig_val, eig_vec = linalg.eigh(hesse_matrix, eigvals=(len(hesse_matrix)-2, len(hesse_matrix)-1))
        v1 = eig_vec.T[0].reshape(256, 10)
        v2 = eig_vec.T[1].reshape(256, 10)
        
        cross_entropy_values_2D = []
        for c1 in tqdm([(_ / 100) for _ in range(-30, 31, 5)]):
            cross_entropy_values_1D = []

            for c2 in [(_ / 100) for _ in range(-30, 31, 5)]:
                add_updates_freeze = optimizer.add_eigvec_update(c1, v1, c2, v2)
                update = tf.group(*add_updates_freeze)
                self.sess.run(update, feed_dict=feed)
                cl_value = self.sess.run(self.cost, feed_dict=feed)
                cross_entropy_values_1D.append(cl_value)
                # TODO: modify numerical error
                sub_updates_freeze = optimizer.sub_eigvec_update(c1, v1, c2, v2)
                update = tf.group(*sub_updates_freeze)
                self.sess.run(update, feed_dict=feed)
            
            cross_entropy_values_2D.append(cross_entropy_values_1D)
        
        return cross_entropy_values_2D
        
    def fit(self):
        n_batches = math.ceil(len(self.x_train) / self.batch_size)

        gpuConfig = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction),
            device_count={'GPU': 1}
        )
        with tf.compat.v1.Session(config=gpuConfig) as self.sess:

            optimizer = SGD(self.cost, self.params, self.learning_rate)
            all_grad = optimizer.get_grad()
            hessian = optimizer.get_hessian(layer_num=-2)
            updates = optimizer.update()
            train = tf.group(*updates)

            self.sess.run(tf.compat.v1.global_variables_initializer())

            costs_train, accs_train, norms_train, hesse_matrixes_train = [], [], [], []
            costs_valid, accs_valid, norms_valid, hesse_matrixes_valid = [], [], [], []

            for self.epoch in range(self.epoch_size):
                self.x_train, self.t_train = shuffle(self.x_train, self.t_train)
                train_feed_dict={self.x: self.x_train[:20000], self.t: self.t_train[:20000], self.is_training: False}
                valid_feed_dict={self.x: self.x_valid, self.t: self.t_valid, self.is_training: False}

                for i in tqdm(range(n_batches)):
                    start = i * self.batch_size
                    end = start + self.batch_size
                    self.sess.run(train, feed_dict={self.x: self.x_train[start:end], 
                                                    self.t: self.t_train[start:end], 
                                                    self.is_training: True})

                norm_train = self.extract_gradient_norm(all_grad, 'Train', train_feed_dict)
                norms_train.append(norm_train)

                norm_valid = self.extract_gradient_norm(all_grad, 'Valid', valid_feed_dict)
                norms_valid.append(norm_valid)

                cost_train, acc_train = self.extract_cost_and_acc('Train', train_feed_dict)
                costs_train.append(cost_train)
                accs_train.append(acc_train)

                cost_valid, acc_valid = self.extract_cost_and_acc('Valid', valid_feed_dict)
                costs_valid.append(cost_valid)
                accs_valid.append(acc_valid)

                hesse_matrix_train = self.sess.run(hessian, feed_dict=train_feed_dict)
                hesse_matrixes_train.append(hesse_matrix_train)

                hesse_matrix_valid = self.sess.run(hessian, feed_dict=valid_feed_dict)
                hesse_matrixes_valid.append(hesse_matrix_valid)
            
            surface_values_train = self.calculate_loss_surface(optimizer, hesse_matrix_train, train_feed_dict)
            surface_values_valid = self.calculate_loss_surface(optimizer, hesse_matrix_valid, valid_feed_dict)

        save_lists = [norms_train, hesse_matrixes_train, costs_train, accs_train, surface_values_train,
                      norms_valid, hesse_matrixes_valid, costs_valid, accs_valid, surface_values_valid]
        reslut_names = ['grad_norm_train', 'hessian_train', 'costs_train', 'accuracy_train', 'surface_values_train',
                        'grad_norm_valid', 'hessian_valid', 'costs_valid', 'accuracy_valid', 'surface_values_valid']
        for (save_list, reslut_name) in zip(save_lists, reslut_names):
            self.save_result(save_list=save_list, reslut_name=reslut_name)

def main():
    batch_size_list = [64, 128, 256, 512, 1024, 2048]
    for batch_size in batch_size_list:
        model = TrainCifar10CNN_SGD(
            learning_rate=0.01,
            momentum=0.9,
            batch_size=batch_size,
            epoch_size=100,
            per_process_gpu_memory_fraction=1.0
        )
        model.fit()
        
if __name__ == "__main__":
    main()