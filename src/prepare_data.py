import numpy as np
import tensorflow as tf

class PrepareTrainValidData(object):

    def mnist(self):
        mnist = tf.keras.datasets.mnist

        (x_train, t_train), (x_valid, t_valid) = mnist.load_data()

        x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
        x_valid = (x_valid.reshape(-1, 28, 28, 1) / 255).astype(np.float32)

        # x_train = (x_train.reshape(self.input_shape) / 255).astype(np.float32)
        # x_valid = (x_valid.reshape(self.input_shape) / 255).astype(np.float32)

        t_train = np.eye(10)[t_train].astype(np.float32)
        t_valid = np.eye(10)[t_valid].astype(np.float32)

        return x_train, x_valid, t_train, t_valid
    
    def fashion_mnist(self):
        f_mnist = tf.keras.datasets.fashion_mnist

        (x_train, t_train), (x_valid, t_valid) = f_mnist.load_data()

        x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
        x_valid = (x_valid.reshape(-1, 28, 28, 1) / 255).astype(np.float32)

        # x_train = (x_train.reshape(self.input_shape) / 255).astype(np.float32)
        # x_valid = (x_valid.reshape(self.input_shape) / 255).astype(np.float32)

        t_train = np.eye(10)[t_train].astype(np.float32)
        t_valid = np.eye(10)[t_valid].astype(np.float32)

        return x_train, x_valid, t_train, t_valid
    
    def cifar10(self):
        cifar10 = tf.keras.datasets.cifar10

        (x_train, t_train), (x_valid, t_valid) = cifar10.load_data()

        x_train = (x_train.reshape(-1, 32, 32, 3) / 255).astype(np.float32)
        x_valid = (x_valid.reshape(-1, 32, 32, 3) / 255).astype(np.float32)

        # x_train = (x_train.reshape(self.input_shape) / 255).astype(np.float32)
        # x_valid = (x_valid.reshape(self.input_shape) / 255).astype(np.float32)

        t_train = np.eye(10)[t_train].astype(np.float32).reshape((t_train.shape[0], 10))
        t_valid = np.eye(10)[t_valid].astype(np.float32).reshape((t_valid.shape[0], 10))

        print(x_train.shape)
        print(t_train.shape)

        return x_train, x_valid, t_train, t_valid
