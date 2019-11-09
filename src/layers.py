import tensorflow as tf
import numpy as np

SEED = np.random.randint(1000)
rng = np.random.RandomState(SEED)

class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        # He Initialization
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/in_dim),
                        high=np.sqrt(6/in_dim),
                        size=(in_dim, out_dim)
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

        self.params = [self.W, self.b]

    def __call__(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)

class Dropout:
    def __init__(self, is_training, dropout_keep_prob=1.0):
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.params = []

    def __call__(self, x):
        return tf.cond(
            pred=self.is_training,
            true_fn=lambda: tf.nn.dropout(x, keep_prob=self.dropout_keep_prob),
            false_fn=lambda: x
        )

class BatchNorm:
    def __init__(self, is_training):
        self.is_training = is_training
        self.params = []

    def __call__(self, x):
        return tf.cond(
            pred=self.is_training,
            true_fn=lambda:tf.nn.batch_normalization(x, 0, 1, 0, 1, 1e-8),
            false_fn=lambda: x
        )

class Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1,1,1,1], padding='VALID'):
        fan_in = np.prod(filter_shape[:3]) 
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/fan_in),
                        high=np.sqrt(6/fan_in),
                        size=filter_shape
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b')
        self.function = function
        self.strides = strides
        self.padding = padding
        self.params = [self.W, self.b]

    def __call__(self, x):
        u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding) + self.b
        return self.function(u)

class Pooling:
    def __init__(self, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.params = []
    
    def __call__(self, x):
        return tf.nn.max_pool2d(x, ksize=self.ksize, strides=self.strides, padding=self.padding)

class Flatten:
    def __init__(self):
        self.params = []     
    def __call__(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

def get_params(layers):
        params_all = []
        for layer in layers:
            params = layer.params
            params_all.extend(params)
        return params_all

def f_props(layers, h):
    for layer in layers:
        h = layer(h)
    return h