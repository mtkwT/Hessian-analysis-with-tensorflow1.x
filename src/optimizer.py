import tensorflow as tf

class BaseOptimizer(object):
    def __init__(self, cost, params):
        self.cost = cost
        self.params = params
        self.grads = tf.gradients(cost, params)

    def get_grad(self):
        all_grad = [] 
        for grad in self.grads:
            all_grad.append(tf.keras.backend.flatten(grad))
        return all_grad

    def get_hessian(self, layer_num=-2): 
        self.hessian = tf.hessians(self.cost, self.params[layer_num])[0]
        shape = (self.params[layer_num].shape[0] * self.params[layer_num].shape[1],
                 self.params[layer_num].shape[0] * self.params[layer_num].shape[1])
        self.hessian = tf.reshape(self.hessian, shape=shape)
        return self.hessian
    
    def add_eigvec_update(self, c1, v1, c2, v2, layer_num=-2): 
        self.updates = []
        with tf.control_dependencies(self.updates):
            self.updates.append(self.params[layer_num].assign_add(c1*v1+c2*v2)) 
        return self.updates

    def sub_eigvec_update(self, c1, v1, c2, v2, layer_num=-2): 
        self.updates = []
        with tf.control_dependencies(self.updates):
            self.updates.append(self.params[layer_num].assign_sub(c1*v1+c2*v2))
        return self.updates

class SGD(BaseOptimizer):
    def __init__(self, cost, params, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.cost = cost
        self.params = params
        self.grads = tf.gradients(cost, params)
    
    def update(self):
        self.updates = []
        for param, grad in zip(self.params, self.grads):        
            v = tf.Variable(tf.zeros_like(param, dtype=tf.float32), name='v')
            self.updates.append(v.assign(self.momentum * v + self.lr * grad))
            with tf.control_dependencies(self.updates):
                self.updates.append(param.assign_sub(v))
        return self.updates

class Adagrad(BaseOptimizer):
    def __init__(self, cost, params, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.cost = cost
        self.params = params
        self.grads = tf.gradients(cost, params)

    def update(self):
        self.updates = []
        for param, grad in zip(self.params, self.grads):
            G = tf.Variable(tf.zeros_like(param, dtype=tf.float32), name='G')
            self.updates.append(G.assign_add(grad**2))

            with tf.control_dependencies(self.updates):
                self.updates.append(param.assign_sub(self.lr / tf.sqrt(G + self.eps) * grad))

        return self.updates
        
class RMSprop(BaseOptimizer):
    def __init__(self, cost, params, lr=0.001, rho=0.9, eps=1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.cost = cost
        self.params = params
        self.grads = tf.gradients(cost, params)

    def update(self):
        self.updates = []
        for param, grad in zip(self.params, self.grads):
            G = tf.Variable(tf.zeros_like(param, dtype=tf.float32), name='G')
            self.updates.append(G.assign(self.rho * G + (1 - self.rho) * grad**2))

            with tf.control_dependencies(self.updates):
                self.updates.append(param.assign_sub(self.lr / tf.sqrt(G + self.eps) * grad))
        
        return self.updates

class Adam(BaseOptimizer):
    def __init__(self, cost, params, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.cost = cost
        self.params = params
        self.grads = tf.gradients(cost, params)

    def update(self):
        self.updates = []
        for param, grad in zip(self.params, self.grads):
            m = tf.Variable(tf.zeros_like(param, dtype=tf.float32), name='m')
            v = tf.Variable(tf.zeros_like(param, dtype=tf.float32), name='v')
            self.updates.append(m.assign(self.beta_1 * m + (1 - self.beta_1) * grad))
            self.updates.append(v.assign(self.beta_2 * v + (1 - self.beta_2) * grad**2))

            with tf.control_dependencies(self.updates):
                self.updates.append(param.assign_sub(self.lr * m / tf.sqrt(v + self.eps)))
        
        return self.updates