import tensorflow as tf
import numpy as np

tf_print = lambda arg, msg: tf.Print(arg,[arg], msg)


def reset_all(seed):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)


def setup_sess():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return sess
    
    
import logging

class VarSet:
    def __init__(self):
        self.weight_loss = None
        self.variables = []
        self.var_list = []
        
    def init_from_vars(self, init_vars):
        weights_num, weights_sum = tf.constant([0.]), tf.constant([0.])
        for w_init, b_init in init_vars:
            w, b = tf.Variable(w_init), tf.Variable(b_init)
            weights_num += tf.to_float(tf.size(w))
            weights_sum += tf.reduce_sum(tf.square(w))
            self.variables.append([w, b])
            self.var_list = self.var_list + [w,b]
        self.weight_loss = weights_sum/weights_num
        return self

    def init_from_dims(self, dim_data, dim_layers):
        weights_num, weights_sum = tf.constant([0.]), tf.constant([0.])
        dim_layer_prev = dim_data
        for dim_layer in dim_layers:
            w, b = VarSet.VarLayer(dim_layer_prev, dim_layer)
            weights_num += tf.to_float(tf.size(w))
            weights_sum += tf.reduce_sum(tf.square(w))
            self.variables.append([w, b])
            self.var_list = self.var_list + [w,b]
            dim_layer_prev = dim_layer
        self.weight_loss = weights_sum/weights_num
        return self
        
    @staticmethod
    def VarLayer(dim_layer_prev, dim_layer):
        w_shape = (dim_layer_prev, dim_layer)
        std_dev = tf.sqrt(3/(dim_layer_prev + dim_layer))
        w_init = tf.truncated_normal(w_shape, stddev=std_dev)
        b_init = tf.zeros([dim_layer])
        return tf.Variable(w_init), tf.Variable(b_init)
    
    def eval_vars(self, sess):
        return tuple(sess.run(lay_vars) for lay_vars in self.variables)


class FullyConnectedNet_Ex:
    def init_from_varset(self, inp_data, varset, activation=tf.nn.relu, last_activation=None, dropout=None):
        self.varset = varset
        data_prev = inp_data
        last_activation = activation if last_activation is None else last_activation
        drop_func = lambda x: x if dropout is None else tf.nn.dropout(x=x,keep_prob=dropout)
        for w, b in varset.variables:
            logits = tf.matmul(data_prev, w) + b
            activn = drop_func(activation(logits))
            data_prev = activn

        self.out_logits, self.out_activn = logits, drop_func(last_activation(logits))
        return self

    def init_from_vars(self, inp_data, init_vars, activation=tf.nn.relu, last_activation=None, dropout=None):
        varset = VarSet().init_from_vars(init_vars)
        return self.init_from_varset(inp_data, varset, activation, last_activation, dropout)
        
    def init_from_dims(self, inp_data, dim_layers, activation=tf.nn.relu, last_activation=None, dropout=None):
        dim_data = inp_data.get_shape().as_list()[1]
        varset = VarSet().init_from_dims(dim_data, dim_layers)
        return self.init_from_varset(inp_data, varset, activation, last_activation, dropout)

    @property
    def eval_vars(self): return self.varset.eval_vars
    @property
    def var_list(self): return self.varset.var_list
    @property
    def weight_loss(self): return self.varset.weight_loss
    
    
class FullyConnectedNet(FullyConnectedNet_Ex):
    def __init__(self, inp_data, dim_layers, activation=tf.nn.relu, last_activation=None, init_vars=None):
        if init_vars:
            self.init_from_vars(inp_data, init_vars, activation, last_activation)
        else:
            self.init_from_dims(inp_data, dim_layers, activation, last_activation)

    
def pdist2(X, Y=None, method=2): # dimensions should be, X: NX x C and Y: NY x C
    if method==1:
        Y = X if Y is None else Y
        NY = tf.shape(Y)[0]
        X_ = tf.expand_dims(X, 1)
        m = tf.tile(X_, [1, NY, 1]) - Y
        dists2 = tf.reduce_sum(tf.square(m), 2)
    else:
        # X2 = sum(X.^2,1); U = repmat(X2,N,1) + repmat(X2',1,N) - 2*(X'*X);
        X2 = tf.reduce_sum(tf.square(X),1)
        Y2 = X2 if Y is None else tf.reduce_sum(tf.square(Y),1)
        X_ = tf.expand_dims(X2, 1)
        Y_ = tf.expand_dims(Y2, 0)
        NX, NY = tf.shape(X2)[0], tf.shape(Y2)[0]
        X_T = tf.tile(X_, [1, NY])
        Y_T = tf.tile(Y_, [NX, 1])
        dists2 = X_T + Y_T - 2 * tf.matmul(X,tf.transpose(X if Y is None else Y))
    return dists2

    
def calc_laplacian(vals, sigma2):
    dists2 = pdist2(vals)
    W = tf.exp(-dists2/sigma2)
    d = tf.reduce_sum(W, 1)
    D = tf.diag(d)
    return D - W

    
def test_pdist2():
    tf.reset_default_graph()
    a = tf.constant([[0, 2, 3], [2, 2, 2]])
    b = tf.constant([[4, 2, 2], [2, 2, 2], [2, 2, 6], [3, 2, 2]])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print(sess.run(pdist2(a,b,method=1)))
    print(sess.run(pdist2(a,b,method=2)))
    print(sess.run(pdist2(b,method=1)))
    print(sess.run(pdist2(b,method=2)))
