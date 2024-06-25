from deeprig.inits import *
import tensorflow.compat.v1 as tf
import numpy as np
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dot(x, y, sparse=False):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:

        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _normalize(self, inputs, eps):
        raise NotImplementedError

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Encoder(Layer):
    """Encoder layer."""

    def __init__(self, input_dim, output_dim, gene_size, placeholders, dropout, act=tf.nn.relu, featureless=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.act = act
        self.adj = placeholders['adjacency_matrix']
        self.featureless = featureless
        self.dropout = dropout

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weight1'] = glorot([input_dim, output_dim])
            self.vars['weight2'] = glorot([gene_size, output_dim])

        if self.logging:
            self._log_vars()

    def _normalize(self, A, eps = 1e-6):
        n = tf.shape(A)[-1]
        A -= tf.linalg.diag(tf.linalg.diag_part(A))
        A_hat = A + tf.cast(tf.eye(n), dtype=A.dtype)[tf.newaxis, :, :]
        A_hat = tf.cast(A_hat, tf.float64)
        print("Data type of A:", A.dtype)
        deg = tf.reduce_sum(A_hat, axis=2)
        deg = tf.cast(deg, tf.float64)

        D_symm = tf.linalg.diag(1./(eps + tf.math.sqrt(deg)))
        D_asymm = tf.linalg.diag(1./(eps + deg))
        print(D_symm.shape)
        print(A_hat.shape)

        normalize_adj = tf.matmul(tf.matmul(D_symm, A_hat), D_symm)
        normalize_adj = tf.squeeze(normalize_adj)
        print(normalize_adj.shape)

    def _call(self, inputs):
        # convolution
        if not self.featureless:
            x = inputs
            x = tf.nn.dropout(x, 1- self.dropout)
            pre_sup = dot(x, self.vars['weight1'])
        else:
            pre_sup = self.vars['weight1']

        # transform        
        T = dot(self.adj, pre_sup)
        hidden = tf.add(T, self.vars['weight2'])
        return self.act(hidden)


class Decoder(Layer):
    """Decoder layer."""

    def __init__(self, size1, latent_factor_num, placeholders, act=tf.nn.sigmoid, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.size1 = size1
        self.act = act
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weight3'] = glorot([latent_factor_num, latent_factor_num])

    def _call(self, hidden):
        M1 = dot(dot(hidden, self.vars['weight3']), tf.transpose(hidden))
        M1 = tf.reshape(M1, [-1, 1])
        return self.act(M1)
