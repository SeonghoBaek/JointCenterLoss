import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import util
import argparse
import os
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_feature_dim = 97
cond_step_dim = 8
cond_wafer_dim = 24
cond_dim = cond_step_dim + cond_wafer_dim

lstm_sequence_length = 20
lstm_hidden_size_layer1 = 64
lstm_hidden_size_layer2 = 64
lstm_feature_dim = lstm_hidden_size_layer1
lstm_z_sequence_dim = 16
lstm_linear_transform_input_dim = 2 * lstm_feature_dim

g_encoder_z_local_dim = 16
g_encoder_z_dim = lstm_z_sequence_dim + g_encoder_z_local_dim + cond_dim
g_encoder_input_dim = input_feature_dim
g_encoder_layer1_dim = 84
g_encoder_layer2_dim = 64
g_encoder_layer3_dim = 32

g_decoder_output_dim = input_feature_dim
g_decoder_layer2_dim = 72
g_decoder_layer1_dim = 84

d_layer_1_dim = input_feature_dim
d_layer_2_dim = 64
d_layer_3_dim = 32
d_layer_4_dim = 16

num_block_layers = 3
dense_layer_depth = 16


def lstm_network(input, scope='lstm_network'):
    with tf.variable_scope(scope):
        # tf.nn.rnn_cell
        lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size_layer1, forget_bias=1.0)
        lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size_layer2, forget_bias=1.0)

        lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell1, lstm_cell2], state_is_tuple=True)

        # tf.nn.rnn_cell
        # lstm_cell1 = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size_layer1, forget_bias=1.0)
        # lstm_cell2 = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size_layer2, forget_bias=1.0)

        #lstm_cells = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell1, lstm_cell2], state_is_tuple=True)

        # initial_state = lstm_cells.zero_state(batch_size,  tf.float32)

        _, states = tf.nn.dynamic_rnn(lstm_cells, input, dtype=tf.float32, initial_state=None)

        # z_sequence_output = states[1].h
        # print(z_sequence_output.get_shape())
        states_concat = tf.concat([states[0].h, states[1].h], 1)

        #def fc(input, scope, out_dim, non_linear_fn=None, initial_value=None, use_bias=True):
        z_sequence_output = fc(states_concat, lstm_z_sequence_dim, scope='linear_transform')

    return z_sequence_output


def fc(input_data, out_dim, non_linear_fn=None, initial_value=None, use_bias=True, scope='fc'):
    with tf.variable_scope(scope):
        input_dims = input_data.get_shape().as_list()

        if len(input_dims) == 4:
            _, input_h, input_w, num_channels = input_dims
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input_data, [-1, in_dim])
        else:
            in_dim = input_dims[-1]
            flat_input = input_data

        if initial_value is None:
            fc_weight = tf.get_variable("weights", shape=[in_dim, out_dim], initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
            fc_bias = tf.get_variable("bias", shape=[out_dim], initializer=tf.constant_initializer(0.0))
        else:
            fc_weight = tf.get_variable("weights", initializer=initial_value[0])
            fc_bias = tf.get_variable("bias", shape=[out_dim], initializer=initial_value[1])

        if use_bias:
            output = tf.add(tf.matmul(flat_input, fc_weight), fc_bias)
        else:
            output = tf.matmul(flat_input, fc_weight)

        if non_linear_fn is None:
            return output
        else:
            activation = non_linear_fn(output)

        return activation


def batch_norm(x, b_train, scope, reuse=False):
    with tf.variable_scope(scope,  reuse=tf.AUTO_REUSE):
        n_out = x.get_shape().as_list()[-1]

        beta = tf.get_variable('beta', initializer=tf.constant(0.0, shape=[n_out]))
        gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, shape=[n_out]))

        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(b_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

        return normed


def conv(input, scope, filter_dims, stride_dims, padding='SAME',
         non_linear_fn=tf.nn.relu, dilation=[1, 1, 1, 1], bias=False, sn=False):
    input_dims = input.get_shape().as_list()

    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in
    assert (len(filter_dims) == 3)  # height, width and num_channels out
    assert (len(stride_dims) == 2)  # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(scope):

        conv_weight = tf.get_variable('conv_weight',
                                      shape=[filter_h, filter_w, num_channels_in, num_channels_out],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        if bias is True:
            conv_bias = tf.get_variable('conv_bias', shape=[num_channels_out],
                                        initializer=tf.zeros_initializer)

        #conv_weight = tf.Variable(
        #    tf.truncated_normal([filter_h, filter_w, num_channels_in, num_channels_out], stddev=0.1, dtype=tf.float32))

        #if bias is True:
        #    conv_bias = tf.Variable(tf.zeros([num_channels_out], dtype=tf.float32))

        conv_filter = conv_weight

        if sn == True:
            conv_filter = spectral_norm(conv_weight, scope='sn')

        map = tf.nn.conv2d(input, filter=conv_filter, strides=[1, stride_h, stride_w, 1], padding=padding, dilations=dilation)

        if bias is True:
            map = tf.nn.bias_add(map, conv_bias)

        if non_linear_fn is not None:
            activation = non_linear_fn(map)
        else:
            activation = map

        # print(activation.get_shape().as_list())
        return activation


def batch_norm_conv(x, b_train, scope):
    with tf.variable_scope(scope,  reuse=tf.AUTO_REUSE):
        n_out = x.get_shape().as_list()[-1]

        beta = tf.get_variable('beta', initializer=tf.constant(0.0, shape=[n_out]))
        gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, shape=[n_out]))

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(b_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

        return normed


def add_dense_layer(layer, filter_dims, act_func=tf.nn.relu, scope='dense_layer',
                    use_bn=True, bn_phaze=False, use_bias=False, dilation=[1, 1, 1, 1], sn=False):
    with tf.variable_scope(scope):
        l = layer

        if use_bn:
            l = batch_norm_conv(l, b_train=bn_phaze, scope='bn')

        l = act_func(l)
        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=[1, 1], dilation=dilation,
                 non_linear_fn=None, bias=use_bias, sn=sn)
        l = tf.concat([l, layer], 3)

    return l


def add_residual_layer(layer, filter_dims, act_func=tf.nn.relu, scope='residual_layer',
                       use_bn=True, bn_phaze=False, use_bias=False, dilation=[1, 1, 1, 1], sn=False):
    with tf.variable_scope(scope):
        l = layer

        if use_bn:
            l = batch_norm_conv(l, b_train=bn_phaze, scope='bn')

        l = act_func(l)
        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=[1, 1],
                 dilation=dilation, non_linear_fn=None, bias=use_bias, sn=sn)

    return l


def add_dense_transition_layer(layer, filter_dims, stride_dims=[1, 1], act_func=tf.nn.relu, scope='transition',
                               use_bn=True, bn_phaze=False, use_pool=True, use_bias=False, sn=False):
    with tf.variable_scope(scope):
        if use_bn:
            l = batch_norm_conv(layer, b_train=bn_phaze, scope='bn')
            l = act_func(l)
        else:
            l = act_func(layer)

        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=stride_dims,
                 non_linear_fn=None, bias=use_bias, sn=sn)

        if use_pool:
            l = tf.nn.max_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return l


def global_avg_pool(input_data, output_length=1, padding='VALID', use_bias=False, scope='gloval_avg_pool'):
    input_dims = input_data.get_shape().as_list()

    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in

    num_channels_in = input_dims[-1]
    height = input_dims[1]
    width = input_dims[2]

    with tf.variable_scope(scope):
        if output_length == 1:
            pool = tf.nn.avg_pool(input_data, [1, height, width, 1], strides=[1, 1, 1, 1], padding=padding)
            pool = tf.reduce_mean(pool, axis=[1, 2])
            pool = tf.squeeze(pool, axis=[1, 2])

            return pool
        else:
            if num_channels_in != output_length:
                conv_weight = tf.get_variable('gap_weight', shape=[1, 1, num_channels_in, output_length],
                                              initializer=tf.truncated_normal_initializer(stddev=1.0))

                conv = tf.nn.conv2d(input_data, conv_weight, strides=[1, 1, 1, 1], padding='SAME')

                if use_bias == True:
                    conv_bias = tf.get_variable('gap_bias', shape=[output_length], initializer=tf.zeros_initializer)
                    conv = tf.nn.bias_add(conv, conv_bias)

                pool = tf.nn.avg_pool(conv, ksize=[1, height, width, 1], strides=[1, 1, 1, 1], padding=padding)
            else:
                pool = tf.nn.avg_pool(input_data, ksize=[1, height, width, 1], strides=[1, 1, 1, 1], padding=padding)
            pool = tf.squeeze(pool, axis=[1, 2])

            return pool


def avg_pool(input, filter_dims, stride_dims, padding='SAME', scope='avgpool'):
    assert (len(filter_dims) == 2)  # filter height and width
    assert (len(stride_dims) == 2)  # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(scope):
        pool = tf.nn.avg_pool(input, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1],
                              padding=padding)

        return pool


def get_deconv2d_output_dims(input_dims, filter_dims, stride_dims, padding):
    batch_size, input_h, input_w, num_channels_in = input_dims
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    if padding == 'SAME':
        out_h = input_h * stride_h
    elif padding == 'VALID':
        out_h = (input_h - 1) * stride_h + filter_h

    if padding == 'SAME':
        out_w = input_w * stride_w
    elif padding == 'VALID':
        out_w = (input_w - 1) * stride_w + filter_w

    return [batch_size, out_h, out_w, num_channels_out]


def deconv(input_data, b_size, scope, filter_dims, stride_dims, padding='SAME', non_linear_fn=tf.nn.relu, sn=False):
    input_dims = input_data.get_shape().as_list()
    # print(scope, 'in', input_dims)
    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in
    assert (len(filter_dims) == 3)  # height, width and num_channels out
    assert (len(stride_dims) == 2)  # stride height and width

    input_dims = [b_size, input_dims[1], input_dims[2], input_dims[3]]
    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    output_dims = get_deconv2d_output_dims(input_dims,
                                           filter_dims,
                                           stride_dims,
                                           padding)

    with tf.variable_scope(scope):
        deconv_weight = tf.get_variable('deconv_weight', shape=[filter_h, filter_w, num_channels_out, num_channels_in],
                                       initializer=tf.random_normal_initializer(stddev=0.1))

        deconv_bias = tf.get_variable('deconv_bias', shape=[num_channels_out], initializer=tf.zeros_initializer)

        conv_filter = deconv_weight

        if sn == True:
            conv_filter = spectral_norm(deconv_weight, scope='deconv_sn')

        map = tf.nn.conv2d_transpose(input_data, conv_filter, output_dims, strides=[1, stride_h, stride_w, 1],
                                     padding=padding)

        map = tf.nn.bias_add(map, deconv_bias)

        activation = non_linear_fn(map)

        # print(scope, 'out', activation.get_shape().as_list())
        return activation


def self_attention(x, channels, act_func=tf.nn.relu, scope='attention'):
    with tf.variable_scope(scope):
        batch_size, height, width, num_channels = x.get_shape().as_list()

        f = conv(x, scope='f_conv', filter_dims=[1, 1, channels // 8], stride_dims=[1, 1], non_linear_fn=act_func)
        f = tf.layers.max_pooling2d(f, pool_size=2, strides=2, padding='SAME')

        print('attention f dims: ' + str(f.get_shape().as_list()))

        g = conv(x, scope='g_conv', filter_dims=[1, 1, channels // 8], stride_dims=[1, 1], non_linear_fn=act_func)

        print('attention g dims: ' + str(g.get_shape().as_list()))

        h = conv(x, scope='h_conv', filter_dims=[1, 1, channels // 2], stride_dims=[1, 1], non_linear_fn=act_func)
        h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='SAME')

        print('attention h dims: ' + str(h.get_shape().as_list()))

        # N = h * w
        g = tf.reshape(g, shape=[-1, g.shape[1] * g.shape[2], g.get_shape().as_list()[-1]])

        print('attention g flat dims: ' + str(g.get_shape().as_list()))

        f = tf.reshape(f, shape=[-1, f.shape[1] * f.shape[2], f.shape[-1]])

        print('attention f flat dims: ' + str(f.get_shape().as_list()))

        s = tf.matmul(g, f, transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        print('attention beta dims: ' + str(s.get_shape().as_list()))

        h = tf.reshape(h, shape=[-1, h.shape[1] * h.shape[2], h.shape[-1]])

        print('attention h flat dims: ' + str(h.get_shape().as_list()))

        o = tf.matmul(beta, h)  # [bs, N, C]

        print('attention o dims: ' + str(o.get_shape().as_list()))

        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[-1, height, width, num_channels // 2])  # [bs, h, w, C]
        o = conv(o, scope='attn_conv', filter_dims=[1, 1, channels], stride_dims=[1, 1], non_linear_fn=act_func)
        x = gamma * o + x

    return x


def spectral_norm(w, iteration=1, scope='sn'):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    with tf.variable_scope(scope):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None

        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
