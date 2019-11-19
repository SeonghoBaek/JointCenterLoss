import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import util
import layers
import argparse
import csv
import pickle
import shutil


representation_dim = 128
input_width = 96
input_height = 96
scale_size = 112.0
num_channel = 3
num_patch = 4
batch_size = 16
test_size = 100
num_class_per_group = 46
num_epoch = 50

# Network Parameters
g_fc_layer1_dim = 1024
g_fc_layer2_dim = 512  # Final representation
g_fc_layer3_dim = 128

g_dense_block_layers = 4
g_dense_block_depth = 128

lstm_hidden_size_layer1 = 128
lstm_hidden_size_layer2 = 128
lstm_sequence_length = 96
lstm_representation_dim = 64

X = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
Y = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])

with tf.device('/device:CPU:0'):
    ANCHOR = tf.placeholder(tf.float32, [None, 24, 24, 128])

bn_train = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)


def fc_network(x, pretrained=False, weights=None, biases=None, activation='swish', scope='fc_network', bn_phaze=False,
               keep_prob=0.5):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        g_fc_layer1 = layers.fc(x, g_fc_layer1_dim, use_bias=False, scope='g_fc_layer1')
        g_fc_layer1 = layers.batch_norm(g_fc_layer1, bn_phaze, scope='g_fc_layer1_bn')
        g_fc_layer1 = act_func(g_fc_layer1)
        g_fc_layer1 = tf.nn.dropout(g_fc_layer1, keep_prob=keep_prob)

        g_fc_layer2 = layers.fc(g_fc_layer1, g_fc_layer2_dim, use_bias=False, scope='g_fc_layer2')
        g_fc_layer2 = layers.batch_norm(g_fc_layer2, bn_phaze, scope='g_fc_layer2_bn')
        g_fc_layer2 = act_func(g_fc_layer2)
        g_fc_layer2 = tf.nn.dropout(g_fc_layer2, keep_prob=keep_prob)

        g_fc_layer3 = layers.fc(g_fc_layer2, g_fc_layer3_dim, use_bias=False, scope='g_fc_layer3')
        g_fc_layer3 = layers.batch_norm(g_fc_layer3, bn_phaze, scope='g_fc_layer3_bn')
        g_fc_layer3 = act_func(g_fc_layer3)
        g_fc_layer3 = tf.nn.dropout(g_fc_layer3, keep_prob=keep_prob)

        return g_fc_layer3


def lstm_network(input_data, scope='lstm_network', forget_bias=1.0, keep_prob=0.5):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_representation_dim / 2)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)

        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_representation_dim / 2)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

        _, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data, dtype=tf.float32)

        # print(states)
        # print(states[0])
        # print(states[1])

        states_concat = tf.concat([states[0].h, states[1].h], 1)

        print('LSTM Representation Dimension: ' + str(states_concat.get_shape().as_list()))

    return states_concat


def decoder_network(latent, anchor_layer=None, activation='swish', scope='g_decoder_network', bn_phaze=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        #l = tf.cond(bn_phaze, lambda: latent, lambda: make_multi_modal_noise(8))
        l = tf.cond(bn_phaze, lambda: latent, lambda: latent)

        l = layers.fc(l, 6*6*32, non_linear_fn=act_func)

        print('decoder input:', str(latent.get_shape().as_list()))
        l = tf.reshape(l, shape=[-1, 6, 6, 32])

        l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth*4], num_layers=4,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False, scope='block_0')

        print('block 0:', str(l.get_shape().as_list()))

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn1')
        l = act_func(l)

        # 12 x 12
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_deconv1', filter_dims=[3, 3, g_dense_block_depth * 3],
                             stride_dims=[2, 2], padding='SAME', non_linear_fn=None)

        print('deconv1:', str(l.get_shape().as_list()))

        l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth * 3], num_layers=4,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False,
                               scope='block_1', use_dilation=True)

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn2')
        l = act_func(l)

        # 24 x 24
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_deconv2', filter_dims=[3, 3, g_dense_block_depth * 2],
                             stride_dims=[2, 2], padding='SAME', non_linear_fn=None)

        print('deconv2:', str(l.get_shape().as_list()))

        l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth * 2], num_layers=4,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False,
                               scope='block_2', use_dilation=True)

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn3')
        l = act_func(l)

        # 48 x 48
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_deconv3', filter_dims=[3, 3, g_dense_block_depth],
                          stride_dims=[2, 2], padding='SAME', non_linear_fn=None)

        print('deconv3:', str(l.get_shape().as_list()))

        l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=4,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False,
                               scope='block_3', use_dilation=True)

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn4')
        l = act_func(l)

        l = layers.self_attention(l, g_dense_block_depth, act_func=act_func)

        if anchor_layer is not None:
            l = tf.concat([l, anchor_layer], axis=3)

        # 96 x 96
        l = layers.deconv(l, b_size=batch_size, scope='g_dec_deconv4', filter_dims=[3, 3, g_dense_block_depth],
                          stride_dims=[2, 2], padding='SAME', non_linear_fn=None)

        l = add_residual_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False,
                               scope='block_4', use_dilation=True)

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, 3], act_func=act_func,
                                              scope='dense_transition_1', bn_phaze=bn_phaze, use_pool=False)

        l = add_residual_block(l, filter_dims=[3, 3, 3], num_layers=2,
                               act_func=act_func, bn_phaze=bn_phaze, use_residual=False,
                               scope='block_5', use_dilation=True)

        l = tf.nn.tanh(l)

        print('final:', str(l.get_shape().as_list()))

        return l


def add_residual_block(in_layer, filter_dims, num_layers, act_func=tf.nn.relu,
                       bn_phaze=False, use_residual=True, scope='residual_block', use_dilation=False):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = input_dims[-1]

        dilation = [1, 1, 1, 1]

        if use_dilation == True:
            dilation = [1, 2, 2, 1]

        for i in range(num_layers):
            l = layers.add_residual_layer(l, filter_dims=filter_dims, act_func=act_func, bn_phaze=bn_phaze,
                                       scope='layer' + str(i), dilation=dilation)

        if use_residual is True:
            l = tf.add(l, in_layer)

    return l


def add_residual_dense_block(in_layer, filter_dims, num_layers, act_func=tf.nn.relu,
                             bn_phaze=False, scope='residual_dense_block', use_dilation=False):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = filter_dims[-1]

        dilation = [1, 1, 1, 1]

        if use_dilation == True:
            dilation = [1, 2, 2, 1]

        for i in range(num_layers):
            l = layers.add_dense_layer(l, filter_dims=filter_dims, act_func=act_func, bn_phaze=bn_phaze,
                                       scope='layer' + str(i), dilation=dilation)
        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, num_channel_out], act_func=act_func,
                                     scope='dense_transition_1', bn_phaze=bn_phaze, use_pool=False)

        l = tf.add(l, in_layer)

    return l


def encoder_network(x, activation='relu', scope='encoder_network', reuse=False, bn_phaze=False, keep_prob=0.5):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # if reuse:
        #    tf.get_variable_scope().reuse_variables()

        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        # [96 x 96]
        l = layers.conv(x, scope='conv1', filter_dims=[3, 3, g_dense_block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_0')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_1')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_1_1')

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn1')
        l = act_func(l)

        # [48 x 48]
        #l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l = layers.conv(l, scope='conv2', filter_dims=[3, 3, g_dense_block_depth], stride_dims=[2, 2],
                        non_linear_fn=act_func, bias=False, dilation=[1, 1, 1, 1])

        l = layers.self_attention(l, g_dense_block_depth)

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_2')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_3')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth], num_layers=2,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_3_1')

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn2')
        l = act_func(l)

        l_share = l

        # [24 x 24]
        #l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l = layers.conv(l, scope='conv3', filter_dims=[3, 3, g_dense_block_depth * 2], stride_dims=[2, 2],
                        non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth * 2],
                                              act_func=act_func,
                                              scope='dense_transition_24', bn_phaze=bn_phaze,
                                              use_pool=False)

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth * 2], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_4')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth * 2], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_5')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth * 2], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_5_1')

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn3')
        l = act_func(l)

        # [12 x 12]
        #l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l = layers.conv(l, scope='conv4', filter_dims=[3, 3, g_dense_block_depth * 3], stride_dims=[2, 2],
                        non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth * 3],
                                              act_func=act_func,
                                              scope='dense_transition_12', bn_phaze=bn_phaze,
                                              use_pool=False)

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth * 3], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_6')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth * 3], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_7')

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth * 3], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_7_1')

        l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn4')
        l = act_func(l)

        # [6 x 6]
        #l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        l = layers.conv(l, scope='conv5', filter_dims=[3, 3, g_dense_block_depth * 4], stride_dims=[2, 2],
                        non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, g_dense_block_depth * 4],
                                              act_func=act_func,
                                              scope='dense_transition_6', bn_phaze=bn_phaze,
                                              use_pool=False)

        l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth * 4], num_layers=3,
                                     act_func=act_func, bn_phaze=bn_phaze, scope='block_8')

        #l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth * 4], num_layers=3,
        #                             act_func=act_func, bn_phaze=bn_phaze, scope='block_9')

        #l = add_residual_dense_block(l, filter_dims=[3, 3, g_dense_block_depth * 4], num_layers=3,
        #                             act_func=act_func, bn_phaze=bn_phaze, scope='block_10')

        with tf.variable_scope('dense_block_last'):
            scale_layer = layers.add_dense_transition_layer(l, filter_dims=[1, 1, representation_dim],
                                                            act_func=act_func,
                                                            scope='dense_transition_1', bn_phaze=bn_phaze,
                                                            use_pool=False)
            last_dense_layer = layers.add_dense_transition_layer(l, filter_dims=[1, 1, representation_dim],
                                                                 act_func=act_func,
                                                                 scope='dense_transition_2', bn_phaze=bn_phaze,
                                                                 use_pool=False)
            scale_layer = act_func(scale_layer)
            last_dense_layer = act_func(last_dense_layer)

    return last_dense_layer, scale_layer, l_share


def load_images_from_folder(folder):
    images = []

    for filename in os.listdir(folder):
        fullname = os.path.join(folder, filename).replace("\\", "/")
        jpg_img = cv2.imread(fullname)
        img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB)  # To RGB format
        # grey_img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2GRAY)

        if img is not None:
            img = np.array(img)
            # grey_img = np.array(grey_img)

            w = img.shape[1]
            h = img.shape[0]

            if h > w:
                scale = scale_size / w
            else:
                scale = scale_size / h

            # print('w: ' + str(w) + ', h: ' + str(h) + ', scale: ' + str(scale))

            if scale > 1.0:
                img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                # grey_img = cv2.resize(grey_img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                # grey_img = cv2.resize(grey_img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            dy = np.random.random_integers(low=1, high=img.shape[0] - input_height, size=num_patch)
            dx = np.random.random_integers(low=1, high=img.shape[1] - input_width, size=num_patch)

            window = zip(dy, dx)

            for i in range(len(window)):
                croped = img[window[i][0]:window[i][0] + input_height, window[i][1]:window[i][1] + input_width].copy()

                croped = croped / 255.0

                # cv2.imwrite(filename + '_crop_' + str(i) + '.jpg', croped)

                images.append(croped)

    return np.array(images)


def get_residual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(target, value)), [1]))
    elif type == 'l2':
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(target, value)), [1]))

    loss = gamma * loss

    return loss


def make_multi_modal_noise(num_mode=8):
    noise = tf.random_normal(shape=[batch_size, 16], mean=0.0, stddev=1.0, dtype=tf.float32)

    for i in range(num_mode - 1):
        n = tf.random_normal(shape=[batch_size, 16], mean=0.0, stddev=1.0, dtype=tf.float32)
        noise = tf.concat([noise, n], axis=1)

    return noise


def train(model_path):
    trX = []
    trY = []

    dir_list = os.listdir(imgs_dirname)
    dir_list.sort(key=str.lower)

    with tf.device('/device:CPU:0'):
        for idx, labelname in enumerate(dir_list):
            for filename in os.listdir(os.path.join(imgs_dirname, labelname)):

                print(os.path.join(imgs_dirname, labelname) + '/' + filename)

                full_path = os.path.join(imgs_dirname, labelname) + '/' + filename
                jpg_img = cv2.imread(full_path)
                img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB)

                img = cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_AREA)

                sample = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                sample = cv2.resize(sample, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

                cv2.imwrite('tmp.jpg', sample)

                sample = cv2.imread('tmp.jpg')
                sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                sample = cv2.resize(sample, dsize=(0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)

                sample = (sample - 127.5) / 127.5
                trX.append(sample)

                img = (img - 127.5) / 127.5
                trY.append(img)

        trX, trY = shuffle(trX, trY)

        trX = np.array(trX)
        trY = np.array(trY)

        trX = trX.reshape(-1, input_height, input_width, num_channel)
        trY = trY.reshape(-1, input_height, input_width, num_channel)

    # Network setup
    cnn_representation, _, anchor_layer = encoder_network(X, activation='lrelu', bn_phaze=bn_train, scope='encoder')
    print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

    cnn_representation = layers.global_avg_pool(cnn_representation, representation_dim, scope='gap')
    print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

    with tf.device('/device:GPU:1'):
        # decoder_input = make_multi_modal_noise(representation, num_mode=8)
        X_fake = decoder_network(latent=cnn_representation, anchor_layer=anchor_layer, activation='lrelu', scope='decoder',
                                 bn_phaze=bn_train)

    # Trainable variable lists
    encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gap')
    decoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

    generator_vars = encoder_var + decoder_var

    with tf.device('/device:CPU:0'):
        residual_loss = get_residual_loss(Y, X_fake, type='l1', gamma=1.0)

    # training operation
    g_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(residual_loss)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model loaded')
        except:
            print('Start New Training. Wait ...')

        num_itr = 0
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))

        for i in range(num_epoch):
            trX, trY = shuffle(trX, trY)

            for start, end in training_batch:

                _, r, fake = sess.run(
                    [g_optimizer, residual_loss, X_fake],
                    feed_dict={X: trX[start:end], Y: trY[start:end],
                               bn_train: True,
                               keep_prob: 0.5})

                sample = fake[0]
                sample = (sample * 127.5) + 127.5
                sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
                #sample = cv2.resize(sample, dsize=(0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
                #cv2.imwrite('training_sr/tmp.jpg', sample)
                #sample = cv2.imread('training_sr/tmp.jpg')
                #sample = cv2.resize(sample, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                cv2.imwrite('training_sr/sample' + str(num_itr) + '.jpg', sample)

                #sample = trX[start]
                #sample = (sample * 127.5) + 127.5
                #sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
                #cv2.imwrite('training_sr/sample' + str(num_itr) + '_1.jpg', sample)

                num_itr = num_itr + 1

                if num_itr % 10 == 0:
                    print('epoch #' + str(i) + ', itr #' + str(num_itr))
                    print('  - residual loss: ' + str(r))

            try:
                saver.save(sess, model_path)
            except:
                print('Save failed')


def test(model_path, test_image_dir):
    trX = []
    trY = []
    test_output_dir = 'sr'

    if os.path.exists(test_output_dir) == False:
        os.mkdir(test_output_dir)

    with tf.device('/device:CPU:0'):
        test_image_dir_list = os.listdir(test_image_dir)

        for idx, labelname in enumerate(test_image_dir_list):

            if os.path.isdir(os.path.join(test_image_dir, labelname).replace("\\", "/")) is False:
                continue

            if os.path.exists(os.path.join(test_output_dir, labelname)) is False:
                os.mkdir(os.path.join(test_output_dir, labelname))

            for filename in os.listdir(os.path.join(test_image_dir, labelname)):
                full_path = os.path.join(test_image_dir, labelname) + '/' + filename
                jpg_img = cv2.imread(full_path)
                img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB)
                img = (img - 127.5) / 127.5
                trX.append(img)
                trY.append(os.path.join(test_output_dir, labelname))

        trX = np.array(trX)
        trY = np.array(trY)
        trX = trX.reshape(-1, input_height, input_width, num_channel)

        # Network setup
    cnn_representation, _, anchor_layer = encoder_network(X, activation='lrelu', bn_phaze=bn_train, scope='encoder')
    print('CNN Output Tensor Dimension: ' + str(cnn_representation.get_shape().as_list()))

    cnn_representation = layers.global_avg_pool(cnn_representation, representation_dim, scope='gap')
    print('CNN Representation Dimension: ' + str(cnn_representation.get_shape().as_list()))

    with tf.device('/device:GPU:1'):
        # decoder_input = make_multi_modal_noise(representation, num_mode=8)
        X_fake = decoder_network(latent=cnn_representation, anchor_layer=anchor_layer, activation='lrelu',
                                 scope='decoder',
                                 bn_phaze=bn_train)

    # Trainable variable lists
    encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gap')
    decoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

    generator_vars = encoder_var + decoder_var

    with tf.device('/device:CPU:0'):
        residual_loss = get_residual_loss(Y, X_fake, type='l1', gamma=1.0)

    # training operation
    g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(residual_loss)

    # Launch the graph in a session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model loaded')
        except:
            print('Model loading failed')
            return

        i = 0

        for img in trX:
            # latent, anchor = sess.run([latent_real, anchor_layer], feed_dict={X: [img], bn_train: False, keep_prob: 1.0})

            fake = sess.run(
                [X_fake],
                feed_dict={X: [img],
                           bn_train: False,
                           keep_prob: 1.0})

            sample = fake[0][0]
            sample = (sample * 127.5) + 127.5
            # print(sample.shape)
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            #sample = cv2.resize(sample, dsize=(0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite(trY[i] + '/tmp.jpg', sample)
            #sample = cv2.imread(trY[i] + '/tmp.jpg')
            #sample = cv2.resize(sample, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            cv2.imwrite(trY[i] + '/' + str(i) + '.jpg', sample)

            i = i + 1


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test', default='train')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='./model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='input')
    parser.add_argument('--test_data', type=str, help='test data directory', default='test')

    args = parser.parse_args()

    if args.mode == 'train':
        model_path = args.model_path

        imgs_dirname = args.train_data

        num_class_per_group = len(os.listdir(imgs_dirname))
        train(model_path)
    else:
        model_path = args.model_path
        test_data = args.test_data
        batch_size = 1
        test(model_path, test_data)
