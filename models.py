"""
Copyright 2017-2019 Pandora Media, Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import tensorflow as tf

'''
models.py: in this script some tensorflow models are build.

See build_model() for an example showing how to use these functions.
'''


def wave_frontend(x, is_training):
    '''Function implementing the front-end proposed by Lee et al. 2017.
       Lee, et al. "Sample-level Deep Convolutional Neural Networks for Music
       Auto-tagging Using Raw Waveforms."
       arXiv preprint arXiv:1703.01789 (2017).

    - 'x': placeholder whith the input.
    - 'is_training': placeholder indicating weather it is training or test
    phase, for dropout or batch norm.
    '''
    initializer = tf.contrib.layers.variance_scaling_initializer()
    conv0 = tf.layers.conv1d(inputs=x,
                             filters=64,
                             kernel_size=3,
                             strides=3,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv0 = tf.layers.batch_normalization(conv0, training=is_training)

    conv1 = tf.layers.conv1d(inputs=bn_conv0,
                             filters=64,
                             kernel_size=3,
                             strides=1,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    pool_1 = tf.layers.max_pooling1d(bn_conv1, pool_size=3, strides=3)

    conv2 = tf.layers.conv1d(inputs=pool_1,
                             filters=64,
                             kernel_size=3,
                             strides=1,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    pool_2 = tf.layers.max_pooling1d(bn_conv2, pool_size=3, strides=3)

    conv3 = tf.layers.conv1d(inputs=pool_2,
                             filters=128,
                             kernel_size=3,
                             strides=1,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    pool_3 = tf.layers.max_pooling1d(bn_conv3, pool_size=3, strides=3)

    conv4 = tf.layers.conv1d(inputs=pool_3,
                             filters=128,
                             kernel_size=3,
                             strides=1,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv4 = tf.layers.batch_normalization(conv4, training=is_training)
    pool_4 = tf.layers.max_pooling1d(bn_conv4, pool_size=3, strides=3)

    conv5 = tf.layers.conv1d(inputs=pool_4,
                             filters=128,
                             kernel_size=3,
                             strides=1,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv5 = tf.layers.batch_normalization(conv5, training=is_training)
    pool_5 = tf.layers.max_pooling1d(bn_conv5, pool_size=3, strides=3)

    conv6 = tf.layers.conv1d(inputs=pool_5,
                             filters=256,
                             kernel_size=3,
                             strides=1,
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv6 = tf.layers.batch_normalization(conv6, training=is_training)
    pool_6 = tf.layers.max_pooling1d(bn_conv6, pool_size=3, strides=3)

    return tf.expand_dims(pool_6, [3])


def spec_frontend(x, is_training, config, num_filt):
    '''Function implementing the proposed spectrogram front-end.

    - 'route_out': is the output of the front-end, and therefore the input of
        this function.
    - 'is_training': placeholder indicating weather it is training or test
        phase, for dropout or batch norm.
    - 'config': dictionary with some configurable parameters like: number of
        output units - config['numOutputNeurons'] or number of frequency bins
        of the spectrogram config['setup_params']['yInput']
    - 'num_filt': multiplicative factor that controls the number of filters
        for every filter shape.
    '''
    initializer = tf.contrib.layers.variance_scaling_initializer()
    y_input = config['setup_params']['yInput']
    input_layer = tf.expand_dims(x, 3)

    # padding only time domain for an efficient 'same' implementation
    # (since we pool throughout all frequency afterwards)
    input_pad_7 = tf.pad(input_layer,
                         [[0, 0], [3, 3], [0, 0], [0, 0]],
                         "CONSTANT")
    input_pad_3 = tf.pad(input_layer,
                         [[0, 0], [1, 1], [0, 0], [0, 0]],
                         "CONSTANT")

    # [TIMBRE] filter shape 1: 7x0.9f
    conv1 = tf.layers.conv2d(inputs=input_pad_7,
                             filters=num_filt,
                             kernel_size=[7, int(0.9 * y_input)],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    pool1 = tf.layers.max_pooling2d(inputs=bn_conv1,
                                    pool_size=[1, bn_conv1.shape[2]],
                                    strides=[1, bn_conv1.shape[2]])
    p1 = tf.squeeze(pool1, [2])

    # [TIMBRE] filter shape 2: 3x0.9f
    conv2 = tf.layers.conv2d(inputs=input_pad_3,
                             filters=num_filt*2,
                             kernel_size=[3, int(0.9 * y_input)],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    pool2 = tf.layers.max_pooling2d(inputs=bn_conv2,
                                    pool_size=[1, bn_conv2.shape[2]],
                                    strides=[1, bn_conv2.shape[2]])
    p2 = tf.squeeze(pool2, [2])

    # [TIMBRE] filter shape 3: 1x0.9f
    conv3 = tf.layers.conv2d(inputs=input_layer,
                             filters=num_filt*4,
                             kernel_size=[1, int(0.9 * y_input)],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    pool3 = tf.layers.max_pooling2d(inputs=bn_conv3,
                                    pool_size=[1, bn_conv3.shape[2]],
                                    strides=[1, bn_conv3.shape[2]])
    p3 = tf.squeeze(pool3, [2])

    # [TIMBRE] filter shape 3: 7x0.4f
    conv4 = tf.layers.conv2d(inputs=input_pad_7,
                             filters=num_filt,
                             kernel_size=[7, int(0.4 * y_input)],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv4 = tf.layers.batch_normalization(conv4, training=is_training)
    pool4 = tf.layers.max_pooling2d(inputs=bn_conv4,
                                    pool_size=[1, bn_conv4.shape[2]],
                                    strides=[1, bn_conv4.shape[2]])
    p4 = tf.squeeze(pool4, [2])

    # [TIMBRE] filter shape 5: 3x0.4f
    conv5 = tf.layers.conv2d(inputs=input_pad_3,
                             filters=num_filt * 2,
                             kernel_size=[3, int(0.4 * y_input)],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv5 = tf.layers.batch_normalization(conv5, training=is_training)
    pool5 = tf.layers.max_pooling2d(inputs=bn_conv5,
                                    pool_size=[1, bn_conv5.shape[2]],
                                    strides=[1, bn_conv5.shape[2]])
    p5 = tf.squeeze(pool5, [2])

    # [TIMBRE] filter shape 6: 1x0.4f
    conv6 = tf.layers.conv2d(inputs=input_layer,
                             filters=num_filt * 4,
                             kernel_size=[1, int(0.4 * y_input)],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv6 = tf.layers.batch_normalization(conv6, training=is_training)
    pool6 = tf.layers.max_pooling2d(inputs=bn_conv6,
                                    pool_size=[1, bn_conv6.shape[2]],
                                    strides=[1, bn_conv6.shape[2]])
    p6 = tf.squeeze(pool6, [2])

    # [TEMPORAL-FEATURES] - average pooling + filter shape 7: 165x1
    pool7 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, y_input],
                                        strides=[1, y_input])
    pool7_rs = tf.squeeze(pool7, [3])
    conv7 = tf.layers.conv1d(inputs=pool7_rs,
                             filters=num_filt,
                             kernel_size=165,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv7 = tf.layers.batch_normalization(conv7, training=is_training)

    # [TEMPORAL-FEATURES] - average pooling + filter shape 8: 128x1
    pool8 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, y_input],
                                        strides=[1, y_input])
    pool8_rs = tf.squeeze(pool8, [3])
    conv8 = tf.layers.conv1d(inputs=pool8_rs,
                             filters=num_filt*2,
                             kernel_size=128,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv8 = tf.layers.batch_normalization(conv8, training=is_training)

    # [TEMPORAL-FEATURES] - average pooling + filter shape 9: 64x1
    pool9 = tf.layers.average_pooling2d(inputs=input_layer,
                                        pool_size=[1, y_input],
                                        strides=[1, y_input])
    pool9_rs = tf.squeeze(pool9, [3])
    conv9 = tf.layers.conv1d(inputs=pool9_rs,
                             filters=num_filt*4,
                             kernel_size=64,
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)
    bn_conv9 = tf.layers.batch_normalization(conv9, training=is_training)

    # [TEMPORAL-FEATURES] - average pooling + filter shape 10: 32x1
    pool10 = tf.layers.average_pooling2d(inputs=input_layer,
                                         pool_size=[1, y_input],
                                         strides=[1, y_input])
    pool10_rs = tf.squeeze(pool10, [3])
    conv10 = tf.layers.conv1d(inputs=pool10_rs,
                              filters=num_filt*8,
                              kernel_size=32,
                              padding="same",
                              activation=tf.nn.relu,
                              kernel_initializer=initializer)
    bn_conv10 = tf.layers.batch_normalization(conv10, training=is_training)

    # concatenate all feature maps
    pool = tf.concat([p1, p2, p3, p4, p5, p6, bn_conv7, bn_conv8, bn_conv9,
                      bn_conv10], 2)
    return tf.expand_dims(pool, 3)


def backend(route_out, is_training, config, num_units):
    '''Function implementing the proposed back-end.

    - 'route_out': is the output of the front-end, and therefore the input of
        this function.
    - 'is_training': placeholder indicating weather it is training or test
        phase, for dropout or batch norm.
    - 'config': dictionary with some configurable parameters like: number of
        output units - config['numOutputNeurons'] or number of frequency bins
        of the spectrogram config['setup_params']['yInput']
    - 'num_units': number of units/neurons of the output dense layer.
    '''

    # conv layer 1 - adapting dimensions
    conv1 = tf.layers.conv2d(inputs=route_out,
                             filters=512,
                             kernel_size=[7, route_out.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             name='1cnnOut',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    bn_conv1_t = tf.transpose(bn_conv1, [0, 1, 3, 2])

    # conv layer 2 - residual connection
    bn_conv1_pad = tf.pad(bn_conv1_t, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv2 = tf.layers.conv2d(inputs=bn_conv1_pad,
                             filters=512,
                             kernel_size=[7, bn_conv1_pad.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             name='2cnnOut',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    conv2_t = tf.transpose(conv2, [0, 1, 3, 2])
    bn_conv2 = tf.layers.batch_normalization(conv2_t, training=is_training)
    res_conv2 = tf.add(bn_conv2, bn_conv1_t)

    # temporal pooling
    pool1 = tf.layers.max_pooling2d(inputs=res_conv2, pool_size=[2, 1], strides=[2, 1], name='poolOut')

    # conv layer 3 - residual connection
    bn_conv4_pad = tf.pad(pool1, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv5 = tf.layers.conv2d(inputs=bn_conv4_pad,
                             filters=512,
                             kernel_size=[7, bn_conv4_pad.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             name='3cnnOut',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    conv5_t = tf.transpose(conv5, [0, 1, 3, 2])
    bn_conv5 = tf.layers.batch_normalization(conv5_t, training=is_training)
    res_conv5 = tf.add(bn_conv5, pool1)

    # global pooling: max and average
    max_pool2 = tf.reduce_max(res_conv5, axis=1)
    avg_pool2, var_pool2 = tf.nn.moments(res_conv5, axes=[1])
    pool2 = tf.concat([max_pool2, avg_pool2], 2)
    flat_pool2 = tf.contrib.layers.flatten(pool2)

    # output - 1 dense layer with droupout
    flat_pool2_dropout = tf.layers.dropout(flat_pool2, rate=0.5, training=is_training)
    dense = tf.layers.dense(inputs=flat_pool2_dropout,
                            units=num_units,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_dense = tf.layers.batch_normalization(dense, training=is_training)
    dense_dropout = tf.layers.dropout(bn_dense, rate=0.5, training=is_training)
    return tf.layers.dense(inputs=dense_dropout,
                           activation=tf.sigmoid,
                           units=config['numOutputNeurons'],
                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer())


def build_model(x, is_training, config):
    '''Function implementing an example of how to build a model with the functions above.

    - 'x': placeholder whith the input.
    - 'is_training': placeholder indicating weather it is training or test phase, for dropout or batch norm.
    - 'config': dictionary with some configurable parameters like: number of output units - config['numOutputNeurons']
                or number of frequency bins of the spectrogram config['setup_params']['yInput']
    '''
    # The following line builds the model that achieved better results in our experiments.
    # It is based on a spectrogram front-end (num_filters=16) with 500 output units in the dense layer.
    return backend(spec_frontend(x, is_training, config, 16), is_training, config, 500)
