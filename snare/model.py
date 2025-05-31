# snare/model.py

import tensorflow as tf
from tensorflow.keras import Model, layers

MAXSEQ = 4980
NUM_FEATURE = 20
NUM_CLASSES = 2

class DeepScan(Model):
    def __init__(self,
                 input_shape=(1, MAXSEQ, NUM_FEATURE),
                 window_sizes=[8, 16, 24, 32, 40, 48],
                 num_filters=256,
                 num_hidden=128):
        super(DeepScan, self).__init__()
        self.window_sizes = window_sizes
        self.conv2d = []
        self.maxpool = []
        self.flatten = []

        for window_size in window_sizes:
            self.conv2d.append(layers.Conv2D(
                filters=num_filters,
                kernel_size=(1, window_size),
                activation='relu',
                padding='valid',
                kernel_initializer='glorot_uniform'
            ))
            self.maxpool.append(layers.MaxPooling2D(
                pool_size=(1, MAXSEQ - window_size + 1),
                strides=(1, MAXSEQ),
                padding='valid'))
            self.flatten.append(layers.Flatten())

        self.dropout = layers.Dropout(rate=0.7)
        self.fc1 = layers.Dense(num_hidden, activation='relu')
        self.fc2 = layers.Dense(NUM_CLASSES, activation='softmax',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, x, training=False):
        features = []
        for i in range(len(self.window_sizes)):
            x_conv = self.conv2d[i](x)
            x_pool = self.maxpool[i](x_conv)
            x_flat = self.flatten[i](x_pool)
            features.append(x_flat)

        x = tf.concat(features, axis=1)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        return self.fc2(x)