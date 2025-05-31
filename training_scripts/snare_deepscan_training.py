#!/usr/bin/env python
# coding: utf-8

# === CONFIGURATION ===
MAXSEQ      = 4980
NUM_FEATURE = 20
BATCH_SIZE  = 128
NUM_CLASSES = 2
CLASS_NAMES = ['SNARE', 'NON-SNARE']
EPOCHS      = 10
NUM_FILTER  = 256
NUM_HIDDEN  = 128

import csv
import numpy as np
import os
import tensorflow as tf
import math
from sklearn import metrics
from tensorflow.keras import Model, layers

# === DIRECTORIES ===
data_dir = ''
LOG_DIR = './snare/logs'
RESULTS_DIR = os.path.join(data_dir, 'results')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# === DATA LOADER ===
def load_ds(file_path):
    with open(file_path) as file:
        lines = file.readlines()

    NUM_SAMPLES = len(lines)
    data = np.zeros((NUM_SAMPLES, MAXSEQ * NUM_FEATURE), dtype=np.float32)
    labels = np.zeros((NUM_SAMPLES, 1), dtype=np.uint8)

    for m, line in enumerate(lines):
        row = line.strip().split(',')
        try:
            labels[m] = int(row[0])  # ✅ label at column 0
            data[m] = np.array(row[1:], dtype='float32')  # ✅ features start from index 1
        except Exception as e:
            print(f"[Error] Line {m} malformed: {e}")
            continue
        print(f"\rReading {file_path}...\t{m+1}/{NUM_SAMPLES}", end='')

    print('\tDone')
    return data, labels

# === LOAD DATA ===
x_train, y_train = load_ds(os.path.join(data_dir, 'dataset', 'pssm.cv.csv'))
x_test, y_test = load_ds(os.path.join(data_dir, 'dataset', 'pssm.cv.csv'))

x_train = np.reshape(x_train, [-1, 1, MAXSEQ, NUM_FEATURE])
x_test = np.reshape(x_test, [-1, 1, MAXSEQ, NUM_FEATURE])
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# === MODEL DEFINITION ===
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

# === METRICS LOGGING ===
def val_binary_init():
    with open(f'{LOG_DIR}/training.csv', 'w') as fout:
        fout.write('Epoch,TP,FP,TN,FN,Sens,Spec,Acc,MCC\n')

def val_binary(epoch, logs):
    with open(f'{LOG_DIR}/training.csv', 'a') as fout:
        y_pred = model.predict(x_test)
        TN, FP, FN, TP = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)).ravel()

        Sens = TP / (TP + FN) if TP + FN > 0 else 0.0
        Spec = TN / (FP + TN) if FP + TN > 0 else 0.0
        Acc = (TP + TN) / (TP + FP + TN + FN)
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if all((TP + FP, TP + FN, TN + FP, TN + FN)) else 0.0

        fout.write(f'{epoch + 1},{TP},{FP},{TN},{FN},{Sens:.4f},{Spec:.4f},{Acc:.4f},{MCC:.4f}\n')

# === BUILD AND COMPILE ===
model = DeepScan(
    num_filters=NUM_FILTER,
    num_hidden=NUM_HIDDEN,
    window_sizes=[8,16,24,32,40,48]
)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ This triggers model building with a dummy batch
_ = model(tf.random.normal([1, 1, MAXSEQ, NUM_FEATURE]))

# ✅ Now the summary will work
model.summary()

# === TRAIN ===
val_binary_init()

model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=[
        tf.keras.callbacks.LambdaCallback(on_epoch_end=val_binary),
        tf.keras.callbacks.ModelCheckpoint(
            LOG_DIR + '/weights.{epoch:02d}.weights.h5',  # ✅ fixed extension
            save_weights_only=True,
            monitor='val_loss',
            mode='max'
        )
    ]
)

# === SAVE PREDICTIONS ===
y_pred = model.predict(x_test)
np.savetxt(os.path.join(RESULTS_DIR, 'fold5.pssm.pred.txt'), y_pred, fmt='%.6f', delimiter=',')
np.savetxt(os.path.join(RESULTS_DIR, 'fold5.y_test.txt'), y_test, fmt='%d', delimiter=',')