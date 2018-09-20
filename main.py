from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import tensorflow as tf
import numpy as np
import os
import random
import math
import glob

from random import shuffle
from random import randint
from operator import add
from numpy import genfromtxt
from logger import Logger

FLAGS = None
N_STEPS = 985   # Max num of frames between train and development sets
rootPath = '/home2/alexgomezalanis/antispoofing-noise/data'

def deepnn(x, keep_prob, seq_length):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 48*31)
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 6), with values
    equal to the logits of classifying the digit into one of 6 classes. keep_prob is a scalar placeholder for the probability of
    dropout.
  """

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 48, 31, 2])

  # First convolutional layer - maps one grayscale image to 64 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([9, 9, 2, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 3X.
  # Input: 48x31x64 -> Output: 16x10x64
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_3x3(h_conv1)

  # Second convolutional layer -- maps 64 feature maps to 128.
  # Input: 16x10x64 -> Output: 16x10x128
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([4, 4, 64, 128])
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  # Input: 16x10x128 -> Output: 5x3x128
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_3x3(h_conv2)
  
  h_pool2_flat = tf.reshape(h_pool2, [3, -1, 5 * 3 * 128])

  gru_cell = tf.contrib.rnn.GRUCell(num_units=1920, activation=tf.nn.relu)

  outputs, states = tf.nn.dynamic_rnn(gru_cell, h_pool2_flat, dtype=tf.float32, sequence_length=seq_length)

  #layers = [tf.contrib.rnn.GRUCell(num_units=1920, activation=tf.nn.relu) for layer in range(1)]

  #multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)

  #outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, h_pool2_flat, dtype=tf.float32, sequence_length=seq_length)

  with tf.name_scope('fc3'):
    W_fc3 = weight_variable([1920, 6])
    b_fc3 = bias_variable([6])

    y_rnn = tf.matmul(states, W_fc3) + b_fc3

  return y_rnn

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
  """max_pool_3x3 downsamples a feature map by 3X."""
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def getNameFiles(kind, N):
  nameFiles = np.empty((6, N), dtype=object)
  for i in range(6):
    fileNames = glob.glob(kind + '/S' + str(i) + '/*.npy')
    shuffle_index = np.random.permutation(len(fileNames))
    fileNames = np.asarray(fileNames)
    fileNames = fileNames[shuffle_index]
    fileNames = fileNames[:N]
    for j, m in np.ndenumerate(fileNames):
      nameFiles[i,j] = fileNames[j]
  return nameFiles

def getBatchFeatures(nameFiles, batch, classes, median, std, kind_features):
  dataFeatures = np.zeros((3, N_STEPS, 2, 48, 31), dtype=np.float32) # [batch, seq_length, channels, height, width]
  dataLabels = np.zeros((3,6), dtype=np.int32)
  seqLengthBatch = np.zeros((3), dtype=np.int32)
  count = 0
  for i in classes:
    dataLabels[count, i] = 1
    for k in range(2):
      os.chdir(rootPath + '/' + kind_features[k] + '/')
      feat = np.transpose(np.load(nameFiles[i, batch]))
      if k == 0:
        numFrames = len(feat[0])
        seqLengthBatch[count] = numFrames
        for frame in range(numFrames):
          feat[:,frame] = (feat[:,frame] - median) / std
      for frame in range(seqLengthBatch[count] - 30):
        dataFeatures[count, frame, k, :, :] = feat[:, frame:frame + 31]
    count += 1
  return dataFeatures, dataLabels, seqLengthBatch

def main(_):
  kind_features_mag = sys.argv[1]   # mfcc-features-htk-48, cqcc-features-48, fbank-features-htk-48
  kind_features_phase = sys.argv[2] # mgd-features-48, rps-features-48
  kind_features = [kind_features_mag, kind_features_phase]

  # Create the model
  x = tf.placeholder(tf.float32, [3, None, 2, 48, 31])

  # Length of the utterance (Number of frames)
  seq_length = tf.placeholder(tf.int32, [3])

  # Keep dropout probability
  keep_prob = tf.placeholder(tf.float32)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [3, 6])

  # Build the graph for the deep net
  y_conv = deepnn(x, keep_prob, seq_length)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  # Graph Location for training 
  graph_location = rootPath + '/graph-cnn-rnn-tensorflow22'
  logger = Logger(graph_location)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  os.chdir(rootPath + '/' + kind_features_mag + '/')

  median_train = np.load('training/median-train.npy')
  variance_ac_train = np.load('training/variance-train.npy')
  N_frames_train = np.load('training/num-frames-train.npy')
  median_train = (1.0 * median_train) / N_frames_train
  variance_ac_train = (1.0 * variance_ac_train) / N_frames_train
  variance_train = variance_ac_train - median_train * median_train
  std_train = np.sqrt(variance_train)

  N_training = 2525
  N_dev = 500
  num_epochs = 300
  nameFilesDev = getNameFiles('development', N_dev)
  
  bestPreviousEpochAccuracy = 0
  numEpochsNotImproving = 0
  classes = [[0,1,2], [3,4,5]]

  with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
      #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
      #print('-' * 10)
      nameFilesTrain = getNameFiles('training', N_training)
      for batch in range(N_training):
        #print('BATCH: ' + str(batch))
        for i in range(2):
          trainFeatures, trainLabels, seqLengthBatch = getBatchFeatures(nameFilesTrain, batch, classes[i], median_train, std_train, kind_features)
          sess.run(train_step, feed_dict={x: trainFeatures, seq_length: seqLengthBatch, y_: trainLabels})
      accuracy_avg = 0
      entropy_avg = 0
      for utterance in range(N_dev):
        for i in range(2):
          devFeatures, devLabels, seqLengthBatch = getBatchFeatures(nameFilesDev, utterance, classes[i], median_train, std_train, kind_features)
          epochAccuracy, epochCrossEntropy = sess.run([accuracy, cross_entropy], feed_dict={x: devFeatures, seq_length: seqLengthBatch, y_: devLabels})
          accuracy_avg += epochAccuracy
          entropy_avg += epochCrossEntropy
      accuracy_avg = (1.0 * accuracy_avg)/(N_dev * 2)
      entropy_avg = (1.0 * entropy_avg)/(N_dev * 2)
      #print("Epoch accuracy: " + str(accuracy_avg))
      #print("Epoch cross entropy: " + str(entropy_avg))
      logger.log_scalar('Epoch accuracy CNN+RNN Tensorflow 22', accuracy_avg, epoch)
      logger.log_scalar('Epoch cross entropy CNN+RNN Tensorflow 22', entropy_avg, epoch)
      if (accuracy_avg > bestPreviousEpochAccuracy):
        saver.save(sess, rootPath + '/model-cnn-rnn-tensorflow22/rnn.ckpt')
        bestPreviousEpochAccuracy = accuracy_avg
        numEpochsNotImproving = 0
      else:
        numEpochsNotImproving = numEpochsNotImproving + 1
        if (numEpochsNotImproving > 10):
          break

if __name__ == '__main__':
  tf.app.run(main=main)
