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
N_FILTS = 48    # Number of filters considered in each frame
N_FRAMES = 31   # Number of consecutive frames of a context
N_CLASSES = 6   # Genuine speech and 5 spoofing attacks of the training set
N_BATCH = 3     # Number of utterances to process per batch
N_ITER_BATCH = int(N_CLASSES / N_BATCH) 
rootPath = os.getcwd()

def CNN_RNN(x, keep_prob, seq_length):
  """
  CNN_RNN builds the graph for a CNN + RNN for classifying the utterance into genuine speech
  or one of the spoofing attacks present in the training set.
  Args:
    x: an input tensor with the dimensions (N_utterances, N_STEPS, 1, N_FILTS, N_FRAMES)
    keep_prob: a scalar placeholder for the probability of dropout.
    seq_length: Sequence length of each utterance with the dimensions (N_utterances)
  Returns:
    y is a tensor of shape (N_utterances, 6), with values equal to the logits of
    classifying the utterance into one of the 6 classes.
  """

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, N_FILTS, N_FRAMES, 1])

  # First convolutional layer - maps one grayscale image to 64 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([9, 9, 1, 64])
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

  _, states = tf.nn.dynamic_rnn(gru_cell, h_pool2_flat, dtype=tf.float32, sequence_length=seq_length)

  with tf.name_scope('fc3'):
    W_fc3 = weight_variable([1920, N_CLASSES])
    b_fc3 = bias_variable([N_CLASSES])

    y = tf.matmul(states, W_fc3) + b_fc3

  return y

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
  nameFiles = np.empty((N_CLASSES, N), dtype=object)
  for i in range(N_CLASSES):
    fileNames = glob.glob(kind + '/S' + str(i) + '/*.npy')
    shuffle_index = np.random.permutation(len(fileNames))
    fileNames = np.asarray(fileNames)
    fileNames = fileNames[shuffle_index]
    fileNames = fileNames[:N]
    for j, _ in np.ndenumerate(fileNames):
      nameFiles[i,j] = fileNames[j]
  return nameFiles

def getBatchFeatures(nameFiles, batch, median, std):
  dataFeatures = np.zeros((N_CLASSES, N_STEPS, 1, N_FILTS, N_FRAMES), dtype=np.float32) # [batch, seq_length, channels, N_FILTS, N_FRAMES]
  dataLabels = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)
  seqLength = np.zeros((N_CLASSES), dtype=np.int32)
  for n in range(N_CLASSES):
    dataLabels[n, n] = 1
    feat = np.transpose(np.load(nameFiles[n, batch]))
    numFrames = len(feat[0])
    seqLength[n] = numFrames
    for frame in range(numFrames):
      feat[:, frame] = (feat[:, frame] - median) / std
    for frame in range(numFrames - N_FRAMES - 1):
      dataFeatures[n, frame, 0, :, :] = feat[:, frame:frame + N_FRAMES]
  shuffle_index = np.random.permutation(N_CLASSES)
  dataLabels = dataLabels[shuffle_index]
  seqLength = seqLength[shuffle_index]
  dataFeatures = dataFeatures[shuffle_index, :, :, :, :]
  return dataFeatures, dataLabels, seqLength

def main(_):
  kind_features = sys.argv[1] # fbank, mfcc, cqcc

  # Create the model
  x = tf.placeholder(tf.float32, [3, None, 1, N_FILTS, N_FRAMES])

  # Length of the utterance (Number of frames)
  seq_length = tf.placeholder(tf.int32, [3])

  # Keep dropout probability
  keep_prob = tf.placeholder(tf.float32)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [3, N_CLASSES])

  # Build the graph for the deep net
  y_conv = CNN_RNN(x, keep_prob, seq_length)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(3e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  # Graph Location for training 
  graph_location = rootPath + '/graph-cnn-rnn-tensorflow'
  logger = Logger(graph_location)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  os.chdir(rootPath + '/' + kind_features + '/')

  median_train = np.load('training/median-train.npy')
  variance_ac_train = np.load('training/variance-train.npy')
  N_frames_train = np.load('training/num-frames-train.npy')
  median_train = (1.0 * median_train) / N_frames_train
  variance_ac_train = (1.0 * variance_ac_train) / N_frames_train
  variance_train = variance_ac_train - median_train * median_train
  std_train = np.sqrt(variance_train)

  N_training = 2525   # Number of utterances per class considered for training
  N_dev = 500         # Number of utterances per class considered for development
  num_epochs = 300
  nameFilesDev = getNameFiles('development', N_dev)
  
  bestPreviousEpochAccuracy = 0
  numEpochsNotImproving = 0

  with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
      print('Epoch {}/{}'.format(epoch, num_epochs - 1))
      print('-' * 10)
      nameFilesTrain = getNameFiles('training', N_training)
      for batch in range(N_training):
        trainFeaturesBatch, trainLabelsBatch, seqLengthBatch = getBatchFeatures(nameFilesTrain, batch, median_train, std_train)
        for n in range(N_ITER_BATCH):
          startBatch = N_BATCH * n
          endBatch = N_BATCH * (n + 1)
          trainFeatures = trainFeaturesBatch[startBatch : endBatch, :, :, :, :]
          trainLabels = trainLabelsBatch[startBatch : endBatch]
          seqLength = seqLengthBatch[startBatch : endBatch]
          sess.run(train_step, feed_dict={x: trainFeatures, seq_length: seqLength, y_: trainLabels})
      accuracy_avg = 0
      entropy_avg = 0
      for utterance in range(N_dev):
        devFeaturesBatch, devLabelsBatch, seqLengthBatch = getBatchFeatures(nameFilesDev, utterance, median_train, std_train)
        for n in range(N_ITER_BATCH):
          startBatch = N_BATCH * n
          endBatch = N_BATCH * (n + 1)
          devFeatures = devFeaturesBatch[startBatch : endBatch, :, :, :, :]
          devLabels = devLabelsBatch[startBatch : endBatch]
          seqLength = seqLengthBatch[startBatch : endBatch]
          epochAccuracy, epochCrossEntropy = sess.run([accuracy, cross_entropy], feed_dict={x: devFeatures, seq_length: seqLength, y_: devLabels})
          accuracy_avg += epochAccuracy
          entropy_avg += epochCrossEntropy
      accuracy_avg = accuracy_avg / (N_dev * N_ITER_BATCH)
      entropy_avg = entropy_avg / (N_dev * N_ITER_BATCH)
      print("Epoch accuracy: " + str(accuracy_avg))
      print("Epoch cross entropy: " + str(entropy_avg))
      logger.log_scalar('Epoch accuracy CNN + RNN Tensorflow', accuracy_avg, epoch)
      logger.log_scalar('Epoch cross entropy CNN + RNN Tensorflow', entropy_avg, epoch)
      saver.save(sess, rootPath + '/model-cnn-rnn-tensorflow/epoch-' + str(epoch) + '.ckpt')
      if (accuracy_avg > bestPreviousEpochAccuracy):
        saver.save(sess, rootPath + '/model-cnn-rnn-tensorflow/best.ckpt')
        bestPreviousEpochAccuracy = accuracy_avg
        numEpochsNotImproving = 0
      else:
        numEpochsNotImproving += 1
        if (numEpochsNotImproving > 10):
          sys.exit()

if __name__ == '__main__':
  tf.app.run(main=main)
