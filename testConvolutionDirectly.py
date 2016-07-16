# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import cPickle
import dataset

batch_size = 100
patch_size = 3
depth = 16

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

"""
train_dataset = np.array([[[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]],
                          [[1, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0]
                           ],
                          [[1, 0, 1, 0],
                           [0, 1, 1, 0],
                           [0, 0, 1, 0],
                           [0, 0, 1, 1]
                           ]])

train_dataset = train_dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)

train_labels = np.array([[1, 0], [1, 0], [0, 1]])

"""

old_trainingSet = cPickle.load(open("/media/tassadar/Work/Google Drive/My/NeuralNet/data/mnist/MNISTTrainingSet_square", 'rb'))
#cvSet = cPickle.load(open("/media/tassadar/Work/Google Drive/My/NeuralNet/data/mnist/MNISTTestSet_square", 'rb'))

trainingSet = dataset.dataset(examples=old_trainingSet.examples, labels=old_trainingSet.labels)
trainingSet.rearrangeToCubic()

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [image_size * image_size * depth, num_labels], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)
    hidden = tf.nn.relu(conv + layer1_biases)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    #print(reshape.get_shape())
    #print(layer2_weights.get_shape())
    return tf.matmul(reshape, layer2_weights) + layer2_biases

  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)

  num_steps = 10000

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        data = trainingSet.getMiniBatch(size=batch_size)
        trainingExamples = data['examples']
        trainingLabels = data['labels']
        feed_dict = {tf_train_dataset: trainingExamples, tf_train_labels: trainingLabels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, trainingLabels))