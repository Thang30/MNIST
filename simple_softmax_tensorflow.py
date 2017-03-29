# this program achieved score 0.87386 on kaggle publicscore
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import helpers.settings as settings
from keras.utils import np_utils


# configure the path
data_dir = os.path.join(os.getcwd(), settings.DATA_DIR)
out_dir = os.path.join(os.getcwd(), settings.OUT_DIR)

# load the data
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
X_train = train.drop("label", axis=1).values
Y_train = train["label"].values
X_test = test.values

# number of samples and feautures and labels
num_samples_train = X_train.shape[0]
num_samples_test = X_test.shape[0]
num_pixels = X_train.shape[1]
num_classes = np.unique(Y_train).shape[0]

# one-hot encode outputs
Y_train = np_utils.to_categorical(Y_train)

# create symbolic values
x = tf.placeholder(tf.float32, [None, num_pixels])
W = tf.Variable(tf.zeros([num_pixels, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))
y_ = tf.placeholder(tf.float32, [None, num_classes])

# simple model with only 1 activation function
y = tf.nn.log_softmax(tf.matmul(x, W) + b)

# the loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                              y, reduction_indices=[1]))

# optimizier for backpropagation error
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# create and run a Tensorflow session
sess = tf.InteractiveSession()

# initialize the variables
tf.global_variables_initializer().run()

# batch size
batch_size = 100
# training session
for _ in range(1000):  # 1000 epochs
  # random sample, with batch size equal to 100
  idx = np.random.randint(num_samples_train, size=batch_size)
  batch_xs = X_train[idx, :]
  batch_ys = Y_train[idx, :]

  # feed batch into placeholder and run the session
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # print out the training session
  if _ % 100 == 0:
    print('Training Step:' + str(_) + '  Loss = ' +
          str(sess.run(cross_entropy, {x: batch_xs, y_: batch_ys})))

# predicting session
Y_test = []
for i in range(num_samples_test):
  # feed the text sample
  feed_dict = {x: X_test[i, :].reshape(1, -1)}

  # predict test label using trained model
  prediction = sess.run(y, feed_dict).argmax()
  Y_test.append(prediction)

print(Y_test)

# predictions for competition!
submission = pd.DataFrame({"ImageId": list(range(1, len(Y_test) + 1)),
                           "Label": Y_test})
submission.to_csv(os.path.join(
    out_dir, 'simple_softmax_tensorflow.csv'), index=False)
