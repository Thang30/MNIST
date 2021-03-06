# this program reach 0.98814 accuracy on Kaggle accuracy

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


# fix random seed for reproducibility
seed = 10
np.random.seed(seed)

# load data
train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')
X_train = train.drop("label", axis=1).values
Y_train = train["label"].values
X_test = test.values

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
num_classes = Y_train.shape[1]

# complex convolution neural network
model = Sequential()

# the first block of convolutional-pooling layers
model.add(Convolution2D(30, 5, 5, border_mode='valid',
                        input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the second block of convolutional-pooling layers
model.add(Convolution2D(15, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# dropout layer
model.add(Dropout(0.2))

# fully connected layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))

# output layer
model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(X_train, Y_train, nb_epoch=10, batch_size=200, verbose=2)

# predict on test data
Y_test = model.predict(X_test, batch_size=200, verbose=0)
Y_test = list(Y_test.argmax(axis=1))

# predictions for competition!
submission = pd.DataFrame({"ImageId": list(range(1, len(Y_test) + 1)),
                           "Label": Y_test})
submission.to_csv('outputs/complex_convolution_NN.csv',
                  index=False)
