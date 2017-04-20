# this program reach 0.98514 accuracy on Kaggle public score

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
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
num_classes = Y_train.shape[1]

# simple convolution neural network
model = Sequential()

# the input layer, a convolutional layer
model.add(Convolution2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))

# the pooling layer to down sample the feature maps
model.add(MaxPooling2D(pool_size=(2, 2)))

# dropout layer to prevent overfitting
model.add(Dropout(0.2))

# flatten the data into standard inputs for fully connected layers
model.add(Flatten())

# a fully connected layer
model.add(Dense(128, activation='relu'))

# the final, output layer
model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# train model and save it
model.fit(X_train, Y_train, epochs=10, batch_size=300, verbose=0)
model.save_weights('models/simple_convolution_NN_keras.h5')

# predict on test data
Y_test = model.predict(X_test, batch_size=300, verbose=0)
Y_test = list(Y_test.argmax(axis=1))

# predictions for competition!
submission = pd.DataFrame({"ImageId": list(range(1, len(Y_test) + 1)),
                           "Label": Y_test})
submission.to_csv('outputs/simple_convolution_NN_keras.csv',
                  index=False)
