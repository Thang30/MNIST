# this program reach 0.97571 accuracy on Kaggle public score

import theano
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


# fix seed for reproducibility
seed = 10
np.random.seed(seed)

# load the data
train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')
X_train = train.drop("label", axis=1).values
Y_train = train["label"].values
X_test = test.values

# number of feautures and labels
num_pixels = X_train.shape[1]
num_classes = np.unique(Y_train).shape[0]

# normalize the data
X_train = X_train / 255
X_test = X_test / 255

# one-hot encode outputs
Y_train = np_utils.to_categorical(Y_train)

# 1 hidden layer neural network
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels,
                init='normal', activation='relu'))
model.add(Dense(num_classes, init='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, nb_epoch=10, batch_size=200, verbose=2)
Y_test = model.predict(X_test, batch_size=200, verbose=0)
Y_test = list(Y_test.argmax(axis=1))

# predictions for competition!
submission = pd.DataFrame({"ImageId": list(range(1, len(Y_test) + 1)),
                           "Label": Y_test})
submission.to_csv('outputs/1-hidden-layer-NN.csv', index=False)
