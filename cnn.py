import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import time

import pickle

X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

#normalize tensors
X = X / 255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #converts our 3D feature into 1D feature vector

#makes val loss higher
#model.add(Dense(64))
#model.add(Activation("relu"))

#output layer
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", 
             optimizer="adam",
             metrics=['accuracy'])

#10% VS, 90% TS
model.fit(X, Y, batch_size=32, epochs=10, validation_split=0.1)