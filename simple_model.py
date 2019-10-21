import tensorflow as tf

#load dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize so its between 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#initialize a sequential model not a functional one
model = tf.keras.models.Sequential()

#input layer
model.add(tf.keras.layers.Flatten())
#2 middle layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#a neural net never aims to maximize accuracy, it always aims to minimize error
#this is the learning model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

#==========================================================================#
import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

#print(x_train[0])

#==========================================================================#
import matplotlib.pyplot as plt
import numpy as np

prediction = model.predict([x_test])
print(np.argmax(prediction[0]))
      
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()
     
