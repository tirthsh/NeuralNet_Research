import os
import cv2
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation, Dense, BatchNormalization

IMG_SIZE = 64
DATADIR = "C:\Datasets\VizWiz"
CATEGORIES = ["Non-Priv", "Priv"]

training_data = []
feature_set = []
label_set = []

batch_size = 32
epochs = 20
num_classes = 2

def initializing_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        list_of_imgs = os.listdir(path)

        for each_img in list_of_imgs:
            try:
                img_array = cv2.imread(os.path.join(path, each_img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print("Exception " + e + " Path: " + path + " does not exist.")
            
initializing_training_data()

random.shuffle(training_data)

for feature, label in training_data:
    feature_set.append(feature)
    label_set.append(label)

# this is 3D, but for plt.imshow(img), it needs to be a 2D image
feature_set = np.array(feature_set).reshape(-1, IMG_SIZE, IMG_SIZE)

# split data using sklearn (80:20 ratio between training and test data)
x_train, x_test, y_train, y_test = train_test_split(feature_set, label_set, test_size = 0.2)

print("Training data shape: ", np.array(x_train).shape, np.array(y_train).shape)
print("Testing data shape: ", np.array(x_test).shape, np.array(y_test).shape)

#classification problem -> [0,1] => [non-priv, priv] 
classes = np.unique(np.array(y_train))
nclasses = len(classes)

print("Total # of outputs: ", nclasses)
print("Output classes: ", classes)

x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(x_train.shape, x_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#normalize between 0 - 1
x_train = x_train / 255.0
x_test = x_test / 255.0

#one hot encoding
# i.e. (1., 0.) for a non-priv image
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Display the change for category label using one-hot encoding
print('Original label:', y_train[0])
print('After conversion to one-hot:', y_train_one_hot[0])

x_train,x_valid,train_label,valid_label = train_test_split(x_train, y_train_one_hot, test_size=0.2, random_state=13)

print(x_train.shape, x_valid.shape, train_label.shape, valid_label.shape)

#begin forming model
vizwiz_model = Sequential()

vizwiz_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(64,64,1),padding='same'))
vizwiz_model.add(MaxPooling2D((2, 2),padding='same'))
vizwiz_model.add(Dropout(0.25))
vizwiz_model.add(LeakyReLU(alpha=0.1))

vizwiz_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
vizwiz_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
vizwiz_model.add(Dropout(0.25))
vizwiz_model.add(LeakyReLU(alpha=0.1))

vizwiz_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
vizwiz_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
vizwiz_model.add(Dropout(0.4))
vizwiz_model.add(LeakyReLU(alpha=0.1))

vizwiz_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
vizwiz_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
vizwiz_model.add(Dropout(0.4))
vizwiz_model.add(LeakyReLU(alpha=0.1))

vizwiz_model.add(Flatten())

vizwiz_model.add(Dense(128, activation='linear'))
vizwiz_model.add(LeakyReLU(alpha=0.1))

vizwiz_model.add(Dropout(0.3))
vizwiz_model.add(Dense(num_classes, activation='softmax'))

vizwiz_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

vizwiz_model.summary()

train_model = vizwiz_model.fit(x_train, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_valid, valid_label), shuffle=True)

accuracy = train_model.history['acc']
val_accuracy = train_model.history['val_acc']
loss = train_model.history['loss']
val_loss = train_model.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
