import os
import cv2
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import tensorflow as tf

import keras
from keras.utils import to_categorical
from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation, Dense, BatchNormalization

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



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

# split data using sklearn (80:20 ratio between training and test data)
x_train, x_test, y_train, y_test = train_test_split(feature_set, label_set, test_size = 0.2)

#classification problem -> [0,1] => [non-priv, priv] 
num_classes = len(np.unique(np.array(y_train)))

#normalize between 0 - 1
x_train = x_train / 255.0
x_test = x_test / 255.0

#one hot encoding
# i.e. (1., 0.) for a non-priv image
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

x_train,x_valid,train_label,valid_label = train_test_split(x_train, y_train_one_hot, test_size=0.2)

#begin forming model
vizwiz_model = Sequential()

vizwiz_model.add(Conv2D(32, (3, 3),input_shape=(IMG_SIZE,IMG_SIZE,1), padding = "same"))
vizwiz_model.add(MaxPooling2D(pool_size=(2, 2), padding = "same"))
vizwiz_model.add(Dropout(0.2))
vizwiz_model.add(Activation("relu"))

vizwiz_model.add(Conv2D(64, (3, 3), padding = "same"))
vizwiz_model.add(MaxPooling2D(pool_size=(2, 2), padding = "same"))
vizwiz_model.add(Dropout(0.3))
vizwiz_model.add(Activation("relu"))

vizwiz_model.add(Conv2D(64, (3, 3), padding = "same"))
vizwiz_model.add(MaxPooling2D(pool_size=(2, 2), padding = "same"))
vizwiz_model.add(Dropout(0.3))
vizwiz_model.add(Activation("relu"))

vizwiz_model.add(Conv2D(64, (3, 3), padding = "same"))
vizwiz_model.add(MaxPooling2D(pool_size=(2, 2), padding = "same"))
vizwiz_model.add(Dropout(0.3))
vizwiz_model.add(Activation("relu"))

vizwiz_model.add(Flatten())

vizwiz_model.add(Dense(128, activation='linear'))
vizwiz_model.add(Activation("relu"))

vizwiz_model.add(Dense(num_classes, activation='sigmoid'))

#define loss function (binary bc only 2 classes), optimzer (adam is usually the most used) 
#and what you want the model to measure
vizwiz_model.compile(loss="binary_crossentropy", 
             optimizer="adam",
             metrics=['accuracy'])

vizwiz_model.summary()

vizwiz_model.fit(x_train, train_label, batch_size=batch_size,epochs=epochs,validation_data=(x_valid, valid_label))

#test our model results with test data
predicted_classes = vizwiz_model.predict(x_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

#print a metrics, essentially a report of precision, recall, f1-score and support
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))
#print a metrics, essentially a report of precision, recall, f1-score and support
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))
