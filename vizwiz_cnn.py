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

#traverse through both folders defined above
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

#randomize for better training
random.shuffle(training_data)

for feature, label in training_data:
    feature_set.append(feature)
    label_set.append(label)

# this is 3D, but for plt.imshow(img), it needs to be a 2D image
feature_set = np.array(feature_set).reshape(-1, IMG_SIZE, IMG_SIZE)

# split data using sklearn (80:20 ratio between training and test data)
x_train, x_test, y_train, y_test = train_test_split(feature_set, label_set, test_size = 0.2)

x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

#one hot encoding
# i.e. (1., 0.) for a non-priv image
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

x_train,x_valid,train_label,valid_label = train_test_split(x_train, y_train_one_hot, test_size=0.2)

#begin forming model
vizwiz_model = Sequential()

vizwiz_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(IMG_SIZE,IMG_SIZE,1),padding='same'))
vizwiz_model.add(MaxPooling2D((2, 2),padding='valid'))
vizwiz_model.add(Dropout(0.3))
vizwiz_model.add(LeakyReLU(alpha = 0.1))

vizwiz_model.add(Conv2D(32, (3, 3), activation='linear',padding='valid'))
vizwiz_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
vizwiz_model.add(Dropout(0.3))
vizwiz_model.add(LeakyReLU(alpha = 0.1))

vizwiz_model.add(Conv2D(64, (3, 3), activation='linear',padding='valid'))
vizwiz_model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
vizwiz_model.add(Dropout(0.5))
vizwiz_model.add(LeakyReLU(alpha = 0.1))

vizwiz_model.add(Conv2D(64, (3, 3), activation='linear',padding='valid'))
vizwiz_model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
vizwiz_model.add(Dropout(0.5))
vizwiz_model.add(ReLU())

vizwiz_model.add(Flatten())

vizwiz_model.add(Dense(64, activation='linear'))
vizwiz_model.add(LeakyReLU(alpha = 0.1))

vizwiz_model.add(Dropout(0.4))
vizwiz_model.add(Dense(num_classes, activation='softmax'))

vizwiz_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

vizwiz_model.summary()

vizwiz_model.fit(x_train, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_valid, valid_label), shuffle=True)

#test your results with test data
predicted_classes = vizwiz_model.predict(x_test)
predicted_classes = np.argmax(np.round(predicted_classes), axis = 1)

#classification problem -> [0,1] => [non-priv, priv] 
classes = np.unique(np.array(y_train))
nclasses = len(classes)

#print a metrics, essentially a report of precision, recall, f1-score and support
metrics = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=metrics))
