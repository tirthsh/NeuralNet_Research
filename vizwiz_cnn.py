import os
import cv2
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


IMG_SIZE = 100
DATADIR = "C:\Datasets\VizWiz"
CATEGORIES = ["Non-Priv", "Priv"]

training_data = []
feature_set = []
label_set = []

# def one_hot_label(img):
#     label = img.split('.')[0]
#     if label == 'Non':
#         ohl = np.array([1,0])
#     else:
#         ohl = np.array([0, 1])
#     return ohl

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
feature_set = np.array(feature_set).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# split data using sklearn (80:20 ratio between training and test data)
x_train, x_test, y_train, y_test = train_test_split(feature_set, label_set, test_size = 0.2)

#tf.image.rgb_to_grayscale(x_train)

# normalize data between 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)

# define model
model = Sequential()

# add layers
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = x_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #converts our 3D feature into 1D feature vector

model.add(Dense(1))
model.add(Activation("sigmoid"))

#define configuration of how your model will learn
model.compile(loss="binary_crossentropy", 
             optimizer="adam",
             metrics=['accuracy'])

#train model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1)
