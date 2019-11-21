import os
import cv2
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.model_selection import train_test_split


IMG_SIZE = 100
DATADIR = "C:\Datasets\VizWiz"
CATEGORIES = ["Non-Priv", "Priv"]

training_data = []
feature_set = []
label_set = []

def one_hot_label(img):
    label = img.split('.')[0]
    if label == 'non':
        ohl = np.array([1,0])
    else:
        ohl = np.array([0, 1])
    return ohl

def initializing_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        list_of_imgs = os.listdir(path)

        for each_img in list_of_imgs:
            try:
                img_array = cv2.imread(os.path.join(path, each_img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                #plt.imshow(new_array, cmap='gray')
                #plt.show()
                training_data.append([new_array, class_num])
            except Exception as e:
                print("Exception " + e + " Path: " + path + " does not exist.")
            
initializing_training_data()

random.shuffle(training_data)

for feature, label in training_data:
    feature_set.append(feature)
    label_set.append(label)


feature_set = np.array(feature_set).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# #split data using sklearn
X_train, X_test, y_train, y_test = train_test_split(feature_set, label_set, test_size = 0.2)

# #normalize data between 0-1
# X_train = tf.keras.utlis.normalize(X_train, axis=1)
# X_test = tf.keras.utlis.normalize(X_test, axis=1)

# #define model
# model = tf.keras.models.Sequential()
# #flatten input
# model.add(tf.keras.layers.FLatten())
