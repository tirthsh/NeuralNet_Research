import numpy as np #for array operations
import matplotlib.pyplot as plt
import os
import cv2 #for image operations
import random #to shuffle data for better training
import pickle #save and load data

IMG_SIZE = 100
DATADIR = "C:/Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #path to dogs and cat folder 
        class_num = CATEGORIES.index(category) #change classification into 0 or 1
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) #turn image into array with whole path, turn it into grayscale to reduce complexity of image  (i.e. RGB data is 3x size of GS image)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
create_training_data()

random.shuffle(training_data)

X = [] #feature set
Y = [] #label set

for features, label in training_data:
    X.append(features)
    Y.append(labels)

#convert X into numpy error - REQUIREMENT for Keras
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


####################### SAVING DATA #############################
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

###################### LOAD SAVED DATA ########################
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

print(X[1])
