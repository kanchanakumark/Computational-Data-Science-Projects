Image Classification using Multi Layer Perceptron

Introduction

Traffic sign recognition is a challenging, real-world problem relevant for AI based transportation systems. Traffic signs show a wide range of variations between classes in terms of color, shape, and the presence of pictograms or text. However, there exist subsets of
classes (e.g., speed limit signs) that are very similar to each other. Further, the classifier
has to be robust against large variations in visual appearances due to changes in illumination, partial
occlusions, rotations, weather conditions etc. Using a comprehensive traffic sign detection dataset, here we will perform classification of traffic signs, train and evaluate the different models and compare to the performance of MLPs.

Dataset

The data for this mini-project is from the German Traffic Sign Detection Benchmark [GTSDB](https://benchmark.ini.rub.de/gtsdb_dataset.html). This archive contains the training set used during the IJCNN 2013 competition. 

The German Traffic Sign Detection Benchmark is a single-image detection assessment for researchers with interest in the field of computer vision, pattern recognition and image-based driver assistance. It is introduced on the IEEE International Joint Conference on Neural Networks 2013. 

Problem Statement

To build and improve upon a machine learning model for the classification of images and achieve a high accuracy final model.


#Download the data
!wget -qq https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip
!unzip -qq FullIJCNN2013.zip

#Import Required packages

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from skimage.io import imread, imshow
from sklearn import preprocessing
import os, glob
from PIL import Image
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Keras
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

#Data Loading and Feature Extraction

#Get the features and labels of data

#Extract the features of the images
#Extract labels of the images
#Resize the images to (30, 30) and convert to numpy 1-D array

#Initial test of data

gt = pd.read_csv('/content/FullIJCNN2013/gt.txt', delimiter=';', header=None)
gt.columns = ['ImgNo', 'leftCol', 'topRow', 'rightCol', 'bottomRow', 'ClassID']
gt.head(10)

path = '/content/FullIJCNN2013/38/00000.ppm'
image = Image.open(path)
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)
# show the image
image

image_resized = image.resize((30,30))
image_resized

img = imread(path, as_gray=False)
img1 = img.copy()
img1.resize((30,30))
print(img1.shape)
print(img.shape, type(img))

plt.imshow(img1)#, cmap='gray')
plt.show()

a = img.ravel()
a.shape

tf_img = tf.image.per_image_standardization(img)
tf_img.shape

tf_img1 = tf.image.resize(tf_img, (30,30))

img_np = tf_img1.numpy()

b = img_np.ravel()
b.shape

plt.imshow(img_np)
plt.show()

#Create dictionary

ID = [x for x in range(43)]
Label = ['speed limit 20 (prohibitory)',
'speed limit 30 (prohibitory)',
'speed limit 50 (prohibitory)',
'speed limit 60 (prohibitory)',
'speed limit 70 (prohibitory)',
'speed limit 80 (prohibitory)',
'restriction ends 80 (other)',
'speed limit 100 (prohibitory)',
'speed limit 120 (prohibitory)',
'no overtaking (prohibitory)',
'no overtaking (trucks) (prohibitory)',
'priority at next intersection (danger)',
'priority road (other)',
'give way (other)',
'stop (other)',
'no traffic both ways (prohibitory)',
'no trucks (prohibitory)',
'no entry (other)',
'danger (danger)',
'bend left (danger)',
'bend right (danger)',
'bend (danger)',
'uneven road (danger)',
'slippery road (danger)',
'road narrows (danger)',
'construction (danger)',
'traffic signal (danger)',
'pedestrian crossing (danger)',
'school crossing (danger)',
'cycles crossing (danger)',
'snow (danger)',
'animals (danger)',
'restriction ends (other)',
'go right (mandatory)',
'go left (mandatory)',
'go straight (mandatory)',
'go right or straight (mandatory)',
'go left or straight (mandatory)',
'keep right (mandatory)',
'keep left (mandatory)',
'roundabout (mandatory)',
'restriction ends (overtaking) (other)',
'restriction ends (overtaking (trucks)) (other)']

sign_labels = pd.DataFrame(ID, columns=['ClassID'])
sign_labels['Label'] = Label
sign_labels.head()

#Start feature extraction

sign_files = glob.glob('/content/FullIJCNN2013/*/*.ppm')
len(sign_files)

def extract_features(file_name):
    df=np.array([])
    flat_arr=np.array([])

    img = imread(file_name, as_gray=False)
    tf_img = tf.image.per_image_standardization(img)
    tf_img1 = tf.image.resize(tf_img, (30,30))
    img_np = tf_img1.numpy()
    # c = img.copy()
    # c.resize((30,30))
    # possible variations - grayscale, brightening and sharpening
    flat_arr = img_np.ravel()
    df = np.hstack((df, flat_arr)) 

    fn = int(file_name.split("/")[3])
    df = np.hstack((df, fn))

    return df

img_features = []
for file_name in sign_files:
    img_features.append(extract_features(file_name))

df = pd.DataFrame(img_features)
df.head()

df.shape

#Data Exploration and Preprocessing

f, ax = plt.subplots(7,7, figsize=(15,15))
m, n = (0,0)
for i in range(43):
    m = int(i/7)
    n = i - m*7
    if i < 10: j = '0'+ str(i)
    else: j = str(i)
    path = '/content/FullIJCNN2013/'+ j + '/00000.ppm'
    image = imread(path, as_gray=False)
    ax[m, n].imshow(image, cmap='gray')
plt.show()

#Plot the distribution of Classes

plt.figure(figsize=(16,5))
df[2701].hist(bins=43, rwidth=0.5)
plt.show()

#Normalize the features

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_normalized = pd.DataFrame(preprocessing.normalize(X, norm='l2'))
X_normalized.shape

y.describe()

X_normalized.head()

plt.figure(figsize=(16,5))
y.hist(bins=43, rwidth=0.5)
plt.show()

#Train the MLP classifier on features

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size = 0.2, random_state = 42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42, stratify=y_train)

# # Using stratified shuffle split
# from sklearn.model_selection import StratifiedShuffleSplit
# sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
# sss.get_n_splits(X, y)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 43), activation="relu", solver='adam', alpha=0.001, max_iter=500).fit(X_train, y_train)

y_pred = np.round(clf.predict(X_test), 0)
y_pred

# from sklearn.metrics import mean_squared_error
# np.sqrt(mean_squared_error(y_test, y_pred))
clf.score(X_test, y_test)

#Tune the hyper-parameters

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100), (100,), (100, 100, 100, 43)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

clf = MLPClassifier(max_iter=500)
rand_clf = RandomizedSearchCV(clf, param_distributions=parameter_space)
search = rand_clf.fit(X_train, y_train)

search.best_score_

search.best_params_

#Ignore keras gridsearch

from keras.optimizers import SGD    
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[900]):
    model = Sequential()
    options = {"input_shape": input_shape}
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation="relu", **options))
        options = {}
    model.add(Dense(43, **options))
    optimizer = SGD(learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

from scipy.stats import reciprocal
param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=10, validation_data=(X_val,y_val))

rnd_search_cv.best_params_

rnd_search_cv.best_score_

model = rnd_search_cv.best_estimator_.model

model.evaluate(X_test, y_test)

#Try the different algorithms and compare the results with MLP classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', max_depth=3)
rf.fit(X_train, y_train)

rf.score(X_test, y_test)

#Ignore

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, validation_data = (X_val, y_val))

vis = pd.DataFrame(history.history)
vis.plot(figsize=(8, 5))
plt.grid(True)
#plt.gca().set_ylim(0, 1) 
plt.show()

model.evaluate(X_test, y_test)

"#Implement simple Neural Networks using keras
print(tf.__version__)

#Create model with 2 hidden layers and one output layer
model = Sequential([
                    Dense(600, activation="relu"),
                    Dense(300, activation="relu"),
                    Dense(100, activation="relu"),
                    Dense(43, activation="softmax")
                    ])

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])

#Fit and Evaluate the model
history = model.fit(X_train, y_train, epochs=30, validation_data = (X_val, y_val))

vis = pd.DataFrame(history.history)
vis.plot(figsize=(8, 5))
plt.grid(True)
#plt.gca().set_ylim(0, 1) 
plt.show()

model.evaluate(X_test, y_test)

model.summary()

y_pred = np.argmax(model.predict(X_test), axis=-1)
y_pred

from tensorflow.keras.layers import Flatten
from keras.regularizers import l2

# YOUR CODE HERE
model = Sequential([
                    Flatten(input_shape=[30, 30]),
                    Dense(100, activation="relu", kernel_initializer="he_normal"),
                    Dense(100, activation="relu"),
                    Dense(100, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
                    Dense(43, activation="softmax")
                    ])

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'], )

history = model.fit(X_train, y_train, epochs=30, validation_data = (X_val, y_val))

vis = pd.DataFrame(history.history)
vis.plot(figsize=(8, 5))
plt.grid(True)
#plt.gca().set_ylim(0, 1) 
plt.show()

model.evaluate(X_test, y_test)

#Experiment using Dropout, Regularization and Batch Normalization

#Without Dropout

from functools import partial
from keras.layers import Activation, Dense, Input, Flatten, Dropout, BatchNormalization

layer = Dense(100, activation="relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))

# creating regularized dense layer for model
RegularizedDense = partial(keras.layers.Dense,
                           activation="relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))

# defining model with regularization
model = Sequential([
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(43, activation="softmax",
                     kernel_initializer="glorot_uniform")
])

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, validation_data = (X_val, y_val))

vis = pd.DataFrame(history.history)
vis.plot(figsize=(8, 5))
plt.grid(True)
#plt.gca().set_ylim(0, 1) 
plt.show()

#Dropout
model = Sequential([
                    #Flatten(input_shape=[30, 30]),
                    Dense(100, activation="relu", kernel_initializer="he_normal"),
                    #Dropout(rate=0.2),
                    Dense(100, activation="relu"),
                    #Dropout(rate=0.2),
                    Dense(100, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
                    Dropout(rate=0.2),
                    Dense(43, activation="softmax")
                    ])

# time based learning-rate scheduling
epochs = 100
learning_rate = 0.1
decay_rate = learning_rate / epochs
# define optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, decay=decay_rate)
 
# Compile model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train,
          batch_size=128,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))

vis = pd.DataFrame(history.history)
vis.plot(figsize=(8, 5))
plt.grid(True)
#plt.gca().set_ylim(0, 1) 
plt.show()

model.evaluate(X_test, y_test)


#Reference: J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011
