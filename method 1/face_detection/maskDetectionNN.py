import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.applications import MobileNet
import json 
import cv2

# initializing base model
conv_base = MobileNet(weights='imagenet',
                     include_top=False,
                     input_shape=(150,150,3))

def modelInitAndTraining():
  # Creacting features for secondd model
  datagen = ImageDataGenerator(rescale=1/255)
  batch_size = 100

  def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 1024))
    labels = np.zeros(shape=(sample_count,3))
    generator = datagen.flow_from_directory(
      directory,
      target_size = (150,150),
      batch_size = batch_size,
      class_mode = 'categorical'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
      features_batch = conv_base.predict(inputs_batch)
      features[i*batch_size:(i+1)*batch_size] = features_batch
      labels[i*batch_size:(i+1)*batch_size] = labels_batch
      i += 1
      if i * batch_size >= sample_count: 
        break
    return features, labels

  train_features, train_labels = extract_features('./training/mask_train', 4428)
  val_features, val_labels = extract_features('./training/mask_val', 1200)
  train_features = np.reshape(train_features, (4428, 4*4*1024))
  val_features = np.reshape(val_features, (1200, 4*4*1024))
  
  # initializing transfer learning model
  model = Sequential()
  model.add(Dense(512, activation='relu', input_dim = 4*4*1024))
  model.add(Dropout(0.5))
  model.add(Dense(3, activation='softmax'))
  sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

  model.fit(train_features, train_labels,epochs=15,batch_size=100, validation_data = (val_features, val_labels))
  return model

""" def predict(filepath):
    datagen = ImageDataGenerator(rescale=1/255)
    batch_size = 1

    def generate_features(directory, sample_count = 1):
      features = np.zeros(shape=(sample_count, 4, 4, 1024))
      labels = np.zeros(shape=(sample_count))
      generator = datagen.flow_from_directory(
        directory,
        target_size = (150,150),
        batch_size = batch_size,
        class_mode = 'categorical'
      )
      i = 0
      for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count: 
          break
      return features
    features = generate_features(filepath)
    new_f = np.reshape(features, (1, 4*4*1024))
    return model.predict(new_f) """

def predict(filepath):
  img1 = cv2.imread(filepath)
  img1 = cv2.resize(img1, (150, 150))
  img1=img1*1.0/255
  img1 = np.reshape(img1, (1,150,150,3))
  feature_Test = conv_base.predict(img1)
  feature_Test = np.reshape(feature_Test, (1, 4*4*1024))
  return model.predict(feature_Test)

def save_model():
  model_json = model.to_json()
  with open("./model.json", "w") as json_file:
      json_file.write(model_json)
  model.save_weights("./model_weights.h5")

def load_model():
    f = open('./face_detection/model.json',)
    data = json.dumps(json.load(f))
    model = model_from_json(data)
    model.load_weights('./face_detection/model_weights.h5')
    return model

#model = modelInitAndTraining()
#save_model()

model = load_model()