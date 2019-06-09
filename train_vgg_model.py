# Note: We leveraged code from https://github.com/Xiaotian-WANG/Fine-Tune-VGG-Networks-Based-on-Stanford-Cars
# to run the transfer learning model

# Import required packages
import numpy as np
import random
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.applications.inception_v3 import InceptionV3
import os
from keras import optimizers, initializers
import tensorflow as tf
import keras
from PIL import Image
from keras.metrics import top_k_categorical_accuracy
import cv2
from keras.preprocessing.image import ImageDataGenerator

# Freeze the first n layers of the neural network
def freeze_layers(model, n_layers_to_freeze):
    print(len(model.layers))
    for i, layer in enumerate(model.layers):
        if i < n_layers_to_freeze:
            layer.trainable = False
        else:
            layer.trainable = True

# Load the appropriate model weights with no top (dense layers) and freeze the first n weights
def model_define(modeltype, inputshape, n_layers_to_freeze):
    if modeltype == 'VGG16':
        model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=inputshape, pooling=None)
        freeze_layers(model, n_layers_to_freeze)
        model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        print('Model: VGG 16, weights loaded!')
    elif modeltype == 'InceptionV3':
        model = InceptionV3(include_top=False, weights=None,input_shape=inputShape)
        freeze_layers(model, n_layers_to_freeze)
        model.load_weights('ModelWeights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
        print('Model:InceptionV3, weights loaded!')
    elif modeltype == 'VGG19':
        model = VGG19(include_top=False, weights=None, input_tensor=None, input_shape=inputshape, pooling=None)
        freeze_layers(model, n_layers_to_freeze)
        model.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        print('Model: VGG 19, weights loaded!')
    else:
        pass

    return model

# Add the fully-connected layers whose weights will be learned during training
def fine_tune(basemodel, method):
    if method == 0:
        x = basemodel.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.7)(x)
        predictions = Dense(196, activation='softmax')(x)
        model = Model(inputs=basemodel.input, outputs=predictions)
        print('VGG fine tune, success!')
        return model
    elif method == 1:
        x = basemodel.output
        x = Flatten()(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.01, center=True,scale=True)(x)
        predictions = Dense(196, activation='softmax')(x)
        model = Model(inputs=basemodel.input, outputs=predictions)
        print('Inception V3 fine tune, success!')
        return model
    else:
        return basemodel

# Encode the y vector to be one-hot matrix: (dimensions: len(y) X 196)
def encode(y):
    temp = np.zeros((len(y), 196))
    for count in range(len(y) - 1):
        temp[count][y[count] - 1] = 1
    return temp

# ---------------------------------------------------------------------------------------
# Model parameters
inputShape = (224,224, 3)
learningRate = 1e-4
modelType = 'VGG16'
batchSize = 20
epochs = 85

def top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5) 

def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

sgd = optimizers.SGD(lr=learningRate, decay=0, momentum=0.9, nesterov=False)
adam = optimizers.Adam(lr = learningRate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 1e-4, amsgrad = False)

# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    baseModel = model_define(modelType, inputShape, 17)
    model = fine_tune(baseModel, 0)
    # THE DEFAULT VALUE 0 HERE IS CORRESPONDING TO THE VGG NETWORK

    # Compile model with categorical cross-entropy loss function, Adam optimizer, and top 1, top 3, and top 5 accuracy metrics
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy', top3, top5])
    print('Model compiled!')
    
    # Load the data
    print('Loading X_train')
    X_train = np.load('../data_224_aug3/X_train_224.npy')

    print('Loading Y_train') 
    Y_train = np.load('../data_224_aug3/Y_train_224.npy')
    print(len(Y_train))
    Y_train = encode(Y_train)

    X_dev = np.load('../data_224_aug3/X_dev_224.npy')
    Y_dev = np.load('../data_224_aug3/Y_dev_224.npy')
    Y_dev = encode(Y_dev)

    train_gen = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.1, rescale = 1./255, horizontal_flip = True, fill_mode = 'nearest')
    dev_gen = ImageDataGenerator(rescale = 1./255)
    
    print(model.summary())
    history = model.fit_generator(train_gen.flow(X_train, Y_train, shuffle = True, batch_size = batchSize), steps_per_epoch = len(X_train)//batchSize, epochs = epochs, validation_data = dev_gen.flow(X_dev, Y_dev, shuffle = True, batch_size = batchSize), validation_steps = len(X_dev)//batchSize, verbose = 2)
    np.save('Model_History.npy', history.history)
    model.save('model.h5')
