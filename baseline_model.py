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

# Freeze pre-trained network layers
def freeze_layers(model):
    for layer in model.layers:
        layer.trainable = False

# Load pre-trained weights for VGG16, VGG19, and Inception V3 models
def model_define(modeltype, inputshape):
    if modeltype == 'VGG16':
        model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=inputshape, pooling=None)
        freeze_layers(model)
        model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        print('Model: VGG 16, weights loaded!')
    elif modeltype == 'InceptionV3':
        model = InceptionV3(include_top=False, weights=None,input_shape=inputShape)
        freeze_layers(model)
        model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
        print('Model:InceptionV3, weights loaded!')
    elif modeltype == 'VGG19':
        model = VGG19(include_top=False, weights=None, input_tensor=None, input_shape=inputshape, pooling=None)
        freeze_layers(model)
        model.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        print('Model: VGG 19, weights loaded!')
    else:
        pass
    return model

# Fine-tune the model for our classification task
def fine_tune(basemodel, method):
    if method == 0: #VGG models
        x = basemodel.output
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        predictions = Dense(196, activation='softmax')(x)
        model = Model(inputs=basemodel.input, outputs=predictions)
        print('VGG fine tune, success!')
        return model
    elif method == 1: #Inception model
        x = basemodel.output
        x = Flatten()(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.01, center=True,scale=True)(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(196, activation='softmax')(x)
        model = Model(inputs=basemodel.input, outputs=predictions)
        print('Inception V3 fine tune, success!')
        return model
    else:
        return basemodel

# Generate mini_batch with a random sample of the data
def make_batch(X,y, batchsize):
    n = len(y)
    slice = random.sample(range(n),batchsize)
    return X[slice],y[slice]

# Encode y values as a 196-dimensional vector
def encode(y):
    temp = np.zeros((len(y), 196))
    # print(temp.shape)
    # print(range(len(y) - 1))
    for count in range(len(y) - 1):
        temp[count][y[count] - 1] = 1

    # print(temp.shape)
    return temp

# Define parameters
inputShape = (224,224, 3)
learningRate = 0.005
modelType = 'VGG16'
batchSize = 16
epochs = 10

sgd = optimizers.SGD(lr=learningRate, decay=1e-4, momentum=0.5, nesterov=False)
adam = optimizers.Adam()


if __name__ == '__main__':
    baseModel = model_define(modelType, inputShape)
    model = fine_tune(baseModel, 0) # 0 corresponds to VGG
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print('Model compiled!')

    # Load the data
    data_dir = '/stanford-cars/data_224/'
    X_train = np.load('data_224/X_train_224.npy')
    X_train = X_train/255
    Y_train = np.load('data_224/Y_train_224.npy')
    Y_train = encode(Y_train)

    X_dev = np.load('data_224/X_dev_224.npy')
    X_dev = X_dev/255
    Y_dev = np.load('data_224/Y_dev_224.npy')
    Y_dev = encode(Y_dev)

    # Fit model, report accuracy at each epoch, and save the final weights
    print(model.summary())
    history = model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=epochs, validation_data = (X_dev, Y_dev), verbose=2)
    np.save('Model_History.npy', history.history)
    model.save('Baseline.h5')
