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
# ------------------------------------------------------------------------------

def freeze_layers(model, n_layers_to_freeze):
    print(len(model.layers))
    for i, layer in enumerate(model.layers):
        if i < n_layers_to_freeze:
            layer.trainable = False
        else:
            layer.trainable = True

def model_define(modeltype, inputshape, n_layers_to_freeze):
    if modeltype == 'VGG16':
        model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=inputshape, pooling=None)
        freeze_layers(model, n_layers_to_freeze)
        model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        # model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
        # model.layers[-1].kernel_initializer = initializers.glorot_normal()
        # model.layers.append(Dense(4096, activation = 'relu'))
        # model.layers.append(Dense(4096, activation='relu'))
        # model.layers.append(Dense(196, activation='softmax'))
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

    # model.layers[-1].trainable = True
    # model.layers[-1] = Dense(196,activation = 'softmax')
    return model


def fine_tune(basemodel, method):
    # Some adjustments can be made in this function
    if method == 0:
        x = basemodel.output
        x = Flatten()(x)
        # x = BatchNormalization(axis=-1, epsilon=0.001, center=True, scale=True)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.8)(x)
       # x = Dense(4096, activation='relu')(x)
        # x = Dropout(0.2)(x)
        # x = BatchNormalization(axis=-1, epsilon=0.001, center=True, scale=True)(x)
        predictions = Dense(196, activation='softmax')(x)
        model = Model(inputs=basemodel.input, outputs=predictions)
        print('VGG fine tune, success!')
        return model
    elif method == 1:
        x = basemodel.output
        x = Flatten()(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.01, center=True,scale=True)(x)
        # x = Dense(256, activation='relu')(x)
	#x = Dropout(0.2)(x)
        predictions = Dense(196, activation='softmax')(x)
        model = Model(inputs=basemodel.input, outputs=predictions)
        print('Inception V3 fine tune, success!')
        return model
    else:
        return basemodel

def encode(y):
    temp = np.zeros((len(y), 196))
    # print(temp.shape)
    # print(range(len(y) - 1))
    for count in range(len(y) - 1):
        temp[count][y[count] - 1] = 1

    # print(temp.shape)
    return temp


# ---------------------------------------------------------------------------------------
inputShape = (224,224, 3)
learningRate = 1e-4
modelType = 'VGG16'
# A valid value for 3 Dense FC layers is 0.005
# A valid learning rate for 3fc with BN layers is 0.05

batchSize = 20
epochs = 100
# One valid number of epochs of 3 FC layers is 25
# print(model.layers[16].trainable)
# ---------------------------------------------------------------------------------------
def top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5) 

def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

sgd = optimizers.SGD(lr=learningRate, decay=0, momentum=0.9, nesterov=False)
adam = optimizers.Adam(lr = learningRate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 1e-4, amsgrad = False)

# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    baseModel = model_define(modelType, inputShape, 17) # freezes everything up until the last block
    model = fine_tune(baseModel, 0)
    # THE DEFAULT VALUE 0 HERE IS CORRESPONDING TO THE VGG NETWORK

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy', top3, top5])
    print('Model compiled!')
    
    # Load the data
    print('Loading X_train')
    X_train = np.load('../data_224_aug3/X_train_224.npy')
    #X_train = X_train/255

    print('Loading Y_train') 
    Y_train = np.load('../data_224_aug3/Y_train_224.npy')
    print(len(Y_train))
    Y_train = encode(Y_train)

    X_dev = np.load('../data_224_aug3/X_dev_224.npy')
    #X_dev = X_dev/255
    Y_dev = np.load('../data_224_aug3/Y_dev_224.npy')
    Y_dev = encode(Y_dev)

    train_gen = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.1, rescale = 1./255, horizontal_flip = True, fill_mode = 'nearest')
    dev_gen = ImageDataGenerator(rescale = 1./255)
    
    print(model.summary())
    model.fit_generator(train_gen.flow(X_train, Y_train, shuffle = True, batch_size = batchSize), steps_per_epoch = len(X_train)//batchSize, epochs = epochs, validation_data = dev_gen.flow(X_dev, Y_dev, shuffle = True, batch_size = batchSize), validation_steps = len(X_dev)//batchSize, verbose = 2)
    model.save('datagen_test.h5')
