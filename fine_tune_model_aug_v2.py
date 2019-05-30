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
# ------------------------------------------------------------------------------

def freeze_layers(model):
    for layer in model.layers:
        layer.trainable = False


def model_define(modeltype, inputshape):
    if modeltype == 'VGG16':
        model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=inputshape, pooling=None)
        freeze_layers(model)
        model.load_weights('ModelWeights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        # model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
        # model.layers[-1].kernel_initializer = initializers.glorot_normal()
        # model.layers.append(Dense(4096, activation = 'relu'))
        # model.layers.append(Dense(4096, activation='relu'))
        # model.layers.append(Dense(196, activation='softmax'))
        print('Model: VGG 16, weights loaded!')
    elif modeltype == 'InceptionV3':
        model = InceptionV3(include_top=False, weights=None,input_shape=inputShape)
        freeze_layers(model)
        model.load_weights('ModelWeights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
        print('Model:InceptionV3, weights loaded!')
    elif modeltype == 'VGG19':
        model = VGG19(include_top=False, weights=None, input_tensor=None, input_shape=inputshape, pooling=None)
        freeze_layers(model)
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
        x = Dropout(0.2)(x)
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


def make_batch(X,y, batchsize):
    n = len(y)
    slice = random.sample(range(n),batchsize)
    return X[slice],y[slice]


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
learningRate = 0.0005
modelType = 'VGG16'
# A valid value for 3 Dense FC layers is 0.005
# A valid learning rate for 3fc with BN layers is 0.05

batchSize = 16
epochs = 100
# One valid number of epochs of 3 FC layers is 25
# print(model.layers[16].trainable)
# ---------------------------------------------------------------------------------------

sgd = optimizers.SGD(lr=learningRate, decay=0, momentum=0.9, nesterov=False)
adam = optimizers.Adam()

# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    baseModel = model_define(modelType, inputShape)
    model = fine_tune(baseModel, 0)
    # THE DEFAULT VALUE 0 HERE IS CORRESPONDING TO THE VGG NETWORK

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print('Model compiled!')
    
    # Load the data
    print('Loading X_train')
    X_train = np.load('data_224_aug3/X_train_224.npy')
    X_train = X_train/255
    print('Loading X_flip_train')
    X_flip_train_224 = np.load('data_224_aug3/X_flip_train_224.npy')
    X_flip_train_224 = X_flip_train_224/255
    print('Loading X_noise_train')
    X_noise_train_224 = np.load('data_224_aug3/X_noise_train_224.npy')
    X_noise_train_224 = X_noise_train_224/255
    print('Loading X_rotate_train')
    X_rotate_train_224 = np.load('data_224_aug3/X_rotate_train_224.npy')
    X_rotate_train_224 = X_rotate_train_224/255 

    print('Concatenating X_trains')
    X_train = np.concatenate((X_train, X_flip_train_224), axis = 0)
    X_train = np.concatenate((X_train, X_noise_train_224), axis = 0)
    X_train = np.concatenate((X_train, X_rotate_train_224), axis = 0)    

    print('Concatenating Y_trains') 
    Y_train = np.load('data_224_aug3/Y_train_224.npy')
    Y_flip_train_224 = np.load('data_224_aug3/Y_flip_train_224.npy')
    Y_noise_train_224 = np.load('data_224_aug3/Y_noise_train_224.npy')
    Y_rotate_train_224 = np.load('data_224_aug3/Y_rotate_train_224.npy')
    Y_train = list(Y_train) + list(Y_flip_train_224) + list(Y_noise_train_224) +list(Y_rotate_train_224) + list(Y_rotate_train_224) 
    print(len(Y_train))
    Y_train = encode(Y_train)

    X_dev = np.load('data_224_aug3/X_dev_224.npy')
    X_dev = X_dev/255
    Y_dev = np.load('data_224_aug3/Y_dev_224.npy')
    Y_dev = encode(Y_dev)

    print(model.summary())
    history = model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=epochs, validation_data = (X_dev, Y_dev), verbose=2)
    np.save('Model_History.npy', history.history)
    model.save('Baseline.h5')
