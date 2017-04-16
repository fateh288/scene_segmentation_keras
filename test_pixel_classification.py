from __future__ import print_function
from keras.layers import Convolution2D
from keras.applications.vgg16 import VGG16
from keras.utils.visualize_util import plot
from keras.layers import UpSampling2D
from keras.models import Sequential
from keras.models import Model
from keras.layers import Activation
from keras.layers import Input
from keras.layers.pooling import MaxPooling2D
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
import cv2

def predict_classes(self, x, batch_size=32, verbose=1):
    '''Generate class predictions for the input samples
    batch by batch.
    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.
    # Returns
        A numpy array of class predictions.
    '''
    proba = self.predict(x, batch_size=batch_size, verbose=verbose)
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
#Unlabelled = [0,0,0]

label_colours = np.array([Sky, Building, Pole,Road_marking, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist])

def visualize(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,11):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    if plot:
        plt.imshow(rgb)
    else:
        return rgb


height=720
width=960
nb_classes=21
data_shape=width*height

model=Sequential()
#Encoding
model.add(Convolution2D(nb_classes,3,3,border_mode='same',input_shape=(height,width,3)))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Decoding
model.add(UpSampling2D(size=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(UpSampling2D(size=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(UpSampling2D(size=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(UpSampling2D(size=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(nb_classes,3,3,border_mode='same'))
model.add(Reshape((data_shape,nb_classes), input_shape=(height,width,nb_classes)))
model.add(Activation('softmax'))


sgd=SGD(lr=0.1,momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.load_weights('model_scene_seg_weight.hdf5')

#path = '/home/projectmanas/Segnet'

with open('test.txt') as f:
    txt = f.readlines()
    print(txt)
    txt[0] = txt[0].strip('\n')
    print(txt)

print(txt)
inp=cv2.imread(txt[0])
print(inp)
inp2=np.reshape(inp,(1,720,960,3))

sgd=SGD(lr=0.1,momentum=0.9)
#f1.compile(loss='categorical_crossentropy', optimizer=sgd)

output = model.predict([inp2],batch_size=1)
print(output)
print(output.shape)
pred = visualize(np.argmax(output[0],axis=1).reshape((height,width)),plot=False)
print(pred[0])
cv2.imshow('pred',pred)
cv2.imshow('image',inp)
cv2.waitKey(0)
