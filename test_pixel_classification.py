from __future__ import print_function
from keras.layers import Convolution2D
from keras.applications.vgg16 import VGG16
from keras.utils.visualize_util import plot
from keras.layers import UpSampling2D
from keras.models import Model
from keras.layers import Activation
from keras.layers import Input
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

m1=Input(shape=(360,480,3))
model=VGG16(include_top=False,input_tensor=m1)

model=UpSampling2D((2,2))(model.output)
#model=ZeroPadding2D(padding=(2,2))(model)
model=Convolution2D(512, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)
model=Convolution2D(512, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)
model=Convolution2D(512, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)


model=UpSampling2D((2,2))(model)
#model=ZeroPadding2D(padding=(2,2))(model)
model=Convolution2D(256, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)
model=Convolution2D(256, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)
model=Convolution2D(256, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)

#model=ZeroPadding2D(padding=(1,1))(model)

model=UpSampling2D((2,2))(model)
#model=ZeroPadding2D(padding=(2,2))(model)
model=Convolution2D(256, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)
model=Convolution2D(256, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)
model=Convolution2D(256, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)

model=UpSampling2D((2,2))(model)
model=Convolution2D(128, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)
model=Convolution2D(128, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)


model=UpSampling2D((2,2))(model)
#model=ZeroPadding2D(padding=(2,2))(model)
model=Convolution2D(64, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)
model=Convolution2D(64, 3, 3,border_mode='same',activation='relu')(model)
model=BatchNormalization()(model)

model=Convolution2D(12, 1, 1,border_mode='same',activation='relu')(model)#now data shape= 360,480
model=ZeroPadding2D(padding=(4,0))(model)

model=Reshape((360*480,12), input_shape=(360,480,12))(model)

model=Activation('softmax')(model)
f1=Model(input=m1,output=model)

sgd=SGD(lr=0.1,momentum=0.9)
f1.compile(loss='categorical_crossentropy', optimizer=sgd)

f1.load_weights('weights.2103-0.07.hdf5')

path = '/home/projectmanas/Segnet'

with open('/media/fateh/01D2023161FD29C0/manas/DeepLearning/Segnet/segnet-master/CamVid/custom_test.txt') as f:
    txt = f.readlines()
    print(txt)
    txt[0] = txt[0].strip('\n')
    print(txt)

print(txt)
inp=cv2.imread(txt[0])
print(inp)
inp2=np.reshape(inp,(1,360,480,3))

sgd=SGD(lr=0.1,momentum=0.9)
#f1.compile(loss='categorical_crossentropy', optimizer=sgd)

output = f1.predict([inp2],batch_size=1)
print(output)
print(output.shape)
pred = visualize(np.argmax(output[0],axis=1).reshape((360,480)),plot=False)
print(pred[0])
cv2.imshow('pred',pred)
cv2.imshow('image',inp)
cv2.waitKey(0)
