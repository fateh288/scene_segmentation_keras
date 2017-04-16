from __future__ import print_function
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Reshape
from keras.utils.visualize_util import plot
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
import cv2

height=480
width=960
class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
path = './'
data_shape = height*width
nb_epoch = 100
batch_size = 1

def normalized(rgb):
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def binarylab(labels):
    x = np.zeros([width,height,12])
    for i in range(width):
        for j in range(height):
            x[i,j,labels[i][j]]=1
    return x

def prep_data():
    train_data = []
    train_label = []
    with open('./train.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        print(path + txt[i][0])
        print(path + txt[i][1])
        train_data.append(normalized(cv2.imread(path + txt[i][0])))
        #cv2.imshow('labels',cv2.imread(path + txt[i][1]))
        train_label.append(binarylab(cv2.imread(path + txt[i][1])))
        print('.',end='')
    return np.array(train_data), np.array(train_label)

model=Sequential()
#Encoding
model.add(Convolution2D(12,3,3,border_mode='same',input_shape=(width,height,3)))
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

model.add(Convolution2D(12,3,3,border_mode='same'))
model.add(Reshape((data_shape,12), input_shape=(width,height,12)))
model.add(Activation('softmax'))

sgd=SGD(lr=0.1,momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

plot(model, to_file='model.png',show_shapes=True,show_layer_names=True)

train_data, train_label = prep_data()
train_label = np.reshape(train_label,(1,data_shape,12))
print(train_data.shape)

early_stopping1=ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',monitor='val_loss',save_best_only=True)
early_stopping2= EarlyStopping(monitor='val_loss', patience=20,min_delta=0.0005)
history = model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(train_data, train_label), class_weight=class_weighting,callbacks=[early_stopping1,early_stopping2])

model.save('model_1.h5')
model.save_weights('model_1_weight.hdf5')
