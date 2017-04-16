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

plot(f1, to_file='model.png',show_shapes=True,show_layer_names=True)

class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
path = '/media/fateh/01D2023161FD29C0/manas/DeepLearning/Segnet/segnet-master'
data_shape = 360*480
nb_epoch = 5000
batch_size = 6

def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def binarylab(labels):
    x = np.zeros([360,480,12])
    for i in range(360):
        for j in range(480):
            x[i,j,labels[i][j]]=1
    return x

def prep_data():
    train_data = []
    train_label = []
    with open('/media/fateh/01D2023161FD29C0/manas/DeepLearning/Segnet/segnet-master/CamVid/train.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        train_data.append(normalized(cv2.imread(path + txt[i][0][7:])))
        train_label.append(binarylab(cv2.imread(path + txt[i][1][7:][:-1])[:,:,0]))
        print('.',end='')
    return np.array(train_data), np.array(train_label)

train_data, train_label = prep_data()
print(train_data.shape)
train_label = np.reshape(train_label,(1,data_shape,12))

early_stopping1=ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',monitor='val_loss',save_best_only=True)
early_stopping2= EarlyStopping(monitor='val_loss', patience=20,min_delta=0.0005)
history = f1.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(train_data, train_label), class_weight=class_weighting,callbacks=[early_stopping1,early_stopping2])

f1.save('model_1.h5')
f1.save_weights('model_1_weight.hdf5')
