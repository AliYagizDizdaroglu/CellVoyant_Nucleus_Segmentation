
#%% Import Libraries

import os
import numpy as np
from skimage.transform import resize
from skimage import io
import cv2

main_path = r"C:\Users\firat\Desktop\Yagiz"

os.chdir(main_path)

#%% Images to Numpy

path = r"C:\Users\firat\Desktop\Yagiz\TrainVal"
os.chdir(path)
dosya = os.listdir(path)

boyut_orj = 1228
boyut = 768

sayi = len(dosya)

images = np.zeros((sayi*4,boyut,boyut,1),dtype=np.float16)
i = 0
for data in dosya:
    i = i + 1
    im = io.imread(data)
    im2 = resize(im, (boyut_orj, boyut_orj)) # Normalization
    
    new_img = np.zeros([boyut*2,boyut*2,1])
    new_img[0:boyut_orj,0:boyut_orj,0] = im2
    
    im_1 = new_img[0:768,0:768,:]
    im_2 = new_img[0:768,768:1536,:]
    im_3 = new_img[768:1540,0:768,:]
    im_4 = new_img[768:1536,768:1536,:]
    
    images[4*i-4,:,:,0] = im_1[:,:,0]
    images[4*i-3,:,:,0] = im_2[:,:,0]
    images[4*i-2,:,:,0] = im_3[:,:,0]
    images[4*i-1,:,:,0] = im_4[:,:,0]

os.chdir(main_path)

np.save('train_x', images)

#%% Labels to Numpy

path = r"C:\Users\firat\Desktop\Yagiz\TrainVal_Masks"
os.chdir(path)
dosya_2 = os.listdir(path)

boyut_orj = 1228
boyut = 768

sayi = len(dosya_2)

labels = np.zeros((sayi*4,boyut,boyut,2),dtype=np.uint8)

i = 0
for data in dosya_2:
    i = i + 1
    im = io.imread(data)  
    
    new_label = np.zeros([boyut*2,boyut*2])
    new_label[0:boyut_orj,0:boyut_orj] = im
    
    new_label = new_label > 0.5
    T = new_label == 1
    F = new_label == 0
    
    label_1 = new_label[0:768,0:768]
    label_2 = new_label[0:768,768:1536]
    label_3 = new_label[768:1540,0:768]
    label_4 = new_label[768:1536,768:1536]
    
    labels[4*i-4,:,:,0] = label_1 == 1
    labels[4*i-4,:,:,1] = label_1 == 0
    
    labels[4*i-3,:,:,0] = label_2 == 1
    labels[4*i-3,:,:,1] = label_2 == 0
    
    labels[4*i-2,:,:,0] = label_3 == 1
    labels[4*i-2,:,:,1] = label_3 == 0
    
    labels[4*i-1,:,:,0] = label_4 == 1
    labels[4*i-1,:,:,1] = label_4 == 0
  
os.chdir(main_path)

np.save('train_y',labels)

#%% Go to main directory

os.chdir(main_path)

#%% Numpy Load

train_x= np.load('train_x.npy')
train_y= np.load('train_y.npy')

#%% Model Create

import os
from keras.layers.core import Activation
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D

model = 'standart_unet'

shape = (768, 768, 1)
class_number = 2

def model_create(input_shape=shape,
                 num_classes=class_number):
    inputs = Input(shape=input_shape)
    # 512

    down0a = Conv2D(64, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(64, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(128, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(128, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(256, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(256, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(512, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(512, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    center = Conv2D(1024, (3, 3), padding='same')(down2_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down2, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down1, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down0, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down0a, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    # classify
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    return model

network = model_create(input_shape=shape, num_classes=class_number)
network.summary()

#%% Model Save

model_file = model + '.json'

model_json = network.to_json()
with open(model_file, 'w') as json_file:
    json_file.write(model_json)
    
#%% Model Load

from keras.models import model_from_json

model = 'standart_unet'

path = model + '.json'

json_file = open(path, 'r')
loaded_model_json = json_file.read()
json_file.close()
network = model_from_json(loaded_model_json)

#%% Training

from keras.callbacks import ModelCheckpoint
import pandas as pd
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Hyper-parameters
opt = Adam(lr=0.001, beta_1=0.5)
loss_func = "binary_crossentropy"
network.compile(loss=loss_func, optimizer=opt, metrics=["accuracy"])
nb_epoch = 50
batch_size = 1

# Chechpoint
filepath=model + 'weights--{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)
callbacks_list = [checkpoint]

# Training
history = network.fit(train_x, train_y, 
                 batch_size=batch_size, 
                 callbacks=callbacks_list,
                 epochs=nb_epoch, verbose=1,
                 validation_split = 0.2
                 )

#%% Save Results (Images)

from skimage.morphology import dilation
from skimage.morphology import disk

network.load_weights('standart_unetweights--27.hdf5')

path = r"C:\Users\firat\Desktop\Yagiz\Test"
os.chdir(path)
test_list = os.listdir(path)

boyut_orj = 1228
boyut = 768    
    
for name in test_list:
    
    im = io.imread(name)
    im = resize(im, (boyut_orj, boyut_orj)) # Normalization
    
    new_img = np.zeros([boyut*2,boyut*2,1])
    new_img[0:boyut_orj,0:boyut_orj,0] = im
    
    im_1 = new_img[0:768,0:768,:]
    im_2 = new_img[0:768,768:1536,:]
    im_3 = new_img[768:1540,0:768,:]
    im_4 = new_img[768:1536,768:1536,:]
    
    A = np.zeros((1,768,768,1))
    A[0,:,:,:] = im_1
    B = network.predict(A)
    C = B[:,:,:,0]
    D = C>0.5
    D_1 = D[0,:,:]
    
    A = np.zeros((1,768,768,1))
    A[0,:,:,:] = im_2
    B = network.predict(A)
    C = B[:,:,:,0]
    D = C>0.5
    D_2 = D[0,:,:]
    
    A = np.zeros((1,768,768,1))
    A[0,:,:,:] = im_3
    B = network.predict(A)
    C = B[:,:,:,0]
    D = C>0.5
    D_3 = D[0,:,:]
    
    A = np.zeros((1,768,768,1))
    A[0,:,:,:] = im_4
    B = network.predict(A)
    C = B[:,:,:,0]
    D = C>0.5
    D_4 = D[0,:,:]
    
    sonuc = np.zeros([boyut*2,boyut*2])
    
    sonuc[0:768,0:768] = D_1
    sonuc[0:768,768:1536] = D_2
    sonuc[768:1536,0:768] = D_3
    sonuc[768:1536,768:1536] = D_4
    
    sonuc = sonuc[0:boyut_orj,0:boyut_orj]
    
    x = 1
    selem = disk(x)
    sonuc = dilation(sonuc, selem)
    
    isim = 'Result_' + 'U-Net' + '_' + name + '.jpg'
    cv2.imwrite(isim, 255*sonuc)

#%% Statistics

