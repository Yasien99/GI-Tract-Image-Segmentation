import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image 
#%matplotlib inline
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.notebook import tqdm
from datetime import datetime
import json,itertools
from typing import Optional
from glob import glob

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

from tensorflow import keras
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import load_model

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib as mpl
#plt.style.use('ggplot')


BATCH_SIZE = 16
EPOCHS=10
n_splits=5
fold_selected=2# 1..10
# fold_selected_test=[1,5]

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


def build_masks(labels,input_shape, colors=True):
    height, width = input_shape
    if colors:
        mask = np.zeros((height, width, 3))
        for label in labels:
            mask += rle_decode(label, shape=(height,width , 3), color=np.random.rand(3))
    else:
        mask = np.zeros((height, width, 1))
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 1))
    mask = mask.clip(0, 1)
    return mask

import keras
# we need to inherit from tf.keras.utils.Sequence to generat data
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size = BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.on_epoch_end()
# if we call the len() on the object this function will be called
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
# after the end of every epoch this function will be called  
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
# when the model.fit() called it call this function until it finish all the data
# we can call this function also by putting [] after the object 
    def __getitem__(self, index): 
        # make an empty holder for the image batch and masks batch
        X = np.empty((self.batch_size,128,128,3))
        y = np.empty((self.batch_size,128,128,3))
        # get the index of the wanted batch 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # get the path of every index to read the image
        for i,img_path in enumerate(self.df['path'].iloc[indexes]):
            # get the width and hight to decode the mask 
            w=self.df['width'].iloc[indexes[i]]
            h=self.df['height'].iloc[indexes[i]]
            # read the image as gray scale
            img = self.__load_grayscale(img_path)
            # putting the batch holder
            X[i] =img
            # here we decode the mask and put in rgb image 
            if self.subset == 'train':
                for k,j in zip([0,1,2],["large_bowel","small_bowel","stomach"]):
                    rles=self.df[j].iloc[indexes[i]]
                    masks = rle_decode(rles, shape=(h, w, 1))
                    #rles=df_train[j][df_train.index==indexes[i]]
                    #masks = build_masks(rles,(h,w), colors=False)
                    masks = cv2.resize(masks, (128, 128))
                    y[i,:,:,k] = masks
                    
        if self.subset == 'train': return X, y
        else: return X
       
    
    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        dsize = (128, 128)
        img = cv2.resize(img, dsize)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)
        # normlize the image between 0 &1 
        img =(img - img.min())/(img.max() - img.min())
        
        return img
        

import keras
# we need to inherit from tf.keras.utils.Sequence to generat data
class DataGenerator1D(tf.keras.utils.Sequence):
    def __init__(self,channel, df, batch_size = BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df
        self.channel = channel
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.on_epoch_end()
# if we call the len() on the object this function will be called
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
# after the end of every epoch this function will be called  
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
# when the model.fit() called it call this function until it finish all the data
# we can call this function also by putting [] after the object 
    def __getitem__(self, index): 
        # make an empty holder for the image batch and masks batch
        X = np.empty((self.batch_size,128,128,3))
        y = np.empty((self.batch_size,128,128,1))
        # get the index of the wanted batch 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # get the path of every index to read the image
        for i,img_path in enumerate(self.df['path'].iloc[indexes]):
            # get the width and hight to decode the mask 
            w=self.df['width'].iloc[indexes[i]]
            h=self.df['height'].iloc[indexes[i]]
            # read the image as gray scale
            img = self.__load_grayscale(img_path)
            # putting the batch holder
            X[i] =img
            # here we decode the mask and put in rgb image 
            if self.subset == 'train':
                for k,j in zip([0,1,2],["large_bowel","small_bowel","stomach"]):
                    if self.channel == k:
                        rles=self.df[j].iloc[indexes[i]]
                        masks = rle_decode(rles, shape=(h, w,1))
                        #rles=df_train[j][df_train.index==indexes[i]]
                        #masks = build_masks(rles,(h,w), colors=False)
                        masks = cv2.resize(masks, (128, 128))
                        masks = np.expand_dims(masks, axis=-1)
                        y[i,:,:] = masks
                    
        if self.subset == 'train': return X, y
        else: return X
       
    
    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        dsize = (128, 128)
        img = cv2.resize(img, dsize)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)
        # normlize the image between 0 &1 
        img =(img - img.min())/(img.max() - img.min())
        
        return img
        
