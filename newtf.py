Jupyter Notebook
gtxDLClassAWS.py
Yesterday at 12:53 PM
Python
File
Edit
View
Language
1
from __future__ import print_function
2
from dicttoxml import dicttoxml
3
from xml.dom.minidom import parseString
4
from sklearn import metrics
5
from csv import writer, reader
6
import xml.etree.ElementTree as ET
7
import matplotlib.pyplot as plt
8
import matplotlib
9
import numpy as np, h5py
10
import os, time, sys
11
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
12
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
13
import scipy.io
14
import pandas as pd
15
from skimage.metrics import structural_similarity as ssim
16
​
17
# Tensorflow / Keras
18
import tensorflow as tf
19
from tensorflow import keras
20
from keras.models import Model, load_model
21
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout
22
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
23
from keras.preprocessing import image
24
​
25
from gtxDLClassAWSUtils import Utils
26
​
27
import boto3 
28
import io
29
import openpyxl
30
​
31
​
32
class DL(Utils):    
33
    # Initialization method runs whenever an instance of the class is initiated
34
    def __init__(self):
35
        self.bucket = '20240909-hikaru'
36
        isCase = input('Choose a case:\n Default\n Default4fx')
37
        self.case = isCase
38
        self.params = self.Defaults(self.case) # Define some default parameters based on the selected case
39
        self.isTesting = False
40
​
41
        self.Model()
42
        return None  
43
    
44
    def Train(self):
45
        """The Train method is designed to guide the user through the process of training a deep neural network; i.e., reading and scaling training data, modelling, fitting, plotting, etc."""
46
        self.importData(isTesting=False,quickTest=False)
