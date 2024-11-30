from __future__ import print_function
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
from sklearn import metrics
from csv import writer, reader
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np, h5py
import os, time, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import scipy.io
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from matplotlib import cm


# Tensorflow / Keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout, MaxPool2D, UpSampling2D, ZeroPadding2D
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing import image

from gtxDLClassAWSUtils import Utils

import boto3 
import io

from matplotlib.colors import LightSource
#import plotly.graph_objects as go 


class DL(Utils):    
    # Initialization method runs whenever an instance of the class is initiated
    def __init__(self):
        self.bucket = '20240912-hikaru-iceberg'
        print('Choose a Default Parameters Case: \n 20221216_Case3AWS\n 20221216_Case3AWS_One3DConv\n 20230517_OPAscale\n 20230726_4SF\n 20230905_ScaledF\n 20230912_4SF_logF\n')
        self.case = input('Select a case from listed directories: ')
        self.params = self.Defaults(self.case) # Define some default parameters based on the selected case
        return None  
    
    def Train(self):
        """The Train method is designed to guide the user through the process of training a deep neural network; i.e., reading and scaling training data, modelling, fitting, plotting, etc."""
        self.importData(isTesting=False,quickTest=False)
        while True:
            callParams = input('Adjust parameters before modelling and fitting? (Y/N) ')
            if callParams in ['Y','y']:
                self.Params()
                break
            elif callParams in ['N','n']:
                break
            else:
                print('Invalid entry.')
        self.Model()
        while True:
            isTransferTrain = input('Are you transfer training (i.e., Initial weights come from another pre-trained model)? (Y/N) ')
            if isTransferTrain in ['Y','y']:
                self.Fit(isTransfer=True)
                break
            elif isTransferTrain in ['N','n']:
                self.Fit(isTransfer=False)
                break
            else:
                print('Invalid entry.')

        self.Plot(isTraining=True)
        return None
    
    def Test(self):
        """The Test method is designed to guide the user through the process of testing the performance of a deep neural network; i.e., reading and scaling testing data, making predictions, and evaluating these predictions."""
        for attr in ['FL','OP','DF','QF']:
            delattr(self,attr) # Delete the training data so that we can load the testing data
        self.importData(isTesting=True,quickTest=False)
        self.Predict()
        self.Evaluate()
        return None
    
    def Params(self):
        print(self.params.keys()) # Print the keys in the parameters dictionary so that the user knows their options
        print('\nChoose a key (above) to change the value of that parameter. Enter nothing to escape.\n')
        while True: # Until we are done changing parameters
            key = input('Key: ') # User inputs one of the key values
            if key == '': # User enters nothing; break
                break
            elif key  in ['optimizer']: # If user chooses optimizer key
                print('Current value of '+key+ ' is '+str(self.params[key])+'. Options for optimizers are: SGD; RMSprop; Adam; Adadelta ;Adagrad ;Adamax ; Nadam ; Ftrl. Optimizers are case sensitive.' ) # Output current value and options
                self.params[key] = input('Value: ') # Update current value based on choice
                break
            elif key in ['activation']:
                print('Current value of '+key+ ' is '+str(self.params[key]) ) # Output current value and options
                self.params[key] = input('Value: ') # Update current value based on choice
                break
            elif key in ['epochs','batch','nFilters3D','nF','xX','yY','BNAct', 'nFilters2D']: # If user chooses one of the integer valued keys
                print('Current value of '+key+ ' is '+str(self.params[key])+' ')
                value = input('Value: ') 
                self.params[key] = int(value) # Input is string, so we need to convert to int
                break
            elif key in ['learningRate','scaleFL','scaleOP0','scaleOP1','scaleDF','scaleQF']: # If user chooses one of the float valued keys
                print('Current value of '+key+ ' is '+str(self.params[key])+' ')
                value = input('Value: ')
                self.params[key] = float(value) # Input is string, so we need to conver to float
                break
            elif key in ['kernelConv3D','strideConv3D','kernelResBlock3D','kernelConv2D','kernelResBlock2D', 'strideConv2D']: # If user chooses one of the tuple valued keys
                print('Current value of '+key+ ' is '+str(self.params[key])+' ')
                value = input('Value (without using brackets): ')
                self.params[key] = tuple(map(int, value.split(','))) # Convert csv string to tuple
                break
            else: # Entry is not one of the listed keys
                print('Key does not exist; valid key values are printed in the dictionary above. Enter nothing to finish. ')
                break
        return None    
    
    def readData(self):
        s3_client = boto3.client('s3')
        # Ask user to input if they are testing are not - indicates if you need to flip
        isTesting = input('Are you testing? (Y/N)')
        if isTesting in ['Y','y']:
            isTesting = True
            # Print out the contents of the bucket (i.e., options for importing)           
            for key in s3_client.list_objects_v2(Bucket=self.bucket)['Contents']: 
                data = key['Key']
                # Display only matlab files
                if data.find("TestingData")!=-1:
                    if data.find(".mat")!=-1:
                        print(data)
        elif isTesting in ['N','n']:
            isTesting = False
            for key in s3_client.list_objects_v2(Bucket=self.bucket)['Contents']: 
                data = key['Key']
                # Display only matlab files
                if data.find("TrainingData")!=-1:
                    if data.find(".mat")!=-1:
                        print(data)
        else:
            print('Invalid option')

        # Enter the name of the dataset you want to import
        # Note: To import new data, go to the desired bucket in AWS and upload data
        self.file_key = str(input('Enter the name of the dataset you want to import e.g. matALL.mat '))
        obj = s3_client.get_object(Bucket=self.bucket, Key=self.file_key)
        dataTemp = obj['Body'].read()
        self.dataset = h5py.File(io.BytesIO(dataTemp))
        
        self.FL = self.dataset['F']
        self.DF = self.dataset['DF']
        self.OP = self.dataset['OP']
        self.QF = self.dataset['QF']
        
        # Check whether the user is using the single or multiple MAT format 
        # I.e., looking at individual MAT files (getDims=3) or looking at MAT files with more than one sample (getDim=4)
        getDims = len(np.shape(self.FL))
        if getDims == 4:
            numSets = int(np.shape(self.FL)[3])
            while True:
                check = input('There are ' + str(numSets) + ' inputs in this directory, use them all? (Y/N) ')
                if check in ['Y','y']:
                    break
                elif check in ['N','n']:
                    trynumSets = int(input('Input the number of sets you want: '))
                    if trynumSets <= numSets:
                        numSets = trynumSets
                    else:
                        raise Exception('The number of sets you want to use is greater than the total number of sets.')
                    break
                elif check == '':
                    break
                else:
                    print('You did not select yes/no, try again or enter nothing to escape')

            sizeFx = int(np.shape(self.FL)[0])
            while True:
                check = input('There are ' + str(sizeFx) + ' spatial frequencies in the FL input, use them all? (Y/N) ')
                if check in ['Y','y']:
                    self.params['nF'] = sizeFx
                    indxFx = np.arange(0,sizeFx,1)
                    break
                elif check in ['N','n']:
                    useFx = int(input('How many spatial frequencies would you like to use? Number must be a factor of the total number of spatial frequencies: '))
                    self.params['nF'] = useFx
                    indxFx = np.arange(0,useFx,1)
                    self.FL = self.FL[0:sizeFx:int(sizeFx/useFx),:,:,:]
                    break
                elif check == '':
                    break
                else:
                    print('You did not select yes/no, try again or enter nothing to escape')
        
            start = time.perf_counter()

            self.FL = np.reshape(self.FL[indxFx,:,:,0:numSets],(len(indxFx),self.params['xX'],self.params['yY'],numSets,1))
            self.OP = self.OP[:,:,:,0:numSets]
            self.DF = self.DF[:,:,0:numSets]
            self.DF = np.reshape(self.DF,(self.params['xX'],self.params['yY'],numSets,1))
            self.QF = self.QF[:,:,0:numSets]
            self.QF = np.reshape(self.QF,(self.params['xX'],self.params['yY'],numSets,1))

            # Reorder data
            self.FL = np.swapaxes(self.FL,0,3)
            self.OP = np.swapaxes(self.OP,0,3)
            self.DF = np.moveaxis(self.DF,2,0)
            self.QF = np.moveaxis(self.QF,2,0)
            
            stop = time.perf_counter()
            print('Load time = ' + str(stop-start))

            # Apply scale
            self.FL *= self.params['scaleFL']
            self.OP[:,:,:,0] *= self.params['scaleOP0']
            self.OP[:,:,:,1] *= self.params['scaleOP1']
            self.DF *= self.params['scaleDF']
            self.QF *= self.params['scaleQF']

            print(np.mean(self.FL))
            print(np.mean(self.OP[:,:,:,0]))
            print(np.mean(self.OP[:,:,:,1]))
            print(np.mean(self.DF))
            print(np.mean(self.QF)) 
            
            

        elif getDims ==3:
            print('There is 1 input in this directory') # Do not ask user to enter number of inputs they want to use (only have one option)
            numSets = 1
            while True:  # Get number of desired spatial frequencies from the user
                sizeFx = int(np.shape(self.FL)[0])
                check = input('There are ' + str(sizeFx) + ' spatial frequencies in the FL input, use them all? (Y/N) ')
                if check in ['Y','y']:
                    self.params['nF'] = sizeFx
                    indxFx = np.arange(0,sizeFx,1)
                    break
                elif check in ['N','n']:
                    useFx = int(input('How many spatial frequencies would you like to use? Number must be a factor of the total number of spatial frequencies: '))
                    self.params['nF'] = useFx
                    indxFx = np.arange(0,useFx,1)
                    break
                elif check == '':
                    break
                else:
                    print('You did not select yes/no, try again or enter nothing to escape')
                    
            start = time.perf_counter()


            self.FL = np.reshape(self.FL[indxFx,:,:],(len(indxFx),self.params['xX'],self.params['yY'],1,1))
            self.OP = self.OP[:,:,:]
            self.OP = self.OP.reshape((2,self.params['xX'], self.params['xX'],1))
            self.DF = self.DF[:,:]
            self.DF = np.reshape(self.DF,(1,self.params['xX'],self.params['yY'],1))
            self.QF = self.QF[:,:]
            self.QF = np.reshape(self.QF,(1,self.params['xX'],self.params['yY'],1))

            stop = time.perf_counter()
            print('Load time = ' + str(stop-start))
            
            self.FL = np.swapaxes(self.FL,0,3)
            self.OP = np.swapaxes(self.OP,0,3)
            
            # Apply scale
            self.FL *= self.params['scaleFL']
            self.OP[0,:,:,0] *= self.params['scaleOP0']
            self.OP[0,:,:,1] *= self.params['scaleOP1']
            self.DF *= self.params['scaleDF']
            self.QF *= self.params['scaleQF']

            print(np.mean(self.FL))
            print(np.mean(self.OP[:,:,:]))
            print(np.mean(self.DF))
            print(np.mean(self.QF))
            

        
        # Apply flipping and rotation for Case 2 and 3 (for irregularly-shaped test data):
        if isTesting==True:
            self.FL = np.rot90(self.FL,1,(1,2))  #rotate from axis 2 to axis 3 (first 33 (xX) to second 33 (yY))
            self.FL = np.fliplr(self.FL) #left-right flip  
            self.DF = np.rot90(self.DF,1,(1,2))  
            self.DF = np.fliplr(self.DF) 
            self.QF = np.rot90(self.QF,1,(1,2))
            self.QF = np.fliplr(self.QF) 
            self.OP = np.rot90(self.OP,1,(1,2)) 
            self.OP = np.fliplr(self.OP) 
             
                
    def importData(self,isTesting=True,quickTest=False):
        s3_client = boto3.client('s3')
        # Ask user to input if they are testing are not - indicates if you need to flip
        if isTesting == True:
            # Print out the contents of the bucket (i.e., options for importing)           
            for key in s3_client.list_objects_v2(Bucket=self.bucket)['Contents']: 
                data = key['Key']
                # Display only matlab files
                if data.find("TestingData")!=-1:
                    if data.find(".mat")!=-1:
                        print(data)
        else:
            isTesting = False
            for key in s3_client.list_objects_v2(Bucket=self.bucket)['Contents']: 
                data = key['Key']
                # Display only matlab files
                if data.find("TrainingData")!=-1:
                    if data.find(".mat")!=-1:
                        print(data)

        # Enter the name of the dataset you want to import
        # Note: To import new data, go to the desired bucket in AWS and upload data
        self.file_key = str(input('Enter the name of the dataset you want to import e.g. matALL.mat '))
        obj = s3_client.get_object(Bucket=self.bucket, Key=self.file_key)
        dataTemp = obj['Body'].read()
        self.dataset = h5py.File(io.BytesIO(dataTemp))
        self.params["training_file_name"] = self.file_key

        
        self.FL = self.dataset['F']
        self.DF = self.dataset['DF']
        self.OP = self.dataset['OP']
        self.QF = self.dataset['QF']
        
        # Check whether the user is using the single or multiple MAT format 
        # I.e., looking at individual MAT files (getDims=3) or looking at MAT files with more than one sample (getDim=4)
        getDims = len(np.shape(self.FL))
        if getDims == 4:
            numSets = int(np.shape(self.FL)[3])
            if quickTest ==False:
                while True:
                    check = input('There are ' + str(numSets) + ' inputs in this directory, use them all? (Y/N) ')
                    if check in ['Y','y']:
                        break
                    elif check in ['N','n']:
                        trynumSets = int(input('Input the number of sets you want: '))
                        if trynumSets <= numSets:
                            numSets = trynumSets
                        else:
                            raise Exception('The number of sets you want to use is greater than the total number of sets.')
                        break
                    elif check == '':
                        break
                    else:
                        print('You did not select yes/no, try again or enter nothing to escape')

                sizeFx = int(np.shape(self.FL)[0])
                while True:
                    check = input('There are ' + str(sizeFx) + ' spatial frequencies in the FL input, use them all? (Y/N) ')
                    if check in ['Y','y']:
                        self.params['nF'] = sizeFx
                        indxFx = np.arange(0,sizeFx,1)
                        break
                    elif check in ['N','n']:
                        useFx = int(input('How many spatial frequencies would you like to use? Number must be a factor of the total number of spatial frequencies: '))
                        self.params['nF'] = useFx
                        indxFx = np.arange(0,useFx,1)
                        self.FL = self.FL[0:sizeFx:int(sizeFx/useFx),:,:,:]
                        break
                    elif check == '':
                        break
                    else:
                        print('You did not select yes/no, try again or enter nothing to escape')
            else:
                numSets = int(np.shape(self.FL)[3])
                sizeFx = self.params['nF']
                indxFx = np.arange(0,sizeFx,1)
        
            start = time.perf_counter()

            self.FL = np.reshape(self.FL[indxFx,:,:,0:numSets],(len(indxFx),self.params['xX'],self.params['yY'],numSets,1))
            self.OP = self.OP[:,:,:,0:numSets]
            self.DF = self.DF[:,:,0:numSets]
            self.DF = np.reshape(self.DF,(self.params['xX'],self.params['yY'],numSets,1))
            self.QF = self.QF[:,:,0:numSets]
            self.QF = np.reshape(self.QF,(self.params['xX'],self.params['yY'],numSets,1))

            # Reorder data
            self.FL = np.swapaxes(self.FL,0,3)
            self.OP = np.swapaxes(self.OP,0,3)
            self.DF = np.moveaxis(self.DF,2,0)
            self.QF = np.moveaxis(self.QF,2,0)
            
            stop = time.perf_counter()
            print('Load time = ' + str(stop-start))

            # Apply scale
            self.OP[:,:,:,0] *= self.params['scaleOP0']
            self.OP[:,:,:,1] *= self.params['scaleOP1']
            self.DF *= self.params['scaleDF']
            self.QF *= self.params['scaleQF']
            if 'scaleFL0' in self.params:
                for i in range(self.params['nF']):
                    scaleN = 'scaleFL'+ str(i)
                    self.FL[:,:,:,i] *= self.params[scaleN]
                    print(np.mean(self.FL[:,:,:,i]))
            else:
                self.FL *= self.params['scaleFL']

            print(np.mean(self.OP[:,:,:,0]))
            print(np.mean(self.OP[:,:,:,1]))


        elif getDims ==3:
            print('There is 1 input in this directory') # Do not ask user to enter number of inputs they want to use (only have one option)
            numSets = 1
            if quickTest ==False:
                while True:  # Get number of desired spatial frequencies from the user
                    sizeFx = int(np.shape(self.FL)[0])
                    check = input('There are ' + str(sizeFx) + ' spatial frequencies in the FL input, use them all? (Y/N) ')
                    if check in ['Y','y']:
                        self.params['nF'] = sizeFx
                        indxFx = np.arange(0,sizeFx,1)
                        break
                    elif check in ['N','n']:
                        useFx = int(input('How many spatial frequencies would you like to use? Number must be a factor of the total number of spatial frequencies: '))
                        self.params['nF'] = useFx
                        indxFx = np.arange(0,useFx,1)
                        break
                    elif check == '':
                        break
                    else:
                        print('You did not select yes/no, try again or enter nothing to escape')
            else:
                sizeFx = self.params['nF']
                indxFx = np.arange(0,sizeFx,1)
                
            start = time.perf_counter()
            
            self.FL = np.reshape(self.FL[indxFx,:,:],(len(indxFx),self.params['xX'],self.params['yY'],1,1))
            self.OP = self.OP[:,:,:]
            self.OP = self.OP.reshape((2,self.params['xX'], self.params['xX'],1))
            self.DF = self.DF[:,:]
            self.DF = np.reshape(self.DF,(1,self.params['xX'],self.params['yY'],1))
            self.QF = self.QF[:,:]
            self.QF = np.reshape(self.QF,(1,self.params['xX'],self.params['yY'],1))

            stop = time.perf_counter()
            print('Load time = ' + str(stop-start))
            
            self.FL = np.swapaxes(self.FL,0,3)
            self.OP = np.swapaxes(self.OP,0,3)
            
            # Apply scale
            self.OP[0,:,:,0] *= self.params['scaleOP0']
            self.OP[0,:,:,1] *= self.params['scaleOP1']
            self.DF *= self.params['scaleDF']
            self.QF *= self.params['scaleQF']
            if 'scaleFL0' in self.params:
                for i in range(self.params['nF']):
                    scaleN = 'scaleFL'+ str(i)
                    self.FL[:,:,:,i] *= self.params[scaleN]
                    print(np.mean(self.FL[:,:,:,i]))
            else:
                self.FL *= self.params['scaleFL']            

            print(np.mean(self.FL))
            print(np.mean(self.OP[:,:,:]))
            print(np.mean(self.DF))
            print(np.mean(self.QF))
            
        
        # Apply flipping and rotation for Case 2 and 3 (for irregularly-shaped test data):
        if isTesting==True:
            self.FL = np.rot90(self.FL,1,(1,2))  #rotate from axis 2 to axis 3 (first 33 (xX) to second 33 (yY))
            self.FL = np.fliplr(self.FL) #left-right flip  
            self.DF = np.rot90(self.DF,1,(1,2))  
            self.DF = np.fliplr(self.DF) 
            self.QF = np.rot90(self.QF,1,(1,2))
            self.QF = np.fliplr(self.QF) 
            self.OP = np.rot90(self.OP,1,(1,2)) 
            self.OP = np.fliplr(self.OP) 
                     
    
    def Model(self):
        """The deep learning architecture gets defined here"""
      

        ## Input Optical Properties ##
        inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],2))
        ## Input Multi-Dimensional Fluorescence ##
        inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))

        ## NOTE: Batch normalization can cause instability in the validation loss

        ## Optical Properties Branch ##
        inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)

        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        
        print("inOP1: ", inOP.shape)
        
        ## Fluorescence Input Branch ##
        input_shape = inFL_beg.shape
        inFL = Conv2D(filters=self.params['nFilters3D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)
        

        inFL = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        print("inFL1: ", inFL.shape)

        ## Concatenate Branch ##
        concat = concatenate([inOP,inFL],axis=-1)
        print("concat: ", concat.shape)

        Max_Pool_1 = MaxPool2D()(concat)
        print("Maxpool1: ", Max_Pool_1.shape)

        Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Max_Pool_1)
        Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_1)
        
        Max_Pool_2 = MaxPool2D()(Conv_1)
        print("Max_Pool_2: ", Max_Pool_2.shape)

        Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Max_Pool_2)
        Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_2)

        Max_Pool_3 = MaxPool2D()(Conv_2)
        print("Max_Pool_3: ", Max_Pool_3.shape)

        Conv_3 = Conv2D(filters=1024, kernel_size=(self.params['kernelConv2D']), strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Max_Pool_3)
        Conv_3 = Conv2D(filters=1024, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_3)
        
        #decoder 
        Up_conv_1 = UpSampling2D()(Conv_3)
        Up_conv_1 = Conv2D(filters=512, kernel_size = (2,2), strides=(1,1), padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Up_conv_1)

        concat_1 = concatenate([Conv_2[:,0:Conv_2.shape[1] - 1, 0:Conv_2.shape[2] - 1, :],Up_conv_1],axis=-1)

        Conv_4 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(concat_1)
 
        Conv_4 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            activation=self.params['activation'], data_format="channels_last")(Conv_4)
        
        Up_conv_2 = UpSampling2D()(Conv_4)

        Up_conv_2 = Conv2D(filters=256, kernel_size = (2,2), strides=(1,1), padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Up_conv_2)
        print("Up_conv_2: ", Up_conv_2.shape)
        print("Conv_1: ", Conv_1.shape)

        Up_conv_2 = ZeroPadding2D()(Up_conv_2)

        concat_2 = concatenate([Conv_1,Up_conv_2],axis=-1)

        Conv_5 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(concat_2)
        Conv_5 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_5)
        
        Up_conv_3 = UpSampling2D()(Conv_5)
        Up_conv_3 = Conv2D(filters=128, kernel_size = (2,2), strides=(1,1), padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Up_conv_3)
        
                       
        Up_conv_3 = ZeroPadding2D(padding = ((1,0), (1,0)))(Up_conv_3)
       
        concat_2 = concatenate([concat,Up_conv_3],axis=-1)
        Conv_6 = Conv2D(filters=128, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(concat_2)

        ## Quantitative Fluorescence Output Branch ##
        outQF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_6)

        outQF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(outQF) #outQF
        
        outQF = BatchNormalization()(outQF)
        
        outQF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(outQF)

        ## Depth Fluorescence Output Branch ##
        #first DF layer 
        outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_6)
  
        outDF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(outDF)
       
        outDF = BatchNormalization()(outDF)
     
        
        outDF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(outDF)

        ## Defining and compiling the model ##
        self.modelD = Model(inputs=[inOP_beg,inFL_beg], outputs=[outQF, outDF])#,outFL])
        self.modelD.compile(loss=['mae', 'mae'],
                      optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                      metrics=['mae', 'mae'])
        self.modelD.summary()
        return None
    
    
    def Fit(self,isTransfer):
        # Where to export information about the fit
        self.exportName = input('Enter a name for exporting the model: ')
        lrDecay = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, verbose=1, min_delta=5e-5)
        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=20, verbose=1, mode='auto')
        callbackList = [earlyStopping,lrDecay]
        
        
        if len(self.exportName) > 0:
            os.makedirs("ModelParameters//"+self.exportName)
            self.exportPath = 'ModelParameters//'+self.exportName+'//'+self.case
            #save dictionary as excel file 
            #fileName = self.exportPath+'_params.xml'
            self.paramsexcelfileName = self.exportPath + '_params.xlsx'
            #xmlParams = dicttoxml(str(self.params))
            print(self.params)
            
            params_dict_for_excel_coversion = self.params.copy()
            for key, value in params_dict_for_excel_coversion.items():
                params_dict_for_excel_coversion[key] = str(value)
                
               
            params_dict_to_dataframe = pd.DataFrame(data=params_dict_for_excel_coversion, index = [0])
            params_dict_to_dataframe.to_excel(self.paramsexcelfileName)
            #with open(fileName,'w') as paramsFile:
            #    paramsFile.write(parseString(xmlParams).toprettyxml("    "))
            
            # Model checkpoint is a keras default callback that saves the model architecture, weights,
            Checkpoint = ModelCheckpoint(self.exportPath+'.keras')
            # CSVLogger is a keras default callback that saves the results of each epoch
            Logger = CSVLogger(self.exportPath+'.log')
            callbackList.append(Checkpoint)
            callbackList.append(Logger)
            xmlParams = dicttoxml(str(self.params))
            with open(self.exportPath+'_params.log','w') as paramsFile:
                paramsFile.write(parseString(xmlParams).toprettyxml("    "))
                
        start = time.perf_counter()
        
        if isTransfer==True:
            h5_files = []
            for folder in os.listdir("ModelParameters"):
                if not folder.endswith((".h5",".log",".xml")):
                    for file in os.listdir("ModelParameters//"+folder):
                        if file.endswith(".h5"):
                            filename = "ModelParameters//"+folder+'//'+file
                            h5_files.append(filename)
                            print(filename)
            loadFile = input('Enter the general and specific directory (e.g. meshLRTests\\\LR2e-5) pertaining to the .h5 (weights) file you would like to load: ')
            self.params['transfer_learning_file_name'] = loadFile #insert the name of the loaded file inside the params dictionary 
            self.params['learningRate']=8e-6
            self.modelD.load_weights(loadFile)
            self.history = self.modelD.fit([self.OP, self.FL], [self.QF, self.DF],validation_split=0.2,batch_size=self.params['batch'],
                                       epochs=50, verbose=1, shuffle=True, callbacks=callbackList)     
        else:

            self.history = self.modelD.fit([self.OP, self.FL], [self.QF, self.DF],validation_split=0.2,batch_size=self.params['batch'],
                                       epochs=self.params['epochs'], verbose=1, shuffle=True, callbacks=callbackList)    
        
        if hasattr(self,'exportPath'):
            fileName = self.exportPath+'_params.xml'
            with open(fileName,'w') as paramsFile:
                paramsFile.write(parseString(xmlParams).toprettyxml("    "))
        
        stop = time.perf_counter()
        print('Fit time = ' + str(stop-start))
        return None
    
    def Plot(self,isTraining=False):
        while True:
            if isTraining==True:
                # Plot loss curves
                plt.plot(self.history.history['loss'])
                plt.plot(self.history.history['val_loss'])
                plt.title('Loss Curves')
                plt.ylabel('Loss (MSE)')
                plt.xlabel('Epoch #')
                plt.legend(['Training', 'Validation'], loc='upper right')
                plt.yscale('log')
                plt.show() 
                break
            elif isTraining==False:
                log_files = []
                for folder in os.listdir("ModelParameters"):
                    if not folder.endswith((".h5",".log",".xml")):
                        for file in os.listdir("ModelParameters//"+folder):
                            if file.endswith(".log"):
                                if  'params' not in file:
                                    filename = "ModelParameters//"+folder+'//'+file
                                    log_files.append(filename)
                                    print(filename)
                name = input('Input the absolute path to the log file: ')
                history = pd.read_csv(name)
                # Plot loss curves
                plt.plot(history['loss'])
                plt.plot(history['val_loss'])
                plt.title('Loss Curves')
                plt.ylabel('Loss (MSE)')
                plt.xlabel('Epoch #')
                plt.legend(['Training', 'Validation'], loc='upper right')
                plt.yscale('log')
                plt.show() 
                break
            else:
                print('You didn\'t select a valid option, type Y/N, or enter nothing to escape: ')
        return None
    
    def Evaluate(self):
        """Evaluate the model's perforance on the testing data"""
        print('\n Evaluating the model\'s prediction performance: \n')
        evaluate = self.modelD.evaluate([self.OP, self.FL],[self.QF, self.DF])
        return None
    
    
    
    def Predict(self,loadandpredict=False):
        """Make predictions of the labels using the testing data (note that this is different from the Evaluate method, since the evaluate method tells us the metrics (without making the predictions) while this method only makes predictions"""
        predict = self.modelD.predict([self.OP, self.FL])  
        #if loadandpredict==True: #debug self.new_model 
        #    predict = self.new_model.predict([self.OP, self.FL])
        QF_P = predict[0] 
        DF_P = predict[1]
        QF_P /= self.params['scaleQF']
        DF_P /= self.params['scaleDF']
        print('\n Making predictions on the test data... \n')  
        
        while True:
            save = input('\n Save these predictions? (Y/N) ')
            if save in ['Y','y']:
                self.exportName = input('The model, best weights, and some assorted parameters and other information will be saved during fitting; enter a name for the export directory or leave this field blank if you do not want to export (note that if you do not export, you will not be able to plot your results in the future once the current history variable is lost from memory, however you can still plot the results now): ')
                if len(self.exportName) > 0:
                    # Export MATLAB file to case
                    self.exportPath = 'Predictions//'+self.exportName+'.mat'
                    a = np.array(DF_P[:,:,:,0])
                    b = np.array(QF_P[:,:,:,0])
                    scipy.io.savemat(self.exportPath,{'DF_pred':a,"QF_pred":b})
                break
            elif save in ['N','n']:
                break
            else:
                print('You didn\'t select a valid option, type Y/N')
        return None
    
    def LoadandPredict(self):
        """In the case that we have a pre-defined model which we would like to test, we can use LoadandPredict to load this model and then evaluate and test using our testing data"""

        h5_files = []
        for folder in os.listdir("ModelParameters"):
            if not folder.endswith((".h5",".log",".xml")):
                for file in os.listdir("ModelParameters//"+folder):
                    if file.endswith(".h5"):
                        filename = "ModelParameters//"+folder+'//'+file
                        h5_files.append(filename)
                        print(filename)
                
        while True: # Loop until user makes acceptable choice for the input directory
            loadFile = input('Enter the general and specific directory (e.g. meshLRTests\\\LR2e-5) pertaining to the .h5 (weights) file you would like to load: ')
            if loadFile == '': # User enters nothing; break
                break
            elif loadFile in h5_files:
                if hasattr(self,'modelD'): # If the a model exists, this instance of the class will have the attribute modelD
                    checkParams = input('\nA model is currently defined but in order for the weights to load successfully this model must have been defined using the same parameters as when the weights were initially defined (i.e. nF, kernel, etc.). Are the parameters/architecture of your model and weights the same? (Y/N) ')
                    if checkParams in ['Y','y']: # If the user decides that the model is already defined using the correct parameters 
                        self.Model()
                        self.modelD = load_model(loadFile) # Load the weights into this model
                        # Whether the model was already defined properly, or did not exist, or needed to be redefined, we now need to load in data for testing and then make predictions based on the loaded weights 
                        self.importData(isTesting=True,quickTest=False)
                        self.Predict() # Make predictions
                        self.Evaluate()
                        break
                    elif checkParams in ['N','n']: # If the user decides that the model is not defined using the correct parameters
                        print('\nI suggest referring to the .xml file associated with your saved weights to determine which parameters you need to change. Calling the Params() method now, and modelling immediately following the changes: \n') # Refer to .xml params
                        self.Params() # Allow the user to define new parameters
                        self.Model() # Model using these new, presumable correct, parameters
                        # Load in model architecture with weights (not the same architecture as the Model function)
                        self.modelD.load_weights(loadFile)
                        # Whether the model was already defined properly, or did not exist, or needed to be redefined, we now need to load in data for testing and then make predictions based on the loaded weights 
                        self.importData(isTesting=True,quickTest=False)
                        self.Predict(loadandpredict=True) # Make predictions
                        self.Evaluate()
                        break
                else: # If the modelD attribute does not exist
                    print('\nA model is not currently defined. I suggest referring to the .xml file associated with your saved weights to determine which parameters you need to change. Calling the Params() method now, and modelling immediately following the changes: ') # Refer to .xml params
                    self.Params() # Allow the user to define new parameters
                    self.Model() # Model using these new, presumable correct, parameters
                    print('Now loading weights: ')
                    self.modelD.load_weights(loadFile) # Load the weights into this model 
                    # Whether the model was already defined properly, or did not exist, or needed to be redefined, we now need to load in data for testing and then make predictions based on the loaded weights 
                    self.importData(isTesting=True,quickTest=False)
                    self.Predict() # Make predictions
                    self.Evaluate()
                    break
            else: # User entered something invalid
                print('The directory name you entered is invalid. Enter nothing to escape. ')       
        return
    
    
    def QuickTest(self):
        h5_files = []
        for folder in os.listdir("ModelParameters"):
            if not folder.endswith((".h5",".log",".xml")):
                for file in os.listdir("ModelParameters//"+folder):
                    if file.endswith(".h5"):
                        filename = "ModelParameters//"+folder+'//'+file
                        h5_files.append(filename)
                        print(filename)        
        while True: # Loop until user makes acceptable choice for the input directory
            loadFile = input('Enter the general and specific directory pertaining to the .h5 (weights) file you would like to load: ')
            if loadFile == '': # User enters nothing; break
                break
            elif loadFile in h5_files:
                self.new_model = tf.keras.models.load_model(loadFile, compile = False)
                self.new_model.load_weights(loadFile)
                self.modelD = load_model(loadFile,  compile = False) 
                #self.new_model.summary()# Load the weights into this model
                while True:
                    self.importData(isTesting=True,quickTest=True)
                    self.Predict()
                    #self.Results()
                    testAgain = input('Test again? (Y/N): ')
                    if testAgain in ['Y','y']:
                        print('Running again')
                    elif testAgain in ['N','n']:
                        return
                    else:
                        return
            else: # If the modelD attribute does not exist
                print('\nModel is not currently defined - select again.') 
        return
    
    def PrintImg(self):
        """Bare bones structure for a method to visualize the input or output maps"""
        dims = len(np.shape(self.FL))
        img = input('Choose the input/output map you want to display (FL, OP, DF, QF): ')
        sample = 0
        if dims ==5:
            if img in ['FL','fl']: # Print fluorescence maps
                fx = self.params['nF'] # Number of spatial frequencies
                for i in range(fx): 
                    plt.figure()
                    plt.imshow(self.FL[sample,:,:,i,:])
                    plt.colorbar()
                    plt.clim(0,self.FL[sample,:,:,i,:].max())
                    plt.show()
                return None
            elif img in ['OP','op']: # Print optical properties
                for i in range(2):
                    plt.figure()
                    plt.imshow(self.OP[sample,:,:,i])
                    plt.colorbar()
                    plt.clim(0,self.OP[sample,:,:,i].max())
                    plt.imshow(self.OP[sample,:,:,i])
                    plt.show()
                return None
            elif img in ['DF','df']: # Print depth maps
                for i in range(1):
                    plt.imshow(self.DF[sample,:,:,i])
                    plt.show()
                return None
            elif img in ['QF','qf']: # Print concentration maps
                for i in range(1):
                    plt.imshow(self.QF[sample,:,:,i])
                    plt.show()
                return None
        else:
            if img in ['FL','fl']: # Print fluorescence maps
                fx = self.params['nF'] # Number of spatial frequencies
                for i in range(fx): 
                    plt.figure()
                    plt.imshow(self.FL[i,:,:,0])
                    plt.colorbar()
                    plt.clim(0,self.FL[i,:,:,0].max())
                    plt.show()
                return None
            elif img in ['OP','op']: # Print optical properties
                for i in range(2):
                    plt.figure()
                    plt.imshow(self.OP[i,:,:,0])
                    plt.colorbar()
                    plt.clim(0,self.OP[i,:,:,0].max())
                    plt.imshow(self.OP[i,:,:,0])
                    plt.show()
                return None
            elif img in ['DF','df']: # Print depth maps
                for i in range(1):
                    plt.imshow(self.DF[:,:,0])
                    plt.show()
                return None
            elif img in ['QF','qf']: # Print concentration maps
                for i in range(1):
                    plt.imshow(self.QF[:,:,0])
                    plt.show()
                return None
    
    def Results(self):
        predict = self.modelD.predict([self.OP, self.FL])  
        QF_P = predict[0] 
        DF_P = predict[1]
        QF_P /= self.params['scaleQF']
        DF_P /= self.params['scaleDF']
        i=0        
        fig, axs = plt.subplots(1,2)
        plt.set_cmap('jet')
        plt.colorbar(axs[0].imshow(DF_P[i,:,:],vmin=0,vmax=10), ax=axs[0],fraction=0.046, pad=0.04)
        axs[0].axis('off')
        axs[0].set_title('Predicted Depth (mm)')
        plt.colorbar(axs[1].imshow(QF_P[i,:,:],vmin=0,vmax=10), ax=axs[1],fraction=0.046, pad=0.04)
        axs[1].axis('off')
        axs[1].set_title('Predicted Concentration (ug/mL)')
        plt.tight_layout()
        plt.show()
        return None
    
    def plot_3d_map(self, DF, QF, title_str):
        
        DF_error = np.abs(DF - self.DF)
        QF_error = np.abs(QF - self.QF)
        
        
        ax = plt.axes(projection = "3d")
        
        x_data = np.linspace(0, DF.shape[1], num = DF.shape[1])
        y_data = np.linspace(0, DF.shape[2], num = DF.shape[2])
        Z = DF[0, :,:,0]
        face_col =DF_error[0, :, :, 0]
        X,Y = np.meshgrid(x_data, y_data)
        
        x_dim, y_dim = face_col.shape
        
        cmap = plt.cm.get_cmap('jet')

        
        face_col_map_jet = cmap(face_col)
        
        #ls = LightSource()
        #illuminated_surface = ls.shade_rgb(face_col_map_jet, face_col)

        ax.plot_surface(X,Y,Z, cmap = 'jet', facecolors = face_col_map_jet, antialiased=False, norm = True)
        ax.set_xlim(0,x_dim)
        ax.set_ylim(0,y_dim)
        ax.set(xticklabels=[],
           yticklabels=[],
           zticklabels=[])
        
        ax.set_zlabel("Depth (mm)")
        plt.title(title_str)
    
        plt.show()
        
        
    def classify(self, DF_P):      
        
        DF_max_per_case = np.nanmax(DF_P[:,:,:,0], axis = (1,2))
        CL = DF_max_per_case > 4 # 4mm or less would be one 
        return CL    
    
    def add_classifier(self):
        print("DF_shape", self.DF.shape)

        DF_zeros = np.array(self.DF)
        for x in range(self.DF.shape[0]):
            for i in range(self.DF.shape[2]):
                DF_zeros_per_column = self.DF[x, :, i, 0] == 0
                DF_zeros[x,DF_zeros_per_column, i, 0] = np.nan
        
        print("DF_zeros shape:", DF_zeros.shape)
        DF_min_per_case = np.nanmax(DF_zeros, axis = (1,2))
        self.CL = DF_min_per_case > 4 # 4mm or more would be one 
        self.CL = self.CL.squeeze()
        
        print("CL_shape", self.CL.shape)

    def confidence_interval(self, metric, num_data):        
        z_95 = 1.96 
        
        
        CI = z_95 * np.sqrt(metric* (1- metric) / (num_data))
            
        return CI
    
    def concentration_error_vs_depth(self, QF, QF_P, DF):
        err_Q_graph = plt.figure()
        err_Q = np.array(np.abs(QF_P-QF)) 
        #err_Q = np.reshape(err_Q, (err_Q.shape[0], -1))
                                                            
        plt.scatter(DF[self.indxIncl],err_Q[self.indxIncl], s = 1)
        plt.ylabel("Absolute Concentration Error (ug/ml))")
        plt.xlabel("True Depth (mm)")
        plt.colorbar()

        err_Q_graph.show()    

                
    def Analysis(self):
        self.save = 0
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix
        h5_files = []
        for folder in os.listdir("ModelParameters"):
            if not folder.endswith((".h5",".log",".xml", ".keras")):
                for file in os.listdir("ModelParameters//"+folder):
                    if file.endswith(".keras") or file.endswith(".h5"):
                        filename = "ModelParameters//"+folder+'//'+file
                        h5_files.append(filename)
                        print(filename)        
        while True: # Loop until user makes acceptable choice for the input directory
            loadFile = input('Enter the general and specific directory pertaining to the .h5 (weights) file you would like to load: ')
            if loadFile == '': # User enters nothing; break
                break
            elif loadFile in h5_files:
                #self.xml_file_name = loadFile.replace(".h5", "_params.xml")
                self.new_model = tf.keras.models.load_model(loadFile, compile=False)
                self.new_model.load_weights(loadFile)
                self.modelD = load_model(loadFile, compile=False) 
                #self.new_model.summary()# Load the weights into this model
                self.importData(isTesting=True,quickTest=True)
                break
                    
            else: # If the modelD attribute does not exist
                print('\nModel is not currently defined - select again.') 
                break
    
        self.Predict()
        predict = self.modelD.predict([self.OP, self.FL])  
        QF_P = predict[0] 
        DF_P = predict[1]
        QF_P /= self.params['scaleQF']
        DF_P /= self.params['scaleDF']  
        
        self.add_classifier() #add classifier to test data 
        CL_P = self.classify(DF_P)
        true = self.CL#np.where(self.DF > 5, 1, 0)
        #true = np.concatenate(true)
        
      
        tn, fp, fn, tp = confusion_matrix(true, CL_P, labels=[0, 1]).ravel()
        accuracy = accuracy_score(true, CL_P)
        specificity = tn/(tn+fp)
        sensitivity = tp/(tp+fn)
        precision = tp/(tp+fp)
        npv = tn/(tn+fn)
        F1 = 2*(precision*sensitivity)/(precision + sensitivity)
        
        num_data = np.shape(true)[0]
        CI_accuracy = self.confidence_interval(accuracy,num_data)
        CI_specificity = self.confidence_interval(specificity,num_data)
        CI_sensitivity = self.confidence_interval(sensitivity,num_data)
        CI_precision = self.confidence_interval(precision,num_data)
        CI_npv = self.confidence_interval(npv,num_data)

        print('Accuracy: ',"%.4f +/- %.4f" %(accuracy, CI_accuracy))
        print('Specificity (TNR): ',"%.4f +/- %.4f "%(specificity, CI_specificity))
        print('Sensitivity (TPR): ',"%.4f +/- %.4f " %(sensitivity, CI_sensitivity))
        print('Precision (PPV): ',"%.4f +/- %.4f"%(precision, CI_precision))
        print('Npv: ',"%.4f +/- %.4f"%(npv, CI_npv))

        
       

        self.indxIncl = np.nonzero(self.DF)
        ## Error Stats
        # Average error
        DF_error = DF_P - self.DF
        QF_error = QF_P - self.QF
        DF_erroravg = np.mean(abs(DF_error[self.indxIncl]))
        DF_errorstd = np.std(abs(DF_error[self.indxIncl]))
        QF_erroravg = np.mean(abs(QF_error[self.indxIncl]))
        QF_errorstd = np.std(abs(QF_error[self.indxIncl]))
        print('Average Depth Error (SD): {}({}) mm'.format(float('%.5g' % DF_erroravg),float('%.5g' % DF_errorstd)))
        print('Average Concentration Error (SD): {}({}) ug/mL'.format(float('%.5g' % QF_erroravg),float('%.5g' % QF_errorstd)))
        # Overall  mean squared error
        DF_mse = np.sum((DF_P - self.DF) ** 2)
        DF_mse /= float(DF_P.shape[0] * DF_P.shape[1] * DF_P.shape[2])
        QF_mse = np.sum((QF_P - self.QF) ** 2)
        QF_mse /= float(QF_P.shape[0] * QF_P.shape[1] * QF_P.shape[2])
        print('Depth Mean Squared Error: {} mm'.format(float('%.5g' % DF_mse)))
        print('Concentration Mean Squared Error: {} ug/mL'.format(float('%.5g' % QF_mse)))
        # Max and Min values per sample
        ua_max = []
        us_max = []
        DF_max = []
        DFP_max = []
        QF_max = []
        QFP_max = []
        
        
        
        if self.save in ['Y','y']:
            plot_save_path = 'Predictions//' + self.exportName + '//'

        for i in range(DF_P.shape[0]):
            ua_max.append(self.OP[i,:,:,0].max())
            us_max.append(self.OP[i,:,:,1].max())
            DF_max.append(self.DF[i,:,:].max())
            DFP_max.append(DF_P[i,:,:].max())
            QF_max.append(self.QF[i,:,:].max())
            QFP_max.append(QF_P[i,:,:].max())

            
        
        # SSIM per sample
        DF_ssim =[]
        QF_ssim =[]
        for i in range(DF_P.shape[0]):
            df_p = np.reshape(DF_P[i,:,:],(DF_P.shape[1],DF_P.shape[2])) # reshape predicted
            df_t = np.reshape(self.DF[i,:,:],(self.DF.shape[1],self.DF.shape[2])) # reshape true
            df_ssim = ssim(df_p,df_t,data_range=max(df_p.max(),df_t.max())-min(df_p.min(),df_t.min()))
            DF_ssim.append(df_ssim)
            qf_p = np.reshape(QF_P[i,:,:],(QF_P.shape[1],QF_P.shape[2])) # reshape predicted
            qf_t = np.reshape(self.QF[i,:,:],(self.QF.shape[1],self.QF.shape[2])) # reshape true
            qf_ssim = ssim(qf_p,qf_t,data_range=max(qf_p.max(),qf_t.max())-min(qf_p.min(),qf_t.min()))
            QF_ssim.append(qf_ssim)
        print('Overall Depth SSIM: {}'.format(float('%.5g' % np.mean(DF_ssim))))
        print('Overall Concentration SSIM: {}'.format(float('%.5g' % np.mean(QF_ssim))))
        ## Plot Correlations
        fig, (plt1, plt2) = plt.subplots(1, 2)
        plt1.scatter(self.DF[self.indxIncl],DF_P[self.indxIncl],s=1)
        plt1.set_xlim([0, 15])
        plt1.set_ylim([0, 15])
        y_lim1 = plt1.set_ylim()
        x_lim1 = plt1.set_xlim()
        plt1.plot(x_lim1, y_lim1,color='k')
        plt1.set_ylabel("Predicted Depth (mm)")
        plt1.set_xlabel("True Depth (mm)")
        plt2.scatter(self.QF[self.indxIncl],QF_P[self.indxIncl],s=1)
        plt2.set_xlim([0, 10])
        plt2.set_ylim([0, 10])
        plt2.set_ylabel("Predicted Concentration (ug/mL)")
        plt2.set_xlabel("True Concentration (ug/mL)")
        plt.tight_layout()
        if self.save in ['Y','y']:
            plot_save_path_depth_and_concentration = plot_save_path + '_DF_QF.png'
            plt.savefig(plot_save_path_depth_and_concentration, dpi=100, bbox_inches='tight')
        plt.show()
        ## Error Histograms
        fig, (plt1, plt2) = plt.subplots(1, 2)
        plt1.hist(DF_error[self.indxIncl],bins=100)
        plt2.hist(QF_error[self.indxIncl],bins=100)
        plt1.set_xlabel("Depth Error (mm)")
        plt1.set_ylabel("Frequency")
        plt2.set_xlabel("Concentration Error (ug/mL)")
        plt2.set_ylabel("Frequency")
        plt.tight_layout()
        if self.save in ['Y','y']:
            plot_save_path_DF_QF_error = plot_save_path + '_DFerror_QFerror.png'
            plt.savefig(plot_save_path_DF_QF_error, dpi=100, bbox_inches='tight')
        plt.show()
        ## Plot Depth SSIM
        fig, axs = plt.subplots(2,2)
        fig.suptitle('Depth SSIM')
        axs[0,0].scatter(ua_max,DF_ssim,s=1)
        axs[0,0].set_xlabel('Max Absorption')
        axs[0,0].set_ylim([0,1])
        axs[0,1].scatter(us_max,DF_ssim,s=1)
        axs[0,1].set_xlabel('Max Scattering')
        axs[0,1].set_ylim([0,1])
        axs[1,0].scatter(DF_max,DF_ssim,s=1)
        axs[1,0].set_xlabel('Max Depth')
        axs[1,0].set_ylim([0,1])
        axs[1,1].scatter(QF_max,DF_ssim,s=1)
        axs[1,1].set_xlabel('Max Concentration')
        axs[1,1].set_ylim([0,1])
        plt.tight_layout()
        if self.save in ['Y','y']:
            plot_save_path_depth_SSIM = plot_save_path + 'depth_SSIM.png'
            plt.savefig(plot_save_path_depth_SSIM, dpi=100, bbox_inches='tight')
        plt.show()
        ## Plot Concentration SSIM
        fig, axs = plt.subplots(2,2)
        fig.suptitle('Concentration SSIM')
        axs[0,0].scatter(ua_max,QF_ssim,s=1)
        axs[0,0].set_xlabel('Max Absorption')
        axs[0,0].set_ylim([0,1])
        axs[0,1].scatter(us_max,QF_ssim,s=1)
        axs[0,1].set_xlabel('Max Scattering')
        axs[0,1].set_ylim([0,1])
        axs[1,0].scatter(DF_max,QF_ssim,s=1)
        axs[1,0].set_xlabel('Max Depth')
        axs[1,0].set_ylim([0,1])
        axs[1,1].scatter(QF_max,QF_ssim,s=1)
        axs[1,1].set_xlabel('Max Concentration')
        axs[1,1].set_ylim([0,1])
        plt.tight_layout()
        if self.save in ['Y','y']:
            plot_save_path_concentration_SSIM = plot_save_path + 'concentration_SSIM.png'
            plt.savefig(plot_save_path_concentration_SSIM, dpi=100, bbox_inches='tight')
        plt.show()
        ## Plot Min True Depth vs. Min Predicted Depth
        fig
        plt.scatter(DF_max,DFP_max,s=1)

        
        DF_max_classify = np.array(DF_max) < 5 
        DFP_max_classify = np.array(DFP_max) < 5
        
        failed_result = DF_max_classify !=DFP_max_classify
        DF_max = np.array(DF_max)
        DFP_max = np.array(DFP_max)
        QF_max = np.array(QF_max)
        QFP_max = np.array(QFP_max)
        plt.scatter(DF_max[failed_result],DFP_max[failed_result],s=1, color = ['red'])

        plt.xlim([0, 15])
        plt.ylim([0, 15])
        plt.plot(plt.xlim([0, 15]), plt.ylim([0, 15]),color='k')
        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("Max Depth")
        plt.tight_layout()
        if self.save in ['Y','y']:
            plot_save_path_min_depth = plot_save_path + 'min_depth.png'
            plt.savefig(plot_save_path_min_depth, dpi=100, bbox_inches='tight')
        plt.show()

        max_depth_error = np.mean(np.abs(DFP_max - DF_max))
        max_depth_error_std = np.std(np.abs(DFP_max - DF_max))
        print("Average Minimum Depth Error (SD) : {max_depth_error} ({max_depth_error_std})".format(max_depth_error = max_depth_error, max_depth_error_std = max_depth_error_std))
        
        
        #plot depth error as a function of absorption, scattering 
        #self.error_vs_absorption(DF_max, DFP_max, ua_max)
        #self.error_vs_scattering(DF_max, DFP_max, us_max)

        #self.scatter_vs_absorption_failed_result(DF_max, DFP_max, ua_max, us_max, failed_result)
        
        #self.concentration_vs_depth_error(DF_max, DFP_max, QF_max)
            
        #self.concentration_error_vs_depth_error(DF_max, DFP_max, QF_max, QFP_max)
        #self.concentration_vs_depth_error_classify(DF_max, DFP_max, QF_max, failed_result)
        #self.concentration_error_vs_depth_error_classify(DF_max, DFP_max, QF_max, QFP_max, failed_result)
        

        #self.concentration_error_vs_concentration(QF_max, QFP_max, failed_result)
        
        #self.max_concentration_graph(QF_max, QFP_max)
        # Plot true and predicted depth and concentration
        print("this is self.indxIncl", self.indxIncl)
        self.concentration_error_vs_depth(self.QF, QF_P, self.DF)


        num_plot_display = 10
        if self.DF.shape[0] < 10:
            for i in range(self.DF.shape[0]):
                fig, axs = plt.subplots(2,3)
                plt.set_cmap('jet')
                plt.colorbar(axs[0,0].imshow(self.DF[i,:,:],vmin=0,vmax=15), ax=axs[0, 0],fraction=0.046, pad=0.04)
                
                
                axs[0,0].axis('off')
                axs[0,0].set_title('True Depth (mm)')
                plt.colorbar(axs[0,1].imshow(DF_P[i,:,:],vmin=0,vmax=15), ax=axs[0, 1],fraction=0.046, pad=0.04)
                axs[0,1].axis('off')
                axs[0,1].set_title('Predicted Depth (mm)')
                plt.colorbar(axs[0,2].imshow(abs(DF_error[i,:,:]),vmin=0,vmax=15), ax=axs[0, 2],fraction=0.046, pad=0.04)
                axs[0,2].axis('off')
                axs[0,2].set_title('|Error (mm)|')
                plt.colorbar(axs[1,0].imshow(self.QF[i,:,:],vmin=0,vmax=10), ax=axs[1, 0],fraction=0.046, pad=0.04)
                axs[1,0].axis('off')
                axs[1,0].set_title('True Conc (ug/mL)')
                plt.colorbar(axs[1,1].imshow(QF_P[i,:,:],vmin=0,vmax=10), ax=axs[1, 1],fraction=0.046, pad=0.04)
                axs[1,1].axis('off')
                axs[1,1].set_title('Predicted Conc (ug/mL)')
                plt.colorbar(axs[1,2].imshow(abs(QF_error[i,:,:]),vmin=0,vmax=10), ax=axs[1, 2],fraction=0.046, pad=0.04)
                axs[1,2].axis('off')
                axs[1,2].set_title('|Error (ug/mL)|')
                plt.tight_layout()   
                if i == 0 and self.save in ['Y','y']:
                    plot_save_path_DF = plot_save_path + 'DF.png'
                    plt.savefig(plot_save_path_DF, dpi=100, bbox_inches='tight')
                plt.show()
        else:
            for i in range(num_plot_display):
                fig, axs = plt.subplots(2,3)
                plt.set_cmap('jet')
                plt.colorbar(axs[0,0].imshow(self.DF[i,:,:],vmin=0,vmax=15), ax=axs[0, 0],fraction=0.046, pad=0.04)
                #value = -1
                #min_index = np.where(self.DF[i,:,:] == np.max(self.DF[i,:,:]))
                #temp_value = self.DF[i, min_index[0], min_index[1]]
                #self.DF[i, min_index[0], min_index[1]] = value 
                
                #masked_array = np.ma.masked_where(self.DF[i,:,:] == value, self.DF[i,:,:])
                #cmap = plt.cm.jet  # Can be any colormap that you want after the cm
                #cmap.set_bad(color='white')
                #plt.imshow(masked_array, cmap=cmap)
                
                #plt.colorbar(axs[0,0].imshow(masked_array,vmin=0,vmax=15), ax=axs[0, 0],fraction=0.046, pad=0.04, cmap = cmap)
                #temp_value_str_1 = temp_value[0]

                #axs[0,0].text(5, 5, 'max_depth = ' + str(temp_value_str_1), bbox={'facecolor': 'white', 'pad': 4}, size = 'x-small')
                #self.DF[i, min_index[0], min_index[1]] = temp_value 


                axs[0,0].axis('off')
                axs[0,0].set_title('True Depth (mm)')
                
                #min_index = np.where(DF_P[i,:,:] == np.max(DF_P[i,:,:]))
                #print("min_index", min_index)
                #temp_value = DF_P[i, min_index[0], min_index[1]]
                #DF_P[i, min_index[0], min_index[1]] = value 
                
                #masked_array = np.ma.masked_where(DF_P[i,:,:] == value, DF_P[i,:,:])
                #cmap = plt.cm.jet  # Can be any colormap that you want after the cm
                #cmap.set_bad(color='white')
                #plt.imshow(masked_array, cmap=cmap)
                
                #temp_value_str_2 = temp_value[0]
                #plt.colorbar(axs[0,1].imshow(masked_array,vmin=0,vmax=15), ax=axs[0, 1],fraction=0.046, pad=0.04, cmap = cmap)
                #axs[0,1].text(5, 5, 'max_depth = ' + str(temp_value_str_2), bbox={'facecolor': 'white', 'pad': 4}, size = 'x-small')
                #DF_P[i, min_index[0], min_index[1]] = temp_value 
                
                
                plt.colorbar(axs[0,1].imshow(DF_P[i,:,:],vmin=0,vmax=15), ax=axs[0, 1],fraction=0.046, pad=0.04)
                axs[0,1].axis('off')
                axs[0,1].set_title('Predicted Depth (mm)')
                plt.colorbar(axs[0,2].imshow(abs(DF_error[i,:,:]),vmin=0,vmax=15), ax=axs[0, 2],fraction=0.046, pad=0.04)
                axs[0,2].axis('off')
                axs[0,2].set_title('|Error (mm)|')
                plt.colorbar(axs[1,0].imshow(self.QF[i,:,:],vmin=0,vmax=10), ax=axs[1, 0],fraction=0.046, pad=0.04)
                axs[1,0].axis('off')
                axs[1,0].set_title('True Conc (ug/mL)')
                plt.colorbar(axs[1,1].imshow(QF_P[i,:,:],vmin=0,vmax=10), ax=axs[1, 1],fraction=0.046, pad=0.04)
                axs[1,1].axis('off')
                axs[1,1].set_title('Predicted Conc (ug/mL)')
                plt.colorbar(axs[1,2].imshow(abs(QF_error[i,:,:]),vmin=0,vmax=10), ax=axs[1, 2],fraction=0.046, pad=0.04)
                #axs[0,2].text(5, 5, 'max_depth error = ' + str(temp_value_str_1 - temp_value_str_2), bbox={'facecolor': 'white', 'pad': 4}, size = 'x-small')

                axs[1,2].axis('off')
                axs[1,2].set_title('|Error (ug/mL)|')
                plt.tight_layout()
                if i == 0 and self.save in ['Y','y']:
                    plot_save_path_DF = plot_save_path + 'DF.png'
                    plt.savefig(plot_save_path_DF, dpi=100, bbox_inches='tight')
                plt.show()
        
    def PrintFeatureMap(self):
        """Generate Feature Maps"""
        feature_maps = self.modelFM.predict([self.OP, self.FL]) # Output for each layer
        layer_names = [layer.name for layer in self.modelFM.layers] # Define all the layer names
        layer_outputs = [layer.output for layer in self.modelFM.layers] # Outputs of each layer
        print('Feature map names and shapes:')
        for layer_name, feature_map in zip(layer_names, feature_maps):
            print(f" {layer_name} shape is --> {feature_map.shape}")
        while True:
            fm = input('Choose the feature map you want to display: ')
            if fm in layer_names:
                self.fm = fm
                break
            else:
                print('Invalid entry, try again')       
        for layer_name, feature_map in zip(layer_names, feature_maps):  
            if layer_name == self.fm:
                if len(feature_map.shape) == 5:
                    for j in range(4): # Number of feature maps (stick to 4 at a time)
                        for i in range(self.params['nF']): # Spatial frequency
                            feature_image = feature_map[0, :, :, i, j]
                            feature_image-= feature_image.mean()
                            feature_image*=  64
                            feature_image+= 128
                            plt.figure(  )
                            plt.title ( layer_name +' Filter: '+str(j+1)+' SF: '+str(i+1) )
                            plt.grid  ( False )
                            plt.imshow( feature_image, aspect='auto')
                else:
                    for i in range(1): # Number of feature maps
                        feature_image = feature_map[0, :, :, i]
                        feature_image-= feature_image.mean()
                        feature_image*=  64
                        feature_image+= 128
                        plt.figure(  )
                        plt.title ( layer_name +' Filter: ' +str(i+1))
                        plt.grid  ( False )
                        plt.imshow( feature_image, aspect='auto')


if __name__ == "__main__":
    test = DL()
    test.Model()