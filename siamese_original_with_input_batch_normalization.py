from __future__ import print_function
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
from sklearn import metrics
from csv import writer, reader
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
import numpy as np, h5py
import os, time, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import scipy.io
import pandas as pd
from skimage.metrics import structural_similarity as ssim

# Tensorflow / Keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout, Activation
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing import image

from gtxDLClassAWSUtils import Utils

import boto3 
import io
import openpyxl


class DL(Utils):    
    # Initialization method runs whenever an instance of the class is initiated
    def __init__(self):
        self.bucket = '20240909-hikaru'
        isCase = input('Choose a case:\n Default\n Default4fx')
        self.case = isCase
        self.params = self.Defaults(self.case) # Define some default parameters based on the selected case
        self.isTesting = False

        self.Model()
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
    
    def convert_background_val(self):
        #self.background_val = 25
        self.DF = np.array(self.DF)
        #self.RE = np.array(self.RE)
        #print("np.sum QF", np.sum(np.isnan(self.QF)))
        #print("np.sum OP", np.sum(np.isnan(self.OP)))
        #print("np.sum FL", np.sum(np.isnan(self.FL)))
        #print("np.sum RE", np.sum(np.isnan(self.RE)))


        for x in range(self.DF.shape[0]): #for each case 
            for i in range(self.DF.shape[1]):
                    
                DF_zeros_per_column = self.DF[x, i, :] == 0
                self.DF[x, i, DF_zeros_per_column] = self.background_val
                
        #for x in range(self.RE.shape[0]): #for each case 
        #    for i in range(self.RE.shape[1]):
                    
        #        RE_nan_per_column = np.isnan(self.RE[x, i, :])
        #        self.RE[x, i, RE_nan_per_column] = 0
                
           
        
        
    def add_classifier(self):

        DF_zeros = np.array(self.DF)
        for x in range(self.DF.shape[2]):
            for i in range(self.DF.shape[1]):
                DF_zeros_per_column = self.DF[:, i, x] == 0
                DF_zeros[DF_zeros_per_column, i, x] = np.nan
                           
        DF_min_per_case = np.nanmin(DF_zeros, axis = (0,1))
        self.CL = DF_min_per_case < 5 # 5mm or less would be one 
        self.CL = np.reshape(self.CL, (1, self.CL.shape[0]))
            
    def check_distribution(self):
        
        DF_max = []
        QF_max = []
        
        s3_client = boto3.client('s3')
        isTesting = False
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

       

        file_key = str(input('Enter the name of the dataset you want to import e.g. matALL.mat '))
        
        obj = s3_client.get_object(Bucket=self.bucket, Key=file_key)
        dataTemp = obj['Body'].read()
        dataset = h5py.File(io.BytesIO(dataTemp))
        FL = dataset['F']
        DF = dataset['DF']
        OP = dataset['OP']
        QF = dataset['QF']
        
        for i in range(DF.shape[2]):
            
            QF_max.append(QF[:,:,i].max())
            #DF_max.append(DF[:,:,i].max())
            
        
        data_num = list(range(0,len(QF_max)))
        plt.scatter(data_num, QF_max)
        plt.xlim([0, 100])
        #plt.plot(data_num, DF_max)
                
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
        
        self.params["training_file_name"] = self.file_key
        obj = s3_client.get_object(Bucket=self.bucket, Key=self.file_key)
        dataTemp = obj['Body'].read()
        self.dataset = h5py.File(io.BytesIO(dataTemp))
        self.FL = self.dataset['F']
        three_branch = False
        if three_branch:
            self.DF = self.dataset['DF_sub']
        else:
            self.DF = self.dataset['DF']
        self.OP = self.dataset['OP']
        self.QF = self.dataset['QF']
        self.RE = self.dataset['RE']

        self.temp_DF_pre_conversion = self.DF

        self.add_classifier() #add classifier before changing background value 
        
        self.background_val = str(input('Enter the value of the background'))
        if self.background_val == '':
            self.background_val = 0 #default to zero
        
        self.convert_background_val() #convert values of DF background
        
        print(self.DF[:,:,0])
        print(self.OP[:,:,:,0])
        print(self.OP[:,:,:, 1])
        print(self.QF[:,:,0])
        print(self.FL[:,:,0])

    
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
            
          
#
            #numSets = 100
            self.FL = np.reshape(self.FL[indxFx,:,:,0:numSets],(len(indxFx),self.params['xX'],self.params['yY'],numSets,1))
            
            self.RE = np.reshape(self.RE[indxFx,:,:,0:numSets],(len(indxFx),self.params['xX'],self.params['yY'],numSets,1))

            self.OP = self.OP[:,:,:,0:numSets]
            self.DF = self.DF[:,:,0:numSets]
            self.DF = np.reshape(self.DF,(self.params['xX'],self.params['yY'],numSets,1))
            self.temp_DF_pre_conversion = self.temp_DF_pre_conversion[:,:,0:numSets]
            self.temp_DF_pre_conversion = np.reshape(self.temp_DF_pre_conversion,(self.params['xX'],self.params['yY'],numSets,1))
            self.QF = self.QF[:,:,0:numSets]
            self.QF = np.reshape(self.QF,(self.params['xX'],self.params['yY'],numSets,1))

            # Reorder data
            self.RE = np.swapaxes(self.RE,0,3)

            self.FL = np.swapaxes(self.FL,0,3)
            self.OP = np.swapaxes(self.OP,0,3)
            self.DF = np.moveaxis(self.DF,2,0)
            self.temp_DF_pre_conversion = np.moveaxis(self.temp_DF_pre_conversion,2,0)
            self.QF = np.moveaxis(self.QF,2,0)
            
            stop = time.perf_counter()
            print('Load time = ' + str(stop-start))

            # Apply scale
            self.OP[:,:,:,0] *= self.params['scaleOP0']
            self.OP[:,:,:,1] *= self.params['scaleOP1']
            self.DF *= self.params['scaleDF']
            self.QF *= self.params['scaleQF']
            self.RE[:,:,:,0] *= self.params['scaleRE']
            #self.OP[:,:,:,0] = self.OP[:,:,:,0] / OP_0_max
            #self.OP[:,:,:,1] = self.OP[:,:,:,1] / OP_1_max
            #self.DF /= DF_max
            #self.QF /= QF_max
            #self.RE[:,:,:,0] /= np.max(self.RE[:,:,:,0])
            
            if 'scaleFL0' in self.params:
                for i in range(self.params['nF']):
                    scaleN = 'scaleFL'+ str(i)
                    self.FL[:,:,:,i] *= self.params[scaleN]
                    print(np.mean(self.FL[:,:,:,i]))
            else:
                self.FL *= self.params['scaleFL']
                #self.FL[:,:,:,:] /= FL_max


  


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
            self.RE = np.reshape(self.RE[indxFx,:,:],(len(indxFx),self.params['xX'],self.params['yY'],1,1))

            #self.OP = self.OP[:,:,:]
            #self.OP = self.OP.reshape((2,self.params['xX'], self.params['xX'],1))
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
            #self.OP[0,:,:,0] /= OP_0_max
            #self.OP[0,:,:,1] /= OP_1_max
            #self.DF /= DF_max
            #self.QF /= QF_max
            
            
            if 'scaleFL0' in self.params:
                for i in range(self.params['nF']):
                    scaleN = 'scaleFL'+ str(i)
                    self.FL[:,:,:,i] *= self.params[scaleN]
                    print(np.mean(self.FL[:,:,:,i]))
            else:
                self.FL *= self.params['scaleFL']            
                #self.FL[:,:,:,:] /= FL_max

            
        
        # Apply flipping and rotation for Case 2 and 3 (for irregularly-shaped test data):
        if isTesting==True:
            self.FL = np.rot90(self.FL,1,(1,2))  #rotate from axis 2 to axis 3 (first 33 (xX) to second 33 (yY))
            self.FL = np.fliplr(self.FL) #left-right flip  
            self.DF = np.rot90(self.DF,1,(1,2))  
            self.DF = np.fliplr(self.DF) 
            self.temp_DF_pre_conversion = np.rot90(self.temp_DF_pre_conversion,1,(1,2))  
            self.temp_DF_pre_conversion = np.fliplr(self.temp_DF_pre_conversion) 

            self.QF = np.rot90(self.QF,1,(1,2))
            self.QF = np.fliplr(self.QF) 
            self.OP = np.rot90(self.OP,1,(1,2)) 
            self.OP = np.fliplr(self.OP) 
            
        #self.FLRE = self.FL/self.RE
        
    def drop_out(self, x, drop_out = None):
        if drop_out: 
            x = Dropout(drop_out)(x, training = True)
    
        return x 

    
    def Model(self):
        """The deep learning architecture gets defined here"""
        if self.isTesting: 
            drop_out = 0.5
        else: 
            drop_out = None

        print("drop_out", drop_out)
        ## Input Optical Properties ##
        inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],2))
        #inOP = OpData
        ## Input Multi-Dimensional Fluorescence ##
        inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))
        #inFL = FlData
        print("inOP_beg shape: ", inOP_beg.shape)
        print("inFL_beg shape: ", inFL_beg.shape)
        
        ## NOTE: Batch normalization can cause instability in the validation loss

        ## Optical Properties Branch ##
        inOP = Conv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)
        #outOP1 = inOP
        inOP = BatchNormalization()(inOP)  
        #inOP = self.drop_out(inOP, drop_out) #drop out 1
        
        print("inOP 1 shape: ", inOP.shape)

        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        
        print("inOP 2 shape: ", inOP.shape)

        #outOP2 = inOP
        #inOP = self.drop_out(inOP, drop_out) #drop out 3

        inOP = BatchNormalization()(inOP)
        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        
        print("inOP 3 shape: ", inOP.shape)

        #outOP3 = inOP
        #inOP = self.drop_out(inOP, drop_out) #drop out 3

        inOP = BatchNormalization()(inOP)
        inOP = self.resblock_2D(int(self.params['nFilters2D']/2), self.params['kernelResBlock2D'], self.params['strideConv2D'], inOP)
        #inOP = BatchNormalization()(inOP)
        
        print("inOP 4 shape: ", inOP.shape)


        ## Fluorescence Input Branch ##
        input_shape = inFL_beg.shape
        inFL = Conv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)
        
        print("inFL 1 shape: ", inFL.shape)

   
        inFL = BatchNormalization()(inFL)
        inFL = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        print("inFL 2 shape: ", inFL.shape)

        inFL = BatchNormalization()(inFL)
        inFL = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        print("inFL 3 shape: ", inFL.shape)

        #outFL3 = inFL
        #inFL = self.drop_out(inFL, drop_out) #drop out 3

        inFL = BatchNormalization()(inFL)
        inFL = self.resblock_2D(int(self.params['nFilters2D']/2), self.params['kernelResBlock2D'], self.params['strideConv2D'], inFL)
        #inFL = BatchNormalization()(inFL)
        
        print("inFL 4 shape: ", inFL.shape)

        
        ## Reshape ##
        #zReshape = int(((self.params['nFilters2D']/2)*self.params['nF'])/self.params['strideConv2D'][2])
        #inFL = Reshape((self.params['xX'],self.params['yY'],zReshape))(inFL)

        ## Concatenate Branch ##
        concat = concatenate([inOP,inFL],axis=-1)
        #concat = self.drop_out(concat, drop_out) #drop out 3
        #concat = BatchNormalization()(concat)

        concat = SeparableConv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], 
                                 strides=self.params['strideConv2D'], padding='same', activation=self.params['activation'], 
                                 data_format="channels_last")(concat)
        
        #concat = self.drop_out(concat, drop_out) #drop out 3

        concat = BatchNormalization()(concat)
        concat = self.resblock_2D(self.params['nFilters2D'], self.params['kernelResBlock2D'], self.params['strideConv2D'], concat) 
        #concat = self.drop_out(concat, drop_out) #drop out 3

        ## Quantitative Fluorescence Output Branch ##
        outQF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(concat)
        #outQF = self.drop_out(outQF, drop_out) #drop out 3

        #first DF layer 
        outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(concat)
  
        #add reslayer 
        #out_QF_DF = add([outQF, outDF])

        outQF = BatchNormalization()(outQF)
        outQF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(outQF)        
        
        outQF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        data_format="channels_last")(outQF)

        ## Depth Fluorescence Output Branch ##
        #outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
        #               activation=self.params['activation'], data_format="channels_last")(concat)
        #outDF = self.drop_out(outDF, drop_out) #drop out 3

        outDF = BatchNormalization()(outDF)
        outDF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(outDF)       
        outDF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        data_format="channels_last")(outDF)

        ## Defining and compiling the model ##
        self.modelD = Model(inputs=[inOP_beg,inFL_beg], outputs=[outQF, outDF])#,outFL])
        self.modelD.compile(loss=['mae', 'mae'],
                      optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                      metrics=['mae', 'mae'])
        self.modelD.summary()
        ## Outputs for feature maps ##
        #self.modelFM = Model(inputs=[OpData,FlData], outputs=[FlData, OpData, outFL1, outOP1, outFL2, outOP2, outFL3, outOP3]) 
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
        
        
        predict = self.modelD.predict([self.RE, self.FL])  
        if loadandpredict==True:
            predict = self.new_model.predict([self.OP, self.FL])
        QF_P = predict[0] 
        DF_P = predict[1]
        QF_P /= self.params['scaleQF']
        DF_P /= self.params['scaleDF']
        
        #DF_P_max = 23.361952
        #QF_P_max = 9.998467
        


        print('\n Making predictions on the test data... \n')  
        
        save_dict = {}
        
        while True:
            self.save = input('\n Save these predictions? (Y/N) ')
            if self.save in ['Y','y']:
                self.exportName = input('The model, best weights, and some assorted parameters and other information will be saved during fitting; enter a name for the export directory or leave this field blank if you do not want to export (note that if you do not export, you will not be able to plot your results in the future once the current history variable is lost from memory, however you can still plot the results now): ')
                new_path = 'Predictions//' + self.exportName
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                    
                if len(self.exportName) > 0:
                    # Export MATLAB file to case
                    self.exportPath = 'Predictions//'+self.exportName+ '//' + self.exportName+'_DF_and_QF.mat'
                    a = np.array(DF_P[:,:,:,0])
                    b = np.array(QF_P[:,:,:,0])
                    scipy.io.savemat(self.exportPath,{'DF_pred':a,"QF_pred":b})
                    
                    self.exportPath = 'ModelParameters//'+self.exportName+'//'+self.case
                    #save dictionary as excel file 
                    #fileName = self.exportPath+'_params.xml'
                    self.paramsexcelfileName = self.exportPath + '_params.xlsx'
                    
                    if os.path.isfile(self.paramsexcelfileName):
                        df_excel = pd.read_excel(self.paramsexcelfileName)
                        if "training_file_name" in df_excel.keys():
                            save_dict["training_file_name"] = df_excel["training_file_name"]
                    
                        
                    save_dict["testing_file_name"] = self.params["training_file_name"]
                    
                    log_new_path = 'Predictions//'+self.exportName+ '//' + 'prediction_info.log'
                    
                    predict = self.modelD.predict([self.OP, self.FL])  
                    QF_P = predict[0] 
                    DF_P = predict[1]
                    QF_P /= self.params['scaleQF']
                    DF_P /= self.params['scaleDF']  
                    ## Error Stats
                    # Average error
                    DF_error = DF_P - self.DF
                    QF_error = QF_P - self.QF
                    DF_erroravg = np.mean(abs(DF_error[self.indxIncl]))
                    DF_errorstd = np.std(abs(DF_error[self.indxIncl]))
                    QF_erroravg = np.mean(abs(QF_error[self.indxIncl]))
                    QF_errorstd = np.std(abs(QF_error[self.indxIncl]))
                    
                    save_dict["average depth error, SD"] = (DF_erroravg, DF_errorstd)
                    save_dict["average concentration erro, SD"] = (QF_erroravg, QF_errorstd)
                                        
                    predictionParams = dicttoxml(str(save_dict))
                    with open(log_new_path,'w') as paramsFile:
                        paramsFile.write(parseString(predictionParams).toprettyxml("    "))
                    
                    #with open(excel_new_path, "wb"):
                    #    f.write(save_dict)
                break
            elif self.save in ['N','n']:
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
                        self.modelD = load_model(loadFile) # Load the weights into this model
                        # Whether the model was already defined properly, or did not exist, or needed to be redefined, we now need to load in data for testing and then make predictions based on the loaded weights 
                        self.importData(isTesting=True,quickTest=False)
                        self.Predict() # Make predictions
                        self.Evaluate()
                        break
                    elif checkParams in ['N','n']: # If the user decides that the model is not defined using the correct parameters
                        print('\nI suggest referring to the .xml file associated with your saved weights to determine which parameters you need to change. Calling the Params() method now, and modelling immediately following the changes: \n') # Refer to .xml params
                        self.Params() # Allow the user to define new parameters
                        #self.Model() # Model using these new, presumable correct, parameters
                        # Load in model architecture with weights (not the same architecture as the Model function)
                        self.new_model = tf.keras.models.load_model(loadFile)
                        print('Now loading weights: ')
                        self.new_model.load_weights(loadFile) # Load the weights into this model
                        self.new_model.summary()

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
                self.xml_file_name = loadFile.replace(".h5", "_params.xml")
                self.new_model = tf.keras.models.load_model(loadFile, compile=False)
                self.new_model.load_weights(loadFile)
                self.modelD = load_model(loadFile, compile=False) 
                #self.new_model.summary()# Load the weights into this model
                while True:
                    self.importData(isTesting=True,quickTest=True)
                    self.Predict()
                    self.Results()
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
        num_data_set_display = 10
        for x in range(num_data_set_display):
            sample = x
            if dims ==5:
                if img in ['FL','fl']: # Print fluorescence maps
                    fx = self.params['nF'] # Number of spatial frequencies
                    for i in range(fx): 
                        plt.figure()
                        plt.imshow(self.FL[sample,:,:,i,:])
                        plt.colorbar()
                        plt.clim(0,self.FL[sample,:,:,i,:].max())
                        plt.show()
                elif img in ['OP','op']: # Print optical properties
                    for i in range(2):
                        plt.figure()
                        plt.imshow(self.OP[sample,:,:,i])
                        plt.colorbar()
                        plt.clim(0,self.OP[sample,:,:,i].max())
                        plt.imshow(self.OP[sample,:,:,i])
                        plt.show()
                elif img in ['DF','df']: # Print depth maps
                    for i in range(1):
                        plt.figure()
                        plt.imshow(self.DF[sample,:,:,i])
                        plt.colorbar()
                        plt.clim(0,self.DF[sample,:,:,i].max())
                        plt.imshow(self.DF[sample,:,:,i])
                        plt.show()
                elif img in ['QF','qf']: # Print concentration maps
                    for i in range(1):
                        
                        plt.imshow(self.QF[sample,:,:,i])
                        plt.show()
                elif img in ['RE','re']: # Print concentration maps
                    for i in range(6):
                        plt.figure()
                        print(self.RE.shape)
                        plt.imshow(self.RE[i,:,:,sample])
                        plt.colorbar()
                        plt.clim(0,self.RE[i,:,:,sample].max())
                        plt.imshow(self.RE[i,:,:,sample])
                        plt.show()
                
        else:
            if img in ['FL','fl']: # Print fluorescence maps
                fx = self.params['nF'] # Number of spatial frequencies
                for i in range(fx): 
                    plt.figure()
                    plt.imshow(self.FL[i,:,:,0])
                    plt.colorbar()
                    plt.clim(0,self.FL[i,:,:,0].max())
                    plt.show()
            elif img in ['OP','op']: # Print optical properties
                for i in range(2):
                    plt.figure()
                    plt.imshow(self.OP[i,:,:,0])
                    plt.colorbar()
                    plt.clim(0,self.OP[i,:,:,0].max())
                    plt.imshow(self.OP[i,:,:,0])
                    plt.show()
            elif img in ['DF','df']: # Print depth maps
                for i in range(1):
                    plt.imshow(self.DF[:,:,0])
                    plt.show()
            elif img in ['QF','qf']: # Print concentration maps
                for i in range(1):
                    plt.imshow(self.QF[:,:,0])
                    plt.show()
    
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
    
    def get_min(self, DF):
        #DF = np.reshape(DF,(self.params['xX'],self.params['yY'],np.shape(DF)[0]))
    
        print(DF.shape)
        DF_zeros = np.array(DF)

        for x in range(DF.shape[0]):
            for i in range(DF.shape[1]):
                
                #DF_zeros_per_column = DF[x, i, :] == self.background_val
                #DF_zeros[x, i, DF_zeros_per_column] = np.nan
                pass
                           
        DF_min_per_case = np.nanmin(DF_zeros, axis = (1,2))
        
        return DF_min_per_case
    
    def generate_3d(self, z):
        bottom = np.zeros_like(z.shape[0])
        width = depth = 1
        x_dim = z.shape[0]
        y_dim = z.shape[1]
        
        x_val = []
        y_val = []
        
        for x in range(x_dim):
            x_val += [x] * y_dim
            
        
        x = np.linspace(0, x_dim, x_dim)
        y = np.linspace(0, y_dim, x_dim)

        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.bar3d(x,y,bottom, width, depth, z)
        ax.set(xticklabels=[],
        yticklabels=[],
               zticklabels=[])
        plt.show()
        
    def classify(self, DF_P):      
        
        DF_min_per_case = np.nanmin(DF_P[:,:,:,0], axis = (1,2))
        CL = DF_min_per_case < 5 # 5mm or less would be one 
        #CL = np.reshape(self.CL, (1, self.CL.shape[0]))
        
        print(CL.shape)
        return CL
    
    def error_vs_absorption(self, y, y_p, absorption):
        #use this for analysis of min depth predictions 
        #err = np.abs(y-y_p)
        err = np.array(y_p- y)
         
        plt.scatter(absorption,err, s = 1, c = y, cmap='jet')
        plt.ylabel("Depth Error (mm) : prediction - actual")
        plt.xlabel("Absorption (1/mm)")
        plt.colorbar()

        plt.show()
        
    def error_vs_scattering(self, y, y_p, scattering):
        #use this for analysis of min depth predictions 
        #err = np.abs(y-y_p)
        err = np.array(y_p- y)

                 
        plt.scatter(scattering,err, s = 1, c = y, cmap='jet')
        plt.ylabel("Depth Error (mm) : prediction - actual")
        plt.xlabel("Scattering (1/mm)")
        plt.colorbar()

        plt.show()
        
    def scatter_vs_absorption_failed_result(self, y, y_p, absorption, scattering, predicted_incorrectly):
        
        #err = np.abs(y-y_p)
        err = np.array(y_p- y)

        scattering = np.array(scattering)
        scattering = np.reshape(scattering, (scattering.shape[0], -1))
        absorption = np.array(absorption)
        absorption = np.reshape(absorption, (absorption.shape[0], -1))

         
        plt.scatter(scattering,absorption, s = 1)
        plt.ylabel("Scattering (1/mm)")
        plt.xlabel("Absorption (1/mm)")
        plt.scatter(scattering[predicted_incorrectly],absorption[predicted_incorrectly], s=1, color = ['red'])

        plt.show()
    
    def concentration_vs_depth_error(self, DF, DF_P, QF):
        err = np.array(DF_P- DF)     
        plt.scatter(QF,err, s = 1, c = DF, cmap='jet')
        plt.ylabel("Depth Error (mm) : prediction - actual")
        plt.xlabel("Concentration (ug/ml)")
        plt.colorbar()

        plt.show()
        
    def concentration_error_vs_depth_error(self, DF, DF_P, QF, QF_P):
        
        x = DF_P- DF
        y = QF_P - QF
        err_DF = np.array(np.abs(x)) 
        err_Q = np.array(np.abs(y)) 

        plt.scatter(err_Q,err_DF, s = 1, c = DF, cmap='jet')
        plt.ylabel("Absolute Depth Error (mm) ")
        plt.xlabel("Absolute Concentration Error (ug/ml)")
        plt.colorbar()

        plt.show()
        
    def concentration_error_vs_depth_error_classify(self, DF, DF_P, QF, QF_P, predicted_incorrectly):
        
        x = DF_P- DF
        y = QF_P - QF
        err_DF = np.array(np.abs(x)) 
        err_Q = np.array(np.abs(y)) 
        
        
        err_DF = np.reshape(err_DF, (err_DF.shape[0], -1))
        err_Q = np.reshape(err_Q, (err_Q.shape[0], -1))

        plt.scatter(err_Q,err_DF, s = 1)
        plt.scatter(err_Q[predicted_incorrectly],err_DF[predicted_incorrectly], s = 1, color = ['red'])

        plt.ylabel("Absolute Depth Error (mm) ")
        plt.xlabel("Absolute Concentration Error (ug/ml)")
        plt.colorbar()

        plt.show()
        
        
    def concentration_vs_depth_error_classify(self, DF, DF_P, QF, predicted_incorrectly):
        err = np.array(DF_P- DF)     
        plt.scatter(QF,err, s = 1)
        
        QF = np.array(QF)
        QF = np.reshape(QF, (QF.shape[0], -1))
        err = np.array(err)
        err = np.reshape(err, (err.shape[0], -1))
        plt.scatter(QF[predicted_incorrectly],err[predicted_incorrectly], s = 1, color = ['red'])

        plt.ylabel("Depth Error (mm) : prediction - actual")
        plt.xlabel("Concentration (ug/ml)")
        plt.colorbar()

        plt.show()
        
    def max_concentration_graph(self, QF, QF_P):
        
        concentration_graph = plt.figure()
        plt.scatter(QF,QF_P, s = 1)
        plt.plot(plt.xlim([0, 15]), plt.ylim([0, 15]),color='k')

        plt.ylabel("Max Predicted Concentration (ug/ml)")
        plt.xlabel("Max True Concentration (ug/ml)")
        
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        
        plt.tight_layout()
        concentration_graph.show()
        
        
        
    def concentration_error_vs_concentration(self, QF, QF_P, predicted_incorrectly):
        err_Q_graph = plt.figure()
        err_Q = np.array(np.abs(QF_P-QF)) 
        err_Q = np.reshape(err_Q, (err_Q.shape[0], -1))
                                   
        QF = np.array(QF)
        QF = np.reshape(QF, (QF.shape[0], -1))
                                   
        plt.scatter(QF,err_Q, s = 1)
        
        plt.scatter(QF[predicted_incorrectly],err_Q[predicted_incorrectly], s = 1, color = ['red'])
        plt.ylabel("Absolute Concentration Error (mm)")
        plt.xlabel("Concentration (ug/ml)")
        plt.colorbar()

        err_Q_graph.show()
        
    def concentration_error_vs_depth(self, QF, QF_P, DF):
        err_Q_graph = plt.figure()
        err_Q = np.array(np.abs(QF_P-QF)) 
        #err_Q = np.reshape(err_Q, (err_Q.shape[0], -1))
                                                            
        plt.scatter(DF[self.indxIncl],err_Q[self.indxIncl], s = 1)
        plt.ylabel("Absolute Concentration Error (ug/ml))")
        plt.xlabel("True Depth (mm)")
        plt.colorbar()

        err_Q_graph.show()    
        
    def plot_loss_curve_from_log(self):
        h5_files = []
        loss_vals = []
        for folder in os.listdir("ModelParameters"):
            if not folder.endswith((".h5",".log",".xml")):
                for file in os.listdir("ModelParameters//"+folder):
                    if file.endswith(".log"):
                        filename = "ModelParameters//"+folder+'//'+file
                        h5_files.append(filename)
                        print(filename)        
        while True: # Loop until user makes acceptable choice for the input directory
            loadFile = input('Enter the general and specific directory pertaining to the .h5 (weights) file you would like to load: ')
            if loadFile == '': # User enters nothing; break
                break
            elif loadFile in h5_files:
                with open(loadFile) as f:
                    f = f.readlines()
                    
                for line in f:
                    val_list = line.split(',')
                    val = val_list[len(val_list)-1]
                    val = val.replace('\n', '')
                    loss_vals.append(val)
                    
                break
                    
                    
            else: # If the modelD attribute does not exist
                print('\nModel is not currently defined - select again.') 
                break
        
        loss_vals = loss_vals[1:] #excliude the 'val_loss' keyword at the beginning
        
        loss_vals = [float(vals) for vals in loss_vals]
        
        epoch_num = list(range(0,len(loss_vals)))
                          
        plt.plot(epoch_num, loss_vals)
        plt.ylabel("loss value")
        plt.xlabel("epoch #")
        plt.ylim([0, 1])
      
    
    
    def confidence_interval(self, metric, num_data):        
        z_95 = 1.96 
        
        
        CI = z_95 * np.sqrt(metric* (1- metric) / (num_data))
            
        return CI
    
    def calculate_bar(self, min_depth_error, DF, num_bins):
        
        min_depth_error = np.array(min_depth_error)
        ordered_min_depth_error = np.array([x for _, x in sorted(zip(DF,min_depth_error))])
        sorted_DF = np.array(sorted(DF))
        
        indexes_start= np.zeros(num_bins)

        
        mean_vals = np.zeros(num_bins)
        std_vals = np.zeros(num_bins)
        for i in range(1, len(indexes_start)+1):
            
            if i == len(indexes_start):
                sorted_DF_bool = np.logical_and((i - 1 <= sorted_DF), (sorted_DF <= i))

            else: 
                sorted_DF_bool = np.logical_and((i - 1 <= sorted_DF), (sorted_DF < i))
                
            min_depth_error_of_interest = ordered_min_depth_error[sorted_DF_bool]
            
            
            if len(min_depth_error_of_interest) > 0:
                mean_vals[i-1] = (np.mean(ordered_min_depth_error[sorted_DF_bool]))
                std_vals[i-1] = (np.std(ordered_min_depth_error[sorted_DF_bool]))

        return mean_vals, std_vals
        
    def count_predictions_of_zero(self, DF_P):
        bool_DF_P_equals_zero = DF_P == 0

        return np.sum(bool_DF_P_equals_zero)
    
    def display_parameters(self):
        
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

    def min_depth_error_as_function_of_depth(self, DF_max, DFP_max):
        min_depth_error = np.abs(DF_max - DFP_max)
        radius = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        min_depth_error_f_radius = []
        min_depth_error_std_f_radius = []

        for i in range(len(radius)):
            min_depth_error_f_radius.append(np.mean(min_depth_error[i*10:i*10+10]))
            min_depth_error_std_f_radius.append(np.std(min_depth_error[i*10:i*10+10]))

        min_depth_error_plt = plt.figure()
        plt.bar(radius, min_depth_error_f_radius, width = 0.15)
        plt.errorbar(radius,min_depth_error_f_radius , yerr=min_depth_error_std_f_radius, capsize=3, fmt="r--o", ecolor = "black")
        min_depth_error_plt.show()

        plt.ylabel("average min depth prediction error (mm)")
        plt.xlabel("radius (scale factor)")
        plt.title("Min Depth Error vs Radius")

    
    def depth_error_as_function_of_depth(self, DF_max, DFP_max):
        look_up_scale = [0.1, 0.2, 0.3, 0.4, 0.05, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1, 1.1, 1.2, 1.3, 1.4, 1.05, 1.5, 1.6, 1.7, 1.8, 1.9, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 1, 2]
        radius = [look_up_scale[x] for x in range(len(look_up_scale))] 
                
        #norm = matplotlib.colors.Normalize(vmin=1, vmax=20)
        #cmap = matplotlib.cm.jet
        #m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        #color_vals = [m.to_rgb(x) for x in radius]

        cmap = matplotlib.cm.jet
        norm = matplotlib.colors.Normalize(vmin=0.1, vmax=2)
        color_vals = [cmap(norm(x)) for x in radius]

        depth_plt = plt.figure()
        for i in range(len(radius)):
            label_str = "s: " + str(radius[i])
            plt.scatter(DF_max[i*19:i*19+19], DFP_max[i*19:i*19+19], s = 2, label=label_str, norm = norm, cmap = cmap, c = [color_vals[i]])
            plt.legend(loc="upper left", prop={'size': 3})
        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("Min Depth")
        plt.xlim([0, 11])
        plt.ylim([0, 11])
        plt.plot(plt.xlim([0, 11]), plt.ylim([0, 11]),color='k')
        depth_plt.show()

    def depth_error_as_function_of_thickness_and_depth(self, DF, DF_P):
        #DF_error = np.abs(DF-DF_P)
        thickness = [x for x in range(1, 16)]
        #depths = [x for x in range(1,11)]

        cmap = matplotlib.cm.jet
        norm = matplotlib.colors.Normalize(vmin=min(thickness), vmax=max(thickness))
        color_vals = [cmap(norm(x)) for x in thickness]
        depth_plt = plt.figure()

        for i in range(len(thickness)):
            if i in [0, 4, 9, 14]:
                label_str = "t: " + str(thickness[i])
                plt.scatter(DF[i*10:i*10+10], DF_P[i*10:i*10+10], s = 2, label=label_str, norm = norm, cmap = cmap, c = [color_vals[i]])
                plt.legend(loc="upper left", prop={'size': 3})

        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("Min Depth")
        plt.xlim([0, 11])
        plt.ylim([0, 11])
        plt.plot(plt.xlim([0, 11]), plt.ylim([0, 11]),color='k')
        depth_plt.show()

    def depth_error_as_function_of_thickness_and_depth_all(self, DF, DF_P, QF):
        #DF_error = np.abs(DF-DF_P)
        thickness = [x for x in range(1, 16)]
        #depths = [x for x in range(1,11)]

        cmap = matplotlib.cm.jet
        norm = matplotlib.colors.Normalize(vmin=min(thickness), vmax=max(thickness))
        color_vals = [cmap(norm(x)) for x in thickness]
        depth_plt = plt.figure()

        for i in range(len(thickness)):
            if i in [0, 2, 4, 9, 14]:
                label_str = "t: " + str(thickness[i])
                x = DF[i*10:i*10+10, :,:]
                y = DF_P[i*10:i*10+10, :,:]
                non_zero_indx = np.nonzero(self.temp_DF_pre_conversion[i*10:i*10+10,:,:])

                plt.scatter(x[non_zero_indx], y[non_zero_indx], s = 2, label=label_str, norm = norm, cmap = cmap, c = [color_vals[i]])
                plt.legend(loc="upper left", prop={'size': 6})

        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("Overall Depth")
        plt.xlim([0, 11])
        plt.ylim([0, 11])
        plt.plot(plt.xlim([0, 11]), plt.ylim([0, 11]),color='k')
        depth_plt.show()


    def concentration_error_as_function_of_thickness_and_depth(self, DF, QF, QF_P):

        QF_error = (QF_P-QF)
        thickness = [x for x in range(1, 16)]
        depths = [x for x in range(1,11)]
        

        cmap = matplotlib.cm.jet
        norm = matplotlib.colors.Normalize(vmin=min(thickness), vmax=max(thickness))
        color_vals = [cmap(norm(x)) for x in thickness]
        depth_plt = plt.figure()

        for i in range(len(thickness)):
            label_str = "t: " + str(thickness[i])
            x = DF[i*10:i*10+10, :,:]
            y = QF_error[i*10:i*10+10,:,:]
            non_zero_indx = np.nonzero(self.temp_DF_pre_conversion[i*10:i*10+10,:,:])
            #plt.scatter(DF[i*10:i*10+10, :,:], QF_error[i*10:i*10+10,:,:], s = 2, label=label_str, norm = norm, cmap = cmap, c = [color_vals[i]])
            if i in [0, 2, 4, 9, 14]:
                plt.scatter(x[non_zero_indx], y[non_zero_indx], s = 2, label=label_str, norm = norm, cmap = cmap, c = [color_vals[i]])

                plt.legend(loc="upper left", prop={'size': 6})

        plt.ylabel("Concentration Error (Prediction - True) (ug/ml)")
        plt.xlabel("True Depth (mm)")
        plt.title("Concentration Error vs Depth")
        plt.xlim([0, 11])
        plt.ylim([-10, 10])
        plt.axhline(y = 0, color = 'k', linestyle = '--') 
        depth_plt.show()    

    def depth_error_as_function_of_fluorophore_thickness_and_depth(self, DF, DF_P):
        #thickness changes going down the row 
        thickness = np.linspace(0.5, 3, num = 6)
        depth = 10

        cmap = matplotlib.cm.jet
        norm = matplotlib.colors.Normalize(vmin=min(thickness), vmax=max(thickness))
        color_vals = [cmap(norm(x)) for x in thickness]
        depth_plt = plt.figure()
        for i in range(len(thickness)):
            indexes = [x * 6 + i for x in range(10)]
            label_str = "t: " + str(thickness[i])
            x = DF[indexes]
            y = DF_P[indexes]
            plt.scatter(x, y, s = 2, label=label_str, norm = norm, cmap = cmap, c = [color_vals[i]])
            plt.legend(loc="upper left", prop={'size': 6})

        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("Min Depth")
        plt.xlim([0, 11])
        plt.ylim([0, 11])
        plt.plot(plt.xlim([0, 11]), plt.ylim([0, 11]),color='k')
        depth_plt.show()
        
    def concentration_error_as_function_of_fluorophore_thickness_and_depth(self, DF, QF, QF_P):
        #thickness changes going down the row 
        thickness = np.linspace(0.5, 3, num = 6)

        #QF_error = np.mean(QF_P[self.indxIncl] - QF[self.indxIncl], axis = (1,2))

        cmap = matplotlib.cm.jet
        norm = matplotlib.colors.Normalize(vmin=min(thickness), vmax=max(thickness))
        color_vals = [cmap(norm(x)) for x in thickness]
        depth_plt = plt.figure()
        for i in range(len(thickness)):
            indexes = [x * 6 + i for x in range(10)]
            label_str = "t: " + str(thickness[i])
            x = DF[indexes]
            y = QF_P[indexes] - QF[indexes]
            plt.scatter(x, y, s = 2, label=label_str, norm = norm, cmap = cmap, c = [color_vals[i]])
            plt.legend(loc="upper left", prop={'size': 3})

        plt.ylabel("Concentration Error (Prediction - True) (ug/ml)")
        plt.xlabel("True Depth (mm)")
        plt.title("Concentration Error vs Depth")
        plt.xlim([0, 11])
        plt.ylim([-10, 10])
        depth_plt.show()

    
    def depth_error_as_function_of_fluorophore_thickness_and_depth(self, DF, DF_P):
        #thickness changes going down the row 
        concentration = ["1", "3", "7", "10"]
        mus = ["1", "1.5", "2"]
        fHb = ["0.5", "1", "1.5", "2"]
        depth = 10

        cmap = matplotlib.cm.jet
        #norm = matplotlib.colors.Normalize(vmin=min(thickness), vmax=max(thickness))
        #color_vals = [cmap(norm(x)) for x in thickness]
        depth_plt = plt.figure()
        # for i in range(len(thickness)):
        #     indexes = [x * 6 + i for x in range(10)]
        #     label_str = "t: " + str(thickness[i])
        #     x = DF[indexes]
        #     y = DF_P[indexes]
        #     plt.scatter(x, y, s = 2, label=label_str, norm = norm, cmap = cmap, c = [color_vals[i]])
        #     plt.legend(loc="upper left", prop={'size': 6})

        # plt.ylabel("Predicted Depth (mm)")
        # plt.xlabel("True Depth (mm)")
        # plt.title("Min Depth")
        # plt.xlim([0, 11])
        # plt.ylim([0, 11])
        # plt.plot(plt.xlim([0, 11]), plt.ylim([0, 11]),color='k')
        # depth_plt.show()
        
        
    def predict_uncertainty(self): 
        self.isTesting = True
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

                self.Model()
                self.modelD.load_weights(loadFile)

                #self.no_dropout_model = tf.keras.models.load_model(loadFile, compile=False)
                #self.no_dropout_model.load_weights(loadFile)
                #self.modelD = load_model(loadFile, compile=False) 
                #self.new_model.summary()# Load the weights into this model
                self.importData(isTesting=True,quickTest=True)
                break
                    
            else: # If the modelD attribute does not exist
                print('\nModel is not currently defined - select again.') 
                break

        

        num_dropout_ensembles = 16
        num_examples = self.OP.shape[0]
        image_dim = self.OP.shape[1]

        depth_prediction_ensembles_no_dropout = np.ndarray((num_examples, image_dim, image_dim, num_dropout_ensembles))
        concentration_prediction_ensembles_no_dropout = np.ndarray((num_examples, image_dim, image_dim, num_dropout_ensembles))
        
        depth_prediction_ensembles = np.ndarray((num_examples, image_dim, image_dim, num_dropout_ensembles))
        concentration_prediction_ensembles = np.ndarray((num_examples, image_dim, image_dim, num_dropout_ensembles))

        for i in range(num_dropout_ensembles):
            #tmp = self.modelD.predict([self.OP, self.FL])  
            #depth_prediction_ensembles_no_dropout[:,:,:,i] = tmp[1].squeeze()
            #concentration_prediction_ensembles_no_dropout[:,:,:,i] = tmp[0].squeeze()
            tmp = self.modelD.predict([self.OP, self.FL])  
            depth_prediction_ensembles[:,:,:,i] = tmp[1].squeeze()
            concentration_prediction_ensembles[:,:,:,i] = tmp[0].squeeze()

        depth_prediction_mean = np.zeros((num_examples, image_dim, image_dim))
        concentration_prediction_mean = np.zeros((num_examples, image_dim, image_dim))

        depth_prediction_uncertainty = np.zeros((num_examples, image_dim, image_dim))
        concentration_prediction_uncertainty = np.zeros((num_examples, image_dim, image_dim))

        #take the mean of each example 
        for i in range(num_examples):
            depth_prediction_mean[i, :,:] = np.mean(depth_prediction_ensembles[i,:,:,:].squeeze(), axis = 2)
            concentration_prediction_mean[i, :,:] = np.mean(concentration_prediction_ensembles[i,:,:,:].squeeze(), axis = 2)
            #get epistemic uncertainty 
            depth_prediction_uncertainty[i, :,:] = np.std(depth_prediction_ensembles[i,:,:,:].squeeze(), axis = 2) 
            concentration_prediction_uncertainty[i, :,:] = np.std(concentration_prediction_ensembles[i,:,:,:].squeeze(), axis = 2)


        DF_error = depth_prediction_mean - self.DF.squeeze()
        QF_error = concentration_prediction_mean - self.QF.squeeze()
        depth_prediction_uncertainty = np.array(depth_prediction_uncertainty)
        #plot min depth graph 
        error_min_depth = np.zeros((depth_prediction_uncertainty.shape[0], 1))
        
        #for i in range(depth_prediction_uncertainty.shape[0]):
        #    x = (depth_prediction_uncertainty[i,:,:] == np.min(depth_prediction_uncertainty[i,:,:]))
        #    min_index = np.where(depth_prediction_uncertainty[i,:,:] == np.min(depth_prediction_uncertainty[i,:,:]))
        #    error_min_depth[i] = depth_prediction_uncertainty[i,min_index[0][0], min_index[1][0]]

        

        #self.indxIncl = np.nonzero(self.temp_DF_pre_conversion)
        #print(self.indxIncl)
        #print(self.indxIncl.shape)
        #self.indxIncl = self.indxIncl.squeeze()
        #depth_error = abs(self.DF[self.indxIncl] - depth_prediction_mean[self.indxIncl])
        '''
        depth_graph = plt.figure()

        DF_min = self.get_min(self.DF)
        DF_P_min = self.get_min(depth_prediction_mean)
        print(error_min_depth)
        print(DF_min)
        print(DF_P_min)
        #plt.scatter(depth_error,depth_prediction_uncertainty[self.indxIncl],s=1)
        #plt.scatter(DF_min,DF_P_min,s=1)
        plt.errorbar(DF_min, DF_P_min, yerr=error_min_depth.squeeze(), fmt="o")
        plt.xlim([0, 15])
        plt.ylim([0, 15])
        plt.plot(plt.xlim([0, 15]), plt.ylim([0, 15]),color='k')
        plt.ylabel("Depth Error (mm)")
        plt.xlabel("Predicted Depth Uncertainty (mm)")
        plt.title("Depth Uncertainty vs Depth Error")
        plt.tight_layout()
        
        depth_graph.show() 
        '''
        
        for i in range(10):
            fig, axs = plt.subplots(2,4)
            plt.set_cmap('jet')
            plt.colorbar(axs[0,0].imshow(self.DF[i,:,:],vmin=0,vmax=15), ax=axs[0, 0],fraction=0.046, pad=0.04)
            axs[0,0].axis('off')
            axs[0,0].set_title('True Depth (mm)')
            
            plt.colorbar(axs[0,1].imshow(depth_prediction_mean[i,:,:],vmin=0,vmax=15), ax=axs[0, 1],fraction=0.046, pad=0.04)
            axs[0,1].axis('off')
            axs[0,1].set_title('Predicted Depth (mm)')
            plt.colorbar(axs[0,2].imshow(abs(DF_error[i,:,:]),vmin=0,vmax=15), ax=axs[0, 2],fraction=0.046, pad=0.04)
            axs[0,2].axis('off')
            axs[0,2].set_title('|Error (mm)|')
            plt.colorbar(axs[0,3].imshow(depth_prediction_uncertainty[i,:,:],vmin=0,vmax=3), ax=axs[0, 3],fraction=0.046, pad=0.04)

            axs[0,3].axis('off')
            axs[0,3].set_title('|Uncertainty Pred (mm)|')
            plt.colorbar(axs[1,0].imshow(self.QF[i,:,:],vmin=0,vmax=10), ax=axs[1, 0],fraction=0.046, pad=0.04)

            axs[1,0].axis('off')
            axs[1,0].set_title('True Conc (ug/mL)')
            plt.colorbar(axs[1,1].imshow(concentration_prediction_mean[i,:,:],vmin=0,vmax=10), ax=axs[1, 1],fraction=0.046, pad=0.04)
            axs[1,1].axis('off')
            axs[1,1].set_title('Predicted Conc (ug/mL)')
            plt.colorbar(axs[1,2].imshow(abs(QF_error[i,:,:]),vmin=0,vmax=10), ax=axs[1, 2],fraction=0.046, pad=0.04)
            #axs[0,2].text(5, 5, 'min_depth error = ' + str(temp_value_str_1 - temp_value_str_2), bbox={'facecolor': 'white', 'pad': 4}, size = 'x-small')

            axs[1,2].axis('off')
            axs[1,2].set_title('|Error (ug/mL)|')
            
            plt.colorbar(axs[1,3].imshow(concentration_prediction_uncertainty[i,:,:],vmin=0,vmax=3), ax=axs[1, 3],fraction=0.046, pad=0.04)
            #axs[0,2].text(5, 5, 'min_depth error = ' + str(temp_value_str_1 - temp_value_str_2), bbox={'facecolor': 'white', 'pad': 4}, size = 'x-small')

            axs[1,3].axis('off')
            axs[1,3].set_title('|Uncertainty Pred (mm)|')
            plt.tight_layout()
        
    def plot_as_function_of_radius(self, DFP_max):
        #plot depth pre[]diction as a function of radius (depth = 5mm)

        p = plt.figure()
        radius = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        plt.axhline(y = 5, color = 'r', linestyle = 'dashed') 

        plt.scatter(radius, DFP_max)
        plt.xlim([0.1, 2.1])
        plt.ylim([0, 10])
        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("Radius (mm)")
        plt.title("Prediction Depth vs Radius")

        p.show()


    def determine_cutoff(self, mean_error, threshold = 2):
        print("mean error", mean_error)
        mean_error = np.array(mean_error)
        mean_error_above_threshold = mean_error[mean_error > threshold]
        for i in mean_error_above_threshold:
            if i == 1:
                return i
        
        return None

        
    def load(self):

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
                #self.new_model = tf.keras.models.load_model(loadFile, compile=False)
                #self.new_model.load_weights(loadFile)
                self.modelD = load_model(loadFile, compile=False) 
                #self.new_model.summary()# Load the weights into this model
                self.importData(isTesting=True,quickTest=True)
                break
                    
            else: # If the modelD attribute does not exist
                print('\nModel is not currently defined - select again.') 
                break
            

    def Analysis(self):
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix
        '''
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

        num_example_inclusion = [5, 15, 25, 35, 45, 55, 65, 75]
        '''

        self.indxIncl = np.nonzero(self.temp_DF_pre_conversion)

        #self.Predict()
        predict = self.modelD.predict([self.OP, self.FL], batch_size = 1)  

        # if dropout model is used
        '''
        num_dropout_ensembles = 16
        num_examples = self.OP.shape[0]
        image_dim = self.OP.shape[1]
        
        depth_prediction_ensembles = np.ndarray((num_examples, image_dim, image_dim, num_dropout_ensembles))
        concentration_prediction_ensembles = np.ndarray((num_examples, image_dim, image_dim, num_dropout_ensembles))

        for i in range(num_dropout_ensembles):
            #tmp = self.modelD.predict([self.OP, self.FL])  
            #depth_prediction_ensembles_no_dropout[:,:,:,i] = tmp[1].squeeze()
            #concentration_prediction_ensembles_no_dropout[:,:,:,i] = tmp[0].squeeze()
            tmp = self.modelD.predict([self.OP, self.FL])  
            depth_prediction_ensembles[:,:,:,i] = tmp[1].squeeze()
            concentration_prediction_ensembles[:,:,:,i] = tmp[0].squeeze()

        depth_prediction_mean = np.zeros((num_examples, image_dim, image_dim))
        concentration_prediction_mean = np.zeros((num_examples, image_dim, image_dim))

        #take the mean of each example 
        for i in range(num_examples):
            #depth_prediction_mean[i, :,:] = np.mean(depth_prediction_ensembles[i,:,:,:].squeeze(), axis = 2)
            #concentration_prediction_mean[i, :,:] = np.mean(concentration_prediction_ensembles[i,:,:,:].squeeze(), axis = 2)
            depth_prediction_mean[i, :,:] = np.mean(depth_prediction_ensembles[i,:,:,:].squeeze(), axis = 2)
            concentration_prediction_mean[i, :,:] = np.mean(concentration_prediction_ensembles[i,:,:,:].squeeze(), axis = 2)
        '''

        QF_P = predict[0] 
        DF_P = predict[1]
        #DF_P = depth_prediction_mean
        #QF_P = concentration_prediction_mean
        QF_P /= self.params['scaleQF']
        DF_P /= self.params['scaleDF']  

        #print(self.DF[0,:,:])
        #print(DF_P[0,:,:])
        #print(self.QF[0,:,:])
        #print(QF_P[0,:,:])

        self.save = 'n'
        if self.save == 'y':
            plot_save_path = 'Predictions//' + self.exportName + '//'
        print("DF_P", DF_P.shape)

        DF_P = np.reshape(DF_P, (DF_P.shape[0], DF_P.shape[1], DF_P.shape[2], 1))
        QF_P = np.reshape(QF_P, (QF_P.shape[0], QF_P.shape[1], QF_P.shape[2], 1))

        CL_P = self.classify(DF_P)
        print("CL_P", CL_P.shape)
        print("Cl shape", self.CL.shape)
        #self.CL = self.CL[0, 0:100]
        true = self.CL#np.where(self.DF > 5, 1, 0)
        true = np.concatenate(true)
        
        print("true", true.shape)
        print("CL_P", CL_P.shape)
        tn, fp, fn, tp = confusion_matrix(true, CL_P, labels=[0, 1]).ravel()
        accuracy = accuracy_score(true, CL_P)
        specificity = tn/(tn+fp)
        sensitivity = tp/(tp+fn)
        precision = tp/(tp+fp)
        npv = tn/(tn+fn)
        F1 = 2*(precision*sensitivity)/(precision + sensitivity)


        fpr, tpr, threshold = metrics.roc_curve(true, CL_P)
        roc_auc = metrics.auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        print("roc_auc", roc_auc)
        num_data = np.shape(true)[0]
        print(accuracy)
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
        #print('F1: ',"%.4f"%F1)

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
        
        
        DF_min = self.get_min(self.DF)
        DF_P_min = self.get_min(DF_P)
        if self.save in ['Y','y']:
            plot_save_path = 'Predictions//' + self.exportName + '//'

        for i in range(DF_P.shape[0]):
            ua_max.append(self.OP[i,:,:,0].max())
            us_max.append(self.OP[i,:,:,1].max())
            #DF_max.append(self.DF[i,:,:].max())
            #DFP_max.append(DF_P[i,:,:].max())
            QF_max.append(self.QF[i,:,:].max())
            QFP_max.append(QF_P[i,:,:].max())

            
        DF_max = DF_min
        DFP_max = DF_P_min

        DF_max = np.array(DF_max)
        DFP_max = np.array(DFP_max)
        QF_max = np.array(QF_max)
        QFP_max = np.array(QFP_max)

        #compute absolute mindepth error 
        min_depth_error = np.mean(np.abs(DFP_max - DF_max))
        min_depth_error_std = np.std(np.abs(DFP_max - DF_max))
        print("Average Minimum Depth Error (SD) : {min_depth_error} ({min_depth_error_std})".format(min_depth_error = min_depth_error, min_depth_error_std = min_depth_error_std))
        num_predict_zeros = self.count_predictions_of_zero(DFP_max)
        print("number of predictions of zero:", num_predict_zeros)
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
        #beg = 0
        #end = beg + 49

        min_depth_graph = plt.figure()
        #plt.scatter(DF_max[beg:end],DFP_max[beg:end],s=1)
        plt.scatter(DF_max,DFP_max,s=3, label = "Correct Classification", color = ['blue'])

        DF_max_classify = np.array(DF_max) < 5 
        DFP_max_classify = np.array(DFP_max) < 5
        
        failed_result = DF_max_classify !=DFP_max_classify
        
        
        
        plt.scatter(DF_max[failed_result],DFP_max[failed_result],label = "Incorrect Classification", s=3, color = ['red'])
        plt.legend(loc="upper left", prop={'size': 10})

        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.plot(plt.xlim([0, 10]), plt.ylim([0, 10]),color='k')
        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("Minimum Depth")
        plt.tight_layout()
        if self.save in ['Y','y']:
            plot_save_path_min_depth = plot_save_path + 'min_depth.png'
            plt.savefig(plot_save_path_min_depth, dpi=100, bbox_inches='tight')
        min_depth_graph.show()

        
        
        #plot depth error as a function of true depth 
        f = plt.figure()

        depth_error = abs(self.DF[self.indxIncl] - DF_P[self.indxIncl])
        plt.scatter(self.DF[self.indxIncl],depth_error,s=1)
        plt.xlim([0, 15])
        plt.ylim([0, 10])
        
        plt.ylabel("Absolute Depth Error (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("True Depth vs Absolute Depth Error")
        f.show()

        #plot mindepth error as a function of true depth 
        g = plt.figure()
        
      
        min_depth_error = np.abs(DFP_max - DF_max)
        plt.scatter(DF_max,min_depth_error,s=0.2)
        plt.xlim([0, 10])
        plt.ylim([0, 2])
        plt.ylabel("Absolute Depth Error (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("True Depth vs Absolute Minimum Depth Error")
        g.show()

        #plot mindepth bar graph with variance 
        mean_min_depth_error, std_min_depth_error = self.calculate_bar(min_depth_error, DF_max, 10)
        bar_val_graph = plt.figure()

        x_str = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9','9-10']
        #x_str = ['0-1', '1-2', '2-3']
 
        x_str = np.squeeze(x_str)
        plt.bar(x_str, mean_min_depth_error)

        plt.errorbar(x_str,mean_min_depth_error , yerr=std_min_depth_error, capsize=3, fmt="r--o", ecolor = "black")

        plt.xlabel("True Depth (mm)")
        plt.ylabel("Absolute Minimum Depth Error (mm)")
        bar_val_graph.show()
        

        #flattened_depth_error = np.reshape(depth_error, (depth_error.shape[0] * depth_error.shape[1] * depth_error.shape[2]))
        #flattened_depth_error = depth_error 
        #flattened_actual_depth = np.reshape(self.DF[self.indxIncl], (self.DF.shape[0] * self.DF.shape[1] * self.DF.shape[2]))

        #mean_depth_error, std_depth_error = self.calculate_bar(flattened_depth_error, flattened_actual_depth, 15)
        #print(mean_depth_error)
        #cutoff_val = self.determine_cutoff(mean_depth_error)
        #print("cutoff_val: ", cutoff_val)

        #bar_val_graph_all = plt.figure(figsize=(10, 6), dpi=80)

        #x_str = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9','9-10', '10-11', '11-12', '12-13', '13-14', '14-15']
        #x_str = np.squeeze(x_str)

        #plt.bar(x_str, mean_depth_error)
        #plt.errorbar(x_str,mean_depth_error , yerr=std_depth_error, capsize=3, fmt="r--o", ecolor = "black")
        #plt.xlabel("True Depth (mm)")
        #plt.ylabel("Absolute Depth Error (mm)")
        
        #bar_val_graph_all.show()
        
        #plot depth error as a function of absorption, scattering 
        #self.error_vs_absorption(DF_max, DFP_max, ua_max)
        #self.error_vs_scattering(DF_max, DFP_max, us_max)

        #self.scatter_vs_absorption_failed_result(DF_max, DFP_max, ua_max, us_max, failed_result)
        
        #self.concentration_vs_depth_error(DF_max, DFP_max, QF_max)
            
        #self.concentration_error_vs_depth_error(DF_max, DFP_max, QF_max, QFP_max)
        #self.concentration_vs_depth_error_classify(DF_max, DFP_max, QF_max, failed_result)
        #self.concentration_error_vs_depth_error_classify(DF_max, DFP_max, QF_max, QFP_max, failed_result)
        #self.concentration_error_vs_depth(self.QF, QF_P, self.DF)

        #self.concentration_error_vs_concentration(QF_max, QFP_max, failed_result)
        
        #self.max_concentration_graph(QF_max, QFP_max)

        #self.min_depth_error_as_function_of_depth(DF_max, DFP_max)

        #self.depth_error_as_function_of_depth(DF_max, DFP_max)
        #self.depth_error_as_function_of_thickness_and_depth(DF_max, DFP_max)
        self.concentration_error_as_function_of_thickness_and_depth(self.DF, self.QF, QF_P)
        self.depth_error_as_function_of_thickness_and_depth_all(self.DF, DF_P, self.QF)

        #self.depth_error_as_function_of_fluorophore_thickness_and_depth(DF_max, DFP_max)
        #self.concentration_error_as_funcdepth_error_as_function_of_fluorophore_thickness_and_depthtion_of_fluorophore_thickness_and_depth(DF_max, QF_max, QFP_max)
        # Plot true and predicted depth and concentration
        
        
        num_plot_display = 150
        
        num_example_inclusion = [x * 19 for x in range(40)]

        #self.save = 'y'
        plot_save_path = './Predictions/'

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
            #for i in num_example_inclusion:#range(num_plot_display):
            #failed_result_index = np.where(failed_result)
            for i in range(num_plot_display):#range(num_plot_display):
            #for i in failed_result_index[0]:
                fig, axs = plt.subplots(2,3)
                plt.set_cmap('jet')
                plt.colorbar(axs[0,0].imshow(self.DF[i,:,:],vmin=0,vmax=15), ax=axs[0, 0],fraction=0.046, pad=0.04)
                '''
                value = -1
                min_index = np.where(self.DF[i,:,:] == np.min(self.DF[i,:,:]))
                temp_value = self.DF[i, min_index[0], min_index[1]]
                self.DF[i, min_index[0], min_index[1]] = value 
                
                masked_array = np.ma.masked_where(self.DF[i,:,:] == value, self.DF[i,:,:])
                cmap = plt.cm.jet  # Can be any colormap that you want after the cm
                cmap.set_bad(color='white')
                
                #plt.colorbar(axs[0,0].imshow(self.DF[i,:,:],vmin=0,vmax=15), ax=axs[0, 0],fraction=0.046, pad=0.04)
                plt.colorbar(axs[0,0].imshow(masked_array,vmin=0,vmax=15), ax=axs[0, 0],fraction=0.046, pad=0.04, cmap = cmap)

                temp_value_str_1 = temp_value[0]

                axs[0,0].text(5, 5, 'min_depth = ' + str(temp_value_str_1), bbox={'facecolor': 'white', 'pad': 4}, size = 'x-small')
                self.DF[i, min_index[0], min_index[1]] = temp_value 
                '''

                axs[0,0].axis('off')
                axs[0,0].set_title('True Depth (mm)')
                '''
                min_index = np.where(DF_P[i,:,:] == np.min(DF_P[i,:,:]))
                
                temp_value = DF_P[i, min_index[0], min_index[1]]
                DF_P[i, min_index[0], min_index[1]] = value 
                
                masked_array = np.ma.masked_where(DF_P[i,:,:] == value, DF_P[i,:,:])
                cmap = plt.cm.jet  # Can be any colormap that you want after the cm
                cmap.set_bad(color='white')
                
                temp_value_str_2 = temp_value[0]
                plt.colorbar(axs[0,1].imshow(masked_array,vmin=0,vmax=15), ax=axs[0, 1],fraction=0.046, pad=0.04, cmap = cmap)
                axs[0,1].text(5, 5, 'min_depth = ' + str(temp_value_str_2), bbox={'facecolor': 'white', 'pad': 4}, size = 'x-small')
                DF_P[i, min_index[0], min_index[1]] = temp_value 
                '''
                
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
                #axs[0,2].text(5, 5, 'min_depth error = ' + str(temp_value_str_1 - temp_value_str_2), bbox={'facecolor': 'white', 'pad': 4}, size = 'x-small')

                axs[1,2].axis('off')
                axs[1,2].set_title('|Error (ug/mL)|')
                plt.tight_layout()
                if self.save in ['Y','y']:
                    plot_save_path_DF = plot_save_path + 'num_'+ str(i) + '_' + 'DF_QF.png'
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