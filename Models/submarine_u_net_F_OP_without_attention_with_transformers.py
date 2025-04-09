from __future__ import print_function
import locale

import pandas as pd
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
#from sklearn import metrics
from dateutil import parser
from csv import writer, reader
import xml.etree.ElementTree as ET
#import matplotlib.pyplot as plt
#import matplotlib
import numpy as np, h5py
import os, time, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import scipy.io
#from skimage.metrics import structural_similarity as ssim

# Tensorflow / Keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout, MaxPool2D, UpSampling2D, ZeroPadding2D, Activation
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing import image
import trans_u_net as encoder_layers 
from gtxDLClassAWSUtils import Utils

import boto3 
import io
#import openpyxl


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
        self.DF = self.dataset['DF']
        self.OP = self.dataset['OP']
        self.QF = self.dataset['QF']
        self.RE = self.dataset['RE']

        self.temp_DF_pre_conversion = self.DF

        self.add_classifier() #add classifier before changing background value 

        
        #self.background_val = 0 #default to zero

        #self.convert_background_val() #convert values of DF background 
        
        #FL_max = 0.007258041#np.max(self.FL)
        #DF_max = 23.361952 #np.max(self.DF)
        #OP_0_max = 0.042453066 #np.max(self.OP[0,:,:,:])
        #OP_1_max = 2.0797756 #np.max(self.OP[1,:,:,:])
        #QF_max = 9.998467 #np.max(self.QF)
     


        
        
        self.background_val = str(input('Enter the value of the background'))
        if self.background_val == '':
            self.background_val = 0 #default to zero
        
        self.convert_background_val() #convert values of DF background
        
       
    
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
            
            #self.FL = np.reshape(self.FL[indxFx,:,:],(len(indxFx),self.params['xX'],self.params['yY'],1,1))
            #self.RE = np.reshape(self.RE[indxFx,:,:],(len(indxFx),self.params['xX'],self.params['yY'],1,1))

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

    def attention_gate(self, g, s, num_filters):
        Wg = Conv2D(num_filters, 1, strides = (2,2), padding="valid")(g)
        
        Ws = Conv2D(num_filters, 1, padding="same")(s)
       
        out = Activation("relu")(Wg + Ws)
        out = Conv2D(1, 1, padding="same")(out)
        out = Activation("sigmoid")(out)
        out = UpSampling2D()(out)
       
        return out * g

    def Model(self):
        """The deep learning architecture gets defined here"""
      

        ## Input Optical Properties ##
        inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))
        ## Input Multi-Dimensional Fluorescence ##
        inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))

        ## NOTE: Batch normalization can cause instability in the validation loss

        ## Optical Properties Branch ##
        inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)

        inOP = Dropout(0.5)(inOP)

        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        
        inOP = Dropout(0.5)(inOP)
        
        ## Fluorescence Input Branch ##
        input_shape = inFL_beg.shape
        inFL = Conv2D(filters=self.params['nFilters3D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)
        
        inFL = Dropout(0.5)(inFL)

        inFL = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        inFL = Dropout(0.5)(inFL)

        ## Concatenate Branch ##
        concat = concatenate([inOP,inFL],axis=-1)

        Max_Pool_1 = MaxPool2D()(concat)

        Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Max_Pool_1)
        Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_1)
        
        Max_Pool_2 = MaxPool2D()(Conv_1)

        Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Max_Pool_2)
        Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_2)

        Max_Pool_3 = MaxPool2D()(Conv_2)


        #Conv_3 = Conv2D(filters=1024, kernel_size=(self.params['kernelConv2D']), strides=self.params['strideConv2D'], padding='same', 
        #            activation=self.params['activation'], data_format="channels_last")(Max_Pool_3)
        #Conv_3 = Conv2D(filters=1024, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
        #            activation=self.params['activation'], data_format="channels_last")(Conv_3)


        tfk = tf.keras
        tfkl = tfk.layers
        tfm = tf.math
        hidden_size = Conv_3.shape[-1]
        n_layers = 6
        n_heads = 16
        mlp_dim = 512
        dropout = 0.1
        patch_size = 1

        #linear embeddings 

        y = tfkl.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
        trainable=True
        )(Conv_3)

        #flattening out 
        y = tfkl.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
        
        #create the layers 
        y = encoder_layers.AddPositionEmbs( trainable=True)(y)

        y = tfkl.Dropout(0.1)(y)

        # Transformer/Encoder
        for n in range(n_layers):
            y, _ = encoder_layers.TransformerBlock(
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                trainable=True
            )(y)
        y = tfkl.LayerNormalization(
            epsilon=1e-6
        )(y)

        n_patch_sqrt = int(np.sqrt(y.shape[1]))

        print("y shape: ", y.shape)
        print("n_patch_sqrt: ", n_patch_sqrt)

        #revert the shape back to square dimensions 
        y = tfkl.Reshape(
            target_shape=[n_patch_sqrt, n_patch_sqrt, hidden_size])(y)
        
        Up_conv_1 = UpSampling2D()(y)

        s = Conv_2[:,0:Conv_2.shape[1] - 1, 0:Conv_2.shape[2] - 1, :]


        concat_1 = concatenate([s,Up_conv_1],axis=-1)

        Conv_4 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(concat_1)
 
        Conv_4 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            activation=self.params['activation'], data_format="channels_last")(Conv_4)
        
        Up_conv_2 = UpSampling2D()(Conv_4)

        Up_conv_2 = Conv2D(filters=256, kernel_size = (2,2), strides=(1,1), padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Up_conv_2)
    
        Up_conv_2 = ZeroPadding2D()(Up_conv_2)

        #(None, 50, 50, 256)
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
        
        outQF = Conv2D(filters=1, kernel_size=(1,1), strides=self.params['strideConv2D'], padding='same', 
                      data_format="channels_last")(outQF)

        ## Depth Fluorescence Output Branch ##
        #first DF layer 
        outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_6)
  
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
    