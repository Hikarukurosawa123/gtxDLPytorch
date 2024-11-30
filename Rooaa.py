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

# Tensorflow / Keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing import image

from gtxDLUtils import Utils

import boto3 
import io


class DL(Utils):    
    # Initialization method runs whenever an instance of the class is initiated
    def __init__(self):
        print('Choose a bucket: \n 20240920-matthew\n 20241119-rooaa\n')
        self.bucket = input("Select a bucket from list:")
        #self.bucket = '20240920-matthew'
        print('Choose a Default Parameters Case: \n Default4Fx\n Default6Fx_HighRes_SmallFOV\n Default6Fx_LowRes_LargeFOV\n Default6Fx_64x64\n')
        self.case = input('Select a case from listed directories: ')
        self.params = self.Defaults(self.case) # Define some default parameters based on the selected case
        print('\nParameters:')
        print(self.params)
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
                self.params['learningRate'] = float(8e-6)
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
        for attr in ['RE','DF']:
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
            elif key in ['activation']:
                print('Current value of '+key+ ' is '+str(self.params[key]) ) # Output current value and options
                self.params[key] = input('Value: ') # Update current value based on choice
            elif key in ['epochs','batch','nFilters3D','nF','xX','yY','BNAct', 'nFilters2D']: # If user chooses one of the integer valued keys
                print('Current value of '+key+ ' is '+str(self.params[key])+' ')
                value = input('Value: ') 
                self.params[key] = int(value) # Input is string, so we need to convert to int
            elif key in ['learningRate','scaleRE','scaleDF']: # If user chooses one of the float valued keys
                print('Current value of '+key+ ' is '+str(self.params[key])+' ')
                value = input('Value: ')
                self.params[key] = float(value) # Input is string, so we need to conver to float
            elif key in ['kernelConv3D','strideConv3D','kernelResBlock3D','kernelConv2D','kernelResBlock2D', 'strideConv2D']: # If user chooses one of the tuple valued keys
                print('Current value of '+key+ ' is '+str(self.params[key])+' ')
                value = input('Value (without using brackets): ')
                self.params[key] = tuple(map(int, value.split(','))) # Convert csv string to tuple
            else: # Entry is not one of the listed keys
                print('Key does not exist; valid key values are printed in the dictionary above. Enter nothing to finish. ')
         
        return None    
    
    def importData(self, isTesting=True,quickTest=False):
        s3_client = boto3.client('s3')
        # Ask user to input if they are testing are not - indicates if you need to flip
        if isTesting == True:
            # Print out the contents of the bucket (i.e., options for importing)           
            for key in s3_client.list_objects_v2(Bucket=self.bucket)['Contents']: 
                data = key['Key']
                # Display only matlab files
                if data.find("TestingData")!=-1:
                    if data.find(".mat")!=-1 and data.find("Parameters") == -1:
                        print(data)
        else:
            for key in s3_client.list_objects_v2(Bucket=self.bucket)['Contents']: 
                data = key['Key']
                # Display only matlab files
                if data.find("TrainingData")!=-1:
                    if data.find(".mat")!=-1 and data.find("Parameters") == -1:
                        print(data)

        # Enter the name of the dataset you want to import
        # Note: To import new data, go to the desired bucket in AWS and upload data
        self.file_key = str(input('Enter the name of the dataset you want to import e.g. matALL.mat '))
        obj = s3_client.get_object(Bucket=self.bucket, Key=self.file_key)
        dataTemp = obj['Body'].read()
        self.dataset = h5py.File(io.BytesIO(dataTemp))
        
        self.RE = self.dataset['RE']
        self.DF = self.dataset['DF']
        
        # Check whether the user is using the single or multiple MAT format 
        # I.e., looking at individual MAT files (getDims=3) or looking at MAT files with more than one sample (getDim=4)
        getDims = len(np.shape(self.RE))
        if getDims == 4:
            numSets = int(np.shape(self.RE)[3])
            if quickTest == False:
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

                sizeFx = int(np.shape(self.RE)[0])
                while True:
                    check = input('There are ' + str(sizeFx) + ' spatial frequencies in the RE input, use them all? (Y/N) ')
                    if check in ['Y','y']:
                        self.params['nF'] = sizeFx
                        indxFx = np.arange(0,sizeFx,1)
                        break
                    elif check in ['N','n']:
                        useFx = int(input('How many spatial frequencies would you like to use? Number must be a factor of the total number of spatial frequencies: '))
                        self.params['nF'] = useFx
                        indxFx = np.arange(0,useFx,1)
                        self.RE = self.RE[0:sizeFx:int(sizeFx/useFx),:,:,:]
                        break
                    elif check == '':
                        break
                    else:
                        print('You did not select yes/no, try again or enter nothing to escape')
            else:
                numSets = int(np.shape(self.RE)[3])
                sizeFx = self.params['nF']
                indxFx = np.arange(0,sizeFx,1)
        
            start = time.perf_counter()

            self.RE = np.reshape(self.RE[indxFx,:,:,0:numSets],(len(indxFx),self.params['xX'],self.params['yY'],numSets,1))
            self.DF = self.DF[:,:,0:numSets]
            self.DF = np.reshape(self.DF,(self.params['xX'],self.params['yY'],numSets,1))

            # Reorder data
            self.RE = np.swapaxes(self.RE,0,3)
            self.DF = np.moveaxis(self.DF,2,0)
            
            stop = time.perf_counter()
            print('Load time = ' + str(stop-start))

            # Apply scale
            self.RE *= self.params['scaleRE']
            self.DF *= self.params['scaleDF']
            
            # Apply flipping and rotation for Case 2 and 3 (for irregularly-shaped test data):
            if isTesting==True:
                self.RE = np.rot90(self.RE,1,(1,2)) #rotate from axis 2 to 3
                self.RE = np.fliplr(self.RE) #flip left and right
                self.DF = np.rot90(self.DF,1,(1,2))  
                self.DF = np.fliplr(self.DF)

        elif getDims ==3:
            print('There is 1 input in this directory') # Do not ask user to enter number of inputs they want to use (only have one option)
            numSets = 1
            if quickTest == False:
                while True:  # Get number of desired spatial frequencies from the user
                    sizeFx = int(np.shape(self.RE)[0])
                    check = input('There are ' + str(sizeFx) + ' spatial frequencies in the RE input, use them all? (Y/N) ')
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
            
            self.RE = np.reshape(self.RE[indxFx,:,:],(1,len(indxFx),self.params['xX'],self.params['yY'],1))
            self.RE = np.swapaxes(self.RE,1,-2)
            self.DF = self.DF[:,:]
            self.DF = np.reshape(self.DF,(1,self.params['xX'],self.params['yY'],1))
            
            stop = time.perf_counter()
            print('Load time = ' + str(stop-start))
            
            # Apply scale
            self.RE *= self.params['scaleRE']
            self.DF *= self.params['scaleDF']
        
        self.RE[np.isnan(self.RE)] = 0
                
    
    def Model(self):
        """The deep learning architecture gets defined here"""
        ## Input ##
        inputData = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))
        inRE = inputData
        
        ## Reflectance Input Branch ##
        input_shape = inRE.shape
        inRE = Conv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inRE)
        outRE1 = inRE
        
        inRE = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inRE)
        outRE2 = inRE

        inRE = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inRE)
        outRE3 = inRE
        
        inRE = self.resblock_2D(self.params['nFilters2D']//2, self.params['kernelResBlock2D'], self.params['strideConv2D'], inRE)

        ## Reshape ##
        #zReshape = int(((self.params['nFilters3D']//2)*self.params['nF'])/self.params['strideConv3D'][2])
        #inRE = Reshape((self.params['xX'],self.params['yY'],zReshape))(inRE)

        ## Concatenate Branch ##
        concat = SeparableConv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], 
                                 strides=self.params['strideConv2D'], padding='same', activation=self.params['activation'], 
                                 data_format="channels_last")(inRE)

        concat = self.resblock_2D(self.params['nFilters2D'], self.params['kernelResBlock2D'], self.params['strideConv2D'], concat) 

        ## Depth Fluorescence Output Branch ##
        outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(concat)
        
        outDF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(outDF)
       
        outDF = BatchNormalization()(outDF)
        outDF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(outDF)

        
        ## Defining and compiling the model ##
        self.modelD = Model(inputs=[inputData], outputs=[outDF])
        self.modelD.compile(loss='mse',
                      optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                      metrics=['mae'])
        self.modelD.summary()
        ## Outputs for feature maps ##
        self.modelFM = Model(inputs=[inputData], outputs=[inputData, outRE1, outRE2, outRE3]) 
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
            fileName = self.exportPath+'_params.xml'
            xmlParams = dicttoxml(str(self.params))
            with open(fileName,'w') as paramsFile:
                paramsFile.write(parseString(xmlParams).toprettyxml("    "))
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
                        if file.endswith(".h5") or file.endswith(".keras"):
                            filename = "ModelParameters//"+folder+'//'+file
                            h5_files.append(filename)
                            print(filename)
            loadFile = input('Enter the general and specific directory (e.g. meshLRTests\\\LR2e-5) pertaining to the .h5 (weights) file you would like to load: ')
            self.modelD.load_weights(loadFile)
            self.history = self.modelD.fit([self.RE], [self.DF],validation_split=0.2,batch_size=self.params['batch'],
                                       epochs=100, verbose=1, shuffle=True, callbacks=callbackList)     
        else:
            self.history = self.modelD.fit([self.RE], [self.DF],validation_split=0.2,batch_size=self.params['batch'],
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
        evaluate = self.modelD.evaluate([self.RE],[self.DF])
        return None
    
    def Predict(self):
        """Make predictions of the labels using the testing data (note that this is different from the Evaluate method, since the evaluate method tells us the metrics (without making the predictions) while this method only makes predictions"""
        predict = self.modelD.predict([self.RE])  
        DF_P = predict
        DF_P /= self.params['scaleDF']
        print('\n Making predictions on the test data... \n')  
        
        while True:
            save = input('\n Save these predictions? (Y/N) ')
            if save in ['Y','y']:
                self.exportName = input('The model, best weights, and some assorted parameters and other information will be saved during fitting; enter a name for the export directory or leave this field blank if you do not want to export (note that if you do not export, you will not be able to plot your results in the future once the current history variable is lost from memory, however you can still plot the results now): ')
                if len(self.exportName) > 0:
                    # Export MATLAB file to case
                    self.exportPath = 'Predictions//'+self.exportName+'.mat'
                    a = np.array(DF_P[:,:,:])
                    scipy.io.savemat(self.exportPath,{'DF_pred':a})
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
                    if file.endswith(".h5") or file.endswith(".keras"):
                        filename = "ModelParameters//"+folder+'//'+file
                        h5_files.append(filename)
                        print(filename)
                
        while True: # Loop until user makes acceptable choice for the input directory
            loadFile = input('Enter the general and specific directory (e.g. meshLRTests\\\LR2e-5) pertaining to the .h5 (weights) file you would like to load: ')
            if loadFile == '': # User enters nothing; break
                break
            elif loadFile in h5_files:
                if hasattr(self,'modelD'): # If the a model exists, this instance of the class will have the attribute modelD
                    checkParams = input('\nA model is currently defined but in order for the weights to load successfully this model must have been defined using the same parameters as when the weights were initially defined (i.e. nF, kernel, etc.). Are the parameters of your model and weights the same? (Y/N) ')
                    if checkParams in ['Y','y']: # If the user decides that the model is already defined using the correct parameters 
                        self.modelD = load_model(loadFile) # Load the weights into this model
                        # Whether the model was already defined properly, or did not exist, or needed to be redefined, we now need to load in data for testing and then make predictions based on the loaded weights 
                        self.importData()
                        self.Predict() # Make predictions
                        self.Evaluate()
                        break
                    elif checkParams in ['N','n']: # If the user decides that the model is not defined using the correct parameters
                        print('\nI suggest referring to the .xml file associated with your saved weights to determine which parameters you need to change. Calling the Params() method now, and modelling immediately following the changes: \n') # Refer to .xml params
                        self.Params() # Allow the user to define new parameters
                        #self.Model() # Model using these new, presumable correct, parameters
                        print('Now loading weights: ')
                        self.modelD = load_model(loadFile) # Load the weights into this model
                        # Whether the model was already defined properly, or did not exist, or needed to be redefined, we now need to load in data for testing and then make predictions based on the loaded weights 
#                         self.ReadData() # Read in data for testing the loaded weights
                        self.importData()
                        self.Predict() # Make predictions
                        self.Evaluate()
                        break
                else: # If the modelD attribute does not exist
                    print('\nA model is not currently defined. I suggest referring to the .xml file associated with your saved weights to determine which parameters you need to change. Calling the Params() method now, and modelling immediately following the changes: ') # Refer to .xml params
                    self.Params() # Allow the user to define new parameters
                    self.Model() # Model using these new, presumable correct, parameters
                    print('Now loading weights: ')
                    self.modelD.load_weights(loadFile) # Load the weights into this model 
                    # Whether the model was already defined properly, or did not exist, or needed to be redefined, we now need to load in data for testing and then make predictions based on the loaded weights 
                    self.importData()
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
                    if file.endswith(".h5") or file.endswith(".keras"):
                        filename = "ModelParameters//"+folder+'//'+file
                        h5_files.append(filename)
                        print(filename)      
        while True: # Loop until user makes acceptable choice for the input directory
            loadFile = input('Enter the general and specific directory pertaining to the .h5 (weights) file you would like to load: ')
            if loadFile == '': # User enters nothing; break
                break
            elif loadFile in h5_files:
                self.new_model = tf.keras.models.load_model(loadFile)
                self.new_model.load_weights(loadFile)
                self.modelD = load_model(loadFile) 
                self.new_model.summary()# Load the weights into this model
                while True:
                    self.importData(isTesting=True,quickTest=True)
                    self.Predict()
                    self.Results()
                    self.Evaluate()
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
        img = input('Choose the input/output map you want to display (RE, DF): ')
        if img in ['RE','re']: # Print reflectance maps
            fx = self.params['nF'] # Number of spatial frequencies
            for i in range(fx): 
                plt.imshow(self.RE[10,:,:,i,:])
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.show()
            return None
        elif img in ['DF','df']: # Print depth maps
            for i in range(1):
                plt.imshow(self.DF[0,:,:,i])
                plt.show()
            return None
        else:
            print('\n Did not select a valid option - select again.')
        
    def Results(self):
        predict = self.modelD.predict([self.RE])  
        DF_P = predict
        DF_P /= self.params['scaleDF']
        i=0  
        plt.imshow(DF_P[i,:,:],vmin=0,vmax=10)
        plt.set_cmap('jet')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.show()
        return None
    
    def Analysis(self):
        predict = self.modelD.predict([self.RE])  
        DF_P = predict
        DF_P /= self.params['scaleDF']  
        indxIncl = np.nonzero(self.DF)
        ## Error Stats
        # Average error
        DF_error = DF_P - self.DF
        DF_erroravg = np.mean(abs(DF_error[indxIncl]))
        DF_errorstd = np.std(abs(DF_error[indxIncl]))
        print('Average Depth Error (SD): {}({}) mm'.format(float('%.5g' % DF_erroravg),float('%.5g' % DF_errorstd)))
        # Overall  mean squared error
        DF_mse = np.sum((DF_P - self.DF) ** 2)
        DF_mse /= float(DF_P.shape[0] * DF_P.shape[1] * DF_P.shape[2])
        print('Depth Mean Squared Error: {} mm'.format(float('%.5g' % DF_mse)))
        # Max and Min values per sample
        DF_max = []
        for i in range(DF_P.shape[0]):
            DF_max.append(self.DF[i,:,:].max())
        # SSIM per sample
        DF_ssim =[]
        for i in range(DF_P.shape[0]):
            df_p = np.reshape(DF_P[i,:,:],(DF_P.shape[1],DF_P.shape[2])) # reshape predicted
            df_t = np.reshape(self.DF[i,:,:],(self.DF.shape[1],self.DF.shape[2])) # reshape true
            df_ssim = ssim(df_p,df_t,data_range=max(df_p.max(),df_t.max())-min(df_p.min(),df_t.min()))
            DF_ssim.append(df_ssim)
        print('Overall Depth SSIM: {}'.format(float('%.5g' % np.mean(DF_ssim))))
        ## Plot Correlations, Histogram, SSIM
        fig, axs = plt.subplots(1, 3)
        axs[0].scatter(self.DF[indxIncl],DF_P[indxIncl],s=1)
        axs[0].set_xlim([0, 5])
        axs[0].set_ylim([0, 5])
        y_lim1 = axs[0].set_ylim()
        x_lim1 = axs[0].set_xlim()
        axs[0].plot(x_lim1, y_lim1,color='k')
        axs[0].set_ylabel("Predicted Depth (mm)")
        axs[0].set_xlabel("True Depth (mm)")
        axs[0].set_title("Depth Correlation") 
        axs[1].hist(DF_error[indxIncl],bins=100)
        axs[1].set_xlabel("Depth Error (mm)")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("Histogram")
        #plt.tight_layout()
        axs[2].scatter(DF_max,DF_ssim,s=1)
        axs[2].set_xlabel('Max Depth')
        axs[2].set_ylim([0,1])
        axs[2].set_title("SSIM")
        plt.tight_layout()
        plt.show()
        
        ## Maximum Depth Analysis
        DF_max = []
        DFP_max = []
        for i in range(DF_P.shape[0]):
            DF_max.append(self.DF[i,:,:].max())
            DFP_max.append(DF_P[i,:,:].max())
            
        ## Plot Max True Depth vs. Max Predicted Depth
        fig
        plt.scatter(DF_max,DFP_max,s=1)
        plt.xlim([0, 3])
        plt.ylim([0, 3])
        plt.plot(plt.xlim([0, 3]), plt.ylim([0, 3]),color='k')
        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("Max Depth")
        plt.tight_layout()
        plt.show()
        fx = self.params['nF']

        # Plot true and predicted depth and concentration
        maxDepth = input('Enter Maximum Depth: ')
        if self.DF.shape[0] < 50:
            for i in range(DF.shape[0]):
                fig, axs = plt.subplots(1,3)
                plt.set_cmap('jet')
                plt.colorbar(axs[0].imshow(self.DF[i,:,:],vmin=0,vmax=maxDepth), ax=axs[0],fraction=0.046, pad=0.04)
                axs[0].axis('off')
                axs[0].set_title('True Depth (mm)')
                plt.colorbar(axs[1].imshow(DF_P[i,:,:],vmin=0,vmax=maxDepth), ax=axs[1],fraction=0.046, pad=0.04)
                axs[1].axis('off')
                axs[1].set_title('Predicted Depth (mm)')
                plt.colorbar(axs[2].imshow(abs(DF_error[i,:,:]),vmin=0,vmax=maxDepth), ax=axs[2],fraction=0.046, pad=0.04)
                axs[2].axis('off')
                axs[2].set_title('|Error (mm)|')
                plt.tight_layout()   
                plt.show()
        else:
            for i in range(10):
                fig, axs = plt.subplots(1,3)
                plt.set_cmap('jet')
                plt.colorbar(axs[0].imshow(self.DF[i,:,:],vmin=0,vmax=maxDepth), ax=axs[0],fraction=0.046, pad=0.04)
                axs[0].axis('off')
                axs[0].set_title('True Depth (mm)')
                plt.colorbar(axs[1].imshow(DF_P[i,:,:],vmin=0,vmax=maxDepth), ax=axs[1],fraction=0.046, pad=0.04)
                axs[1].axis('off')
                axs[1].set_title('Predicted Depth (mm)')
                plt.colorbar(axs[2].imshow(abs(DF_error[i,:,:]),vmin=0,vmax=maxDepth), ax=axs[2],fraction=0.046, pad=0.04)
                axs[2].axis('off')
                axs[2].set_title('|Error (mm)|')
                plt.tight_layout()
                plt.show()
                
                fig, axs = plt.subplots(1, fx)
                plt.set_cmap('gray')
                for j in range(fx):
                    plt.colorbar(axs[j].imshow(self.RE[i,:,:, j]), ax=axs[j],fraction=0.046, pad=0.04)
                    axs[j].axis('off')
                    axs[j].set_title('RE')    
                plt.tight_layout()
                plt.show()
                
               
           
    def PrintFeatureMap(self):
        """Generate Feature Maps"""
        feature_maps = self.modelFM.predict([self.RE]) # Output for each layer
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