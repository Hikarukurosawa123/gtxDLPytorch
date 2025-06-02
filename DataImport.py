from xml.dom.minidom import parseString
import io
import time
import os, time, sys
import boto3 
import numpy as np, h5py
import pandas as pd
from dicttoxml import dicttoxml
import logging
import boto3
from botocore.exceptions import ClientError
import os
import mat73
import torch 
from Models_pytorch.siamese_pytorch import TinyModel
#PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, images1,labels1, labels2):

        super().__init__()

        self.images1 = torch.tensor(images1, dtype=torch.float32)
        #self.images2 = torch.tensor(images2, dtype=torch.float32)

        self.labels1 = torch.tensor(labels1, dtype=torch.float32)
        self.labels2 = torch.tensor(labels2, dtype=torch.float32)

    def __len__(self):
        return self.images1.shape[0]

    def __getitem__(self, idx):
        image1 = self.images1[idx]       # shape: [2, 101, 101]
        #image2 = self.images2[idx]       # shape: [6, 101, 101]

        label1 = self.labels1[idx]       # shape: [101, 101] # QF
        label2 = self.labels2[idx]       # shape: [101, 101] # QF

        return image1, label1, label2


class Operations():
    
    def __init__(self):
        self.bucket = '20240909-hikaru'
        
        isCase = 'Default'#input('Choose a case:\n Default\n Default4fx')
        self.case = isCase
        self.params = self.Defaults(self.case) # Define some default parameters based on the selected case
        self.isTesting = False
        self.AWS = False        
      

        
    def get_min(self, DF):

        DF_zeros = np.array(DF)

        DF_min_per_case = np.nanmin(DF_zeros, axis = (1,2))
        
        return DF_min_per_case
    
        
    def classify(self, DF_P):      
        
        DF_min_per_case = np.nanmin(DF_P[:,:,:,0], axis = (1,2))
        CL = DF_min_per_case < 5 # 5mm or less would be one 
        
        return CL
    
        
    def add_classifier(self):

        DF_zeros = np.array(self.temp_DF_pre_conversion)
        
                           
        DF_min_per_case = np.min(DF_zeros, axis = (0,1))
        self.CL = DF_min_per_case < 5 # 5mm or less would be one 
        self.CL = np.reshape(self.CL, (1, self.CL.shape[0]))

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

            #self.Plot(isTraining=True)
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
    def importData(self,isTesting=True,quickTest=False):

        #determine where to import testing/training data from 
        s3_client = boto3.client('s3')

        mode = "TestingData" if isTesting else "TrainingData"

        try:
            response = s3_client.list_objects_v2(Bucket=self.bucket, Prefix=mode)
            if 'Contents' not in response:
                print("No files found in the bucket.")
                return

            print(f"Listing .mat files for: {mode}")
            for item in response['Contents']:
                key = item['Key']

                if key.endswith(".mat"):
                    print(key)

        except Exception as e:
            print(f"Error accessing S3 bucket: {e}")

        # Enter the name of the dataset you want to import
        # Note: To import new data, go to the desired bucket in AWS and upload data
        
        #time delay to let data names get printed
        time.sleep(1.5)


        self.file_key = str(input('Enter the name of the dataset you want to import e.g. matALL.mat '))
        
        self.params["training_file_name"] = self.file_key

        #import data either in AWS cloud or in local desktop 
        
        obj = s3_client.get_object(Bucket=self.bucket, Key=self.file_key)

        dataTemp = obj['Body'].read()
    
        self.dataset = mat73.loadmat((io.BytesIO(dataTemp)))
  
        apply_normalization = 0

        apply_mean_normalization = 1

        self.FL = self.dataset['F']
       
        self.DF = self.dataset['DF']
        self.OP = self.dataset['OP']
        self.QF = self.dataset['QF']
        self.RE = self.dataset['RE']


        #expand the first axis if the dimension is of size 2
        if  len(np.shape(self.FL)) == 3:
            self.DF = np.expand_dims(self.dataset['DF'], axis=0)
            self.OP = np.expand_dims(self.dataset['OP'], axis=0)
            self.QF = np.expand_dims(self.dataset['QF'], axis=0)
            self.RE = np.expand_dims(self.dataset['RE'], axis=0)
            self.FL = np.expand_dims(self.dataset['RE'], axis=0)

        #pad with ones temporarily 

        # self.DF = np.pad(self.DF, ((0,0), (0, 1), (0, 1)), mode='constant')
        # self.OP = np.pad(self.OP, ((0,0),(0, 1), (0, 1), (0,0)), mode='constant')
        # self.QF = np.pad(self.QF, ((0,0),(0, 1), (0, 1)), mode='constant')
        # self.RE = np.pad(self.RE, ((0,0),(0, 1), (0, 1), (0,0)), mode='constant')
        # self.FL = np.pad(self.FL, ((0,0),(0, 1), (0, 1), (0,0)), mode='constant')


    
        self.temp_DF_pre_conversion = self.DF
        
        self.background_val = str(input('Enter the value of the background'))
        if self.background_val == '':
            self.background_val = 0 #default to zero
        
        self.convert_background_val() #convert values of DF background

        # Check whether the user is using the single or multiple MAT format 
        # I.e., looking at individual MAT files (getDims=3) or looking at MAT files with more than one sample (getDim=4)
        getDims = len(np.shape(self.FL))

        if getDims == 4:
            numSets = int(np.shape(self.FL)[0])
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

                sizeFx = int(np.shape(self.FL)[-1])
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
            
            stop = time.perf_counter()
            print('Load time = ' + str(stop-start))
            
            print(np.shape(self.OP))

            # Apply scale
            self.OP[:,:,:,0] *= self.params['scaleOP0']
            self.OP[:,:,:,1] *= self.params['scaleOP1']
            self.DF *= self.params['scaleDF']
            self.QF *= self.params['scaleQF']
            #self.RE[:,:,:,0] *= self.params['scaleRE']

            if 'scaleFL0' in self.params:
                for i in range(self.params['nF']):
                    scaleN = 'scaleFL'+ str(i)
                    self.FL[:,:,:,i] *= self.params[scaleN]
                    print(np.mean(self.FL[:,:,:,i]))
            else:
                self.FL *= self.params['scaleFL']
                #self.FL[:,:,:,:] /= FL_max
                
            if apply_normalization:
                #apply normalization on FL 

                FL_mean = np.mean(self.FL, axis = (1,2,3), keepdims = True)
                FL_std = np.std(self.FL, axis = (1,2,3), keepdims = True)

                self.FL = np.array(self.FL)

                self.FL = (self.FL - FL_mean) / FL_std

                #apply normaliztion on OP 

                OP_mean = np.mean(self.OP, axis = (1,2,3), keepdims = True)
                OP_std = np.std(self.OP, axis = (1,2,3), keepdims = True)

                self.OP = np.array(self.OP)

                self.OP = (self.OP - OP_mean) / OP_std

            
            if apply_mean_normalization:
                # Apply min-max normalization on FL
                FL_min = np.min(self.FL, axis=(1, 2, 3), keepdims=True)
                FL_max = np.max(self.FL, axis=(1, 2, 3), keepdims=True)
                self.FL = np.array(self.FL)
                self.FL = (self.FL - FL_min) / (FL_max - FL_min + 1e-8)  # add epsilon to avoid division by zero

                # Apply min-max normalization on OP
                OP_min = np.min(self.OP, axis=(1, 2, 3), keepdims=True)
                OP_max = np.max(self.OP, axis=(1, 2, 3), keepdims=True)
                self.OP = np.array(self.OP)
                self.OP = (self.OP - OP_min) / (OP_max - OP_min + 1e-8)



            
            #obtain min and max 
            self.DF_min = self.get_min(self.DF) 
            self.QF_max = self.get_max(self.QF) 




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
 
            
            if 'scaleFL0' in self.params:
                for i in range(self.params['nF']):
                    scaleN = 'scaleFL'+ str(i)
                    self.FL[:,:,:,i] *= self.params[scaleN]
                    print(np.mean(self.FL[:,:,:,i]))
            else:
                self.FL *= self.params['scaleFL']            
                #self.FL[:,:,:,:] /= FL_max

    def upload_file(self,file_name, object_name=None):
        """Upload a file to an S3 bucket

        :param file_name: File to upload
        :param bucket: Bucket to upload to
        :param object_name: S3 object name. If not specified then file_name is used
        :return: True if file was uploaded, else False
        """
        
        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = file_name

        # Upload the file
        s3_client = boto3.client('s3')
        print(file_name)
        try:
            response = s3_client.upload_file(file_name, self.bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True
    
    def upload_to_S3(self):

        #upload model parameters to s3 to enable access from local computer 
    
        #display the model parameters available for export 
        keras_files = []
        for folder in os.listdir("ModelParameters_PT"):
            if not folder.endswith((".h5",".log",".xml", ".pt")):
                for file in os.listdir("ModelParameters_PT/"+folder):
                    if file.endswith((".pt")):
                        filename = "ModelParameters_PT/"+folder+'/'+file
                        keras_files.append(filename)
                        print(filename)        
        
        
        while True: # Loop until user makes acceptable choice for the input directory
            loadFile = input('Enter the general and specific directory pertaining to the .keras (weights) file you would like to load: ')
            break 

        self.exportPath = filename

    
        pytorch_file_name = self.exportPath 
        self.upload_file(pytorch_file_name)


        print("file uploaded to AWS")

    

    
    def Fit(self,isTransfer):
        # Where to export information about the fit
        self.exportName = input('Enter a name for exporting the model: ')
        #lrDecay = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, verbose=1, min_delta=5e-5)
        #earlyStopping = EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=20, verbose=1, mode='auto')
        #callbackList = [earlyStopping,lrDecay]

        
    
        #structure dataset to be in the form (image, label)

        #image in the shape (N, H, W, C)

        #get 95% of the loaded data for training 
        training_image_OP = self.OP[0:int(self.DF.shape[0] * 0.95), :,:,:]
        validation_image_OP = self.OP[int(self.DF.shape[0] * 0.95):, :,:,:]

        training_image_FL = self.FL[0:int(self.DF.shape[0] * 0.95), :,:,:]
        validation_image_FL = self.FL[int(self.DF.shape[0] * 0.95):, :,:,:]

        training_label_QF = self.QF[0:int(self.DF.shape[0] * 0.95), :,:]
        validation_label_QF = self.QF[int(self.DF.shape[0] * 0.95):, :,:]
        
        training_label_DF = self.DF[0:int(self.DF.shape[0] * 0.95), :,:]
        validation_label_DF = self.DF[int(self.DF.shape[0] * 0.95):, :,:]

    

        

        #reshape image to have the shape (N, H, W, C) --> (N, C, H, W)
        training_image_OP = np.transpose(training_image_OP, (0, 3, 1,2))
        validation_image_OP = np.transpose(validation_image_OP, (0, 3, 1,2))

        training_image_FL = np.transpose(training_image_FL, (0, 3, 1,2))
        validation_image_FL = np.transpose(validation_image_FL, (0, 3, 1,2))

        training_set = MyDataset(training_image_OP, training_image_FL,training_label_QF, training_label_DF)
        validation_set = MyDataset(validation_image_OP, validation_image_FL, validation_label_QF, validation_label_DF)

        #convert shape of the input to accomodate the expected shape

        # Create data loaders for our datasets; shuffle for training, not for validation
        training_loader = DataLoader(training_set, batch_size=32, shuffle=True)
        validation_loader = DataLoader(validation_set, batch_size=32, shuffle=True)

        #run training using torch 
        loss_fn = torch.nn.L1Loss()

        #specify the save path 
        os.makedirs("ModelParameters_PT/"+self.exportName)
        self.exportPath = 'ModelParameters_PT/'+self.exportName+'/'+self.case + '.pt'

        #define the model and train 
        self.train_and_validate(validation_loader, loss_fn, training_loader)

        return 
        