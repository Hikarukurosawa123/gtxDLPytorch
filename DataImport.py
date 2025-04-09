from xml.dom.minidom import parseString
import io
import time
import os, time, sys
import boto3 
import numpy as np, h5py
import pandas as pd
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from dicttoxml import dicttoxml
import logging
import boto3
from botocore.exceptions import ClientError
import os
import mat73


class Operations():
    
    def __init__(self):
        pass
      

        
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
        
        #time delay to let data names get printed
        time.sleep(1.5)


        self.file_key = str(input('Enter the name of the dataset you want to import e.g. matALL.mat '))
        
        self.params["training_file_name"] = self.file_key

        #import data either in AWS cloud or in local desktop 
        
        obj = s3_client.get_object(Bucket=self.bucket, Key=self.file_key)

        dataTemp = obj['Body'].read()
    
        self.dataset = mat73.loadmat((io.BytesIO(dataTemp)))
        

        
        apply_normalization = 0

        self.FL = self.dataset['F']
       
        self.DF = self.dataset['DF']
        self.OP = self.dataset['OP']
        self.QF = self.dataset['QF']
        self.RE = self.dataset['RE']
        
        print("DF shape: ", np.shape(self.DF))
        print("OP shape: ", np.shape(self.OP))
        print("QF shape: ", np.shape(self.QF))
        print("RE shape: ", np.shape(self.RE))

        self.temp_DF_pre_conversion = self.DF
        
        self.background_val = str(input('Enter the value of the background'))
        if self.background_val == '':
            self.background_val = 0 #default to zero
        
        self.convert_background_val() #convert values of DF background
        print("after applying conversion", np.shape(self.DF))

        

    
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
        for folder in os.listdir("ModelParameters"):
            if not folder.endswith((".h5",".log",".xml", ".keras")):
                for file in os.listdir("ModelParameters/"+folder):
                    if file.endswith(".keras") or file.endswith(".h5"):
                        filename = "ModelParameters/"+folder+'/'+file
                        keras_files.append(filename)
                        print(filename)        
        
        
        while True: # Loop until user makes acceptable choice for the input directory
            loadFile = input('Enter the general and specific directory pertaining to the .keras (weights) file you would like to load: ')
            break 

        
        #send to AWS 
        print(type(loadFile))
        self.exportPath = loadFile.replace('.keras', '')

        params_log_file_name = self.exportPath+'_params.log'
        params_xml_file_name = self.exportPath+'_params.xlsx'
        keras_file_name = self.exportPath+'.keras'
        params_case_file_name = self.exportPath+'.log'

        self.upload_file(params_log_file_name)
        self.upload_file(params_xml_file_name)
        self.upload_file(keras_file_name)
        self.upload_file(params_case_file_name)

        print("file uploaded to AWS")

    
    def Fit(self,isTransfer):
        # Where to export information about the fit
        self.exportName = input('Enter a name for exporting the model: ')
        lrDecay = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, verbose=1, min_delta=5e-5)
        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=20, verbose=1, mode='auto')
        callbackList = [earlyStopping,lrDecay]

        run_torch = 1

        if run_torch:
            #run training using torch 

            #define the model 
            


            return 
    
        if len(self.exportName) > 0:
            os.makedirs("ModelParameters/"+self.exportName)
            self.exportPath = 'ModelParameters/'+self.exportName+'/'+self.case
            #save dictionary as excel file 
            #fileName = self.exportPath+'_params.xml'
            self.paramsexcelfileName = self.exportPath + '_params.xlsx'
            #xmlParams = dicttoxml(str(self.params))
            
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
                    for file in os.listdir("ModelParameters/"+folder):
                        if file.endswith((".h5", ".keras")):
                            filename = "ModelParameters/"+folder+'/'+file
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