import matplotlib.pyplot as plt
import numpy as np, h5py
import os 
import pandas as pd 
import scipy.io
import dicttoxml 
from xml.dom.minidom import parseString
import matplotlib.pyplot as plt 
import matplotlib
#from sklearn import metrics
import io
import os, time, sys
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import StandardScaler


from DataImport import Operations
import boto3
import mat73
from os.path import isfile, join
import time
import tempfile
from torch.utils.data import Dataset, DataLoader

import torch


    
class Helper():
    def __init__(self):
        super().__init__()

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
                            for file in os.listdir("ModelParameters/"+folder):
                                if file.endswith(".log"):
                                    if  'params' not in file:
                                        filename = "ModelParameters/"+folder+'/'+file
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

    

    
    def get_min(self, DF):

        DF_zeros = np.array(DF)

        DF_min_per_case = np.nanmin(DF_zeros, axis = (1,2))

        return DF_min_per_case
    
    def get_max(self, QF):

        QF_zeros = np.array(QF)

        QF_min_per_case = np.max(QF_zeros, axis = (1,2))

        return QF_min_per_case
    
    def convert_background_val(self):
        self.DF = np.array(self.DF)
   
        for x in range(self.DF.shape[0]):
            ind_zeros = self.DF[x, :,:] == 0
            self.DF[x,ind_zeros] = self.background_val
   
    def import_data_for_testing(self):
        
        self.importData(isTesting=True,quickTest=True)
    
        
    def load(self):

        s3_client = boto3.client('s3')

        h5_files = []      
        
        time.sleep(1.5)

                
        bucket = self.bucket
        folder = "ModelParameters"
        s3 = boto3.resource("s3")
        s3_bucket = s3.Bucket(bucket)
        files_in_s3 = [f.key.split(folder + "/")[1] for f in s3_bucket.objects.filter(Prefix=folder).all()]

        #filter files 
        for file in files_in_s3:
            if file.endswith((".p")):
                filename = "ModelParameters/"+ file 
                
                h5_files.append(filename)
                print(filename)  

        while True: # Loop until user makes acceptable choice for the input directory
            loadFile = input('Enter the general and specific directory pertaining to the .h5 (weights) file you would like to load: ')
            if loadFile == '': # User enters nothing; break
                break
            elif loadFile in h5_files:

                #load file from s3 
                obj = s3_client.get_object(Bucket=self.bucket, Key=loadFile)

                # Read the binary content of the file
                model_data = obj['Body'].read()

                # Create a temporary file to store the model
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                    # Write the binary data to the temporary file
                    tmp_file.write(model_data)
                    tmp_file_path = tmp_file.name  # Get the path to the temporary file

                # Load the model from the temporary file

                self.modelD = torch.load(tmp_file_path)
                self.modelD.eval()

                #self.modelD = torch.load(tmp_file_path, weights_only = False)

                # Optionally, clean up the temporary file (if delete=False)
                import os
                os.remove(tmp_file_path)  # If you want to delete the temp file manually

                break
                    
            else: # If the modelD attribute does not exist
                print('\nModel is not currently defined - select again.') 
                break

        print("end")
                
    
        



    def Analysis(self):
        
        
        self.import_data_for_testing()

        self.indxIncl = np.nonzero(self.temp_DF_pre_conversion)

        #self.Predict()
        self.OP = np.array(self.OP)
        self.FL = np.array(self.FL) #scale by 2


       

            #convert the data type 
        #reshape image to have the shape (N, H, W, C) --> (N, C, H, W)
        testing_image_OP = np.transpose(self.OP, (0, 3, 1,2))
        testing_image_FL = np.transpose(self.FL, (0, 3, 1,2))

        testing_image_OP = torch.tensor(testing_image_OP, dtype=torch.float32)
        testing_image_FL = torch.tensor(testing_image_FL, dtype=torch.float32)
    
        #load model 
        predict = self.modelD(testing_image_OP, testing_image_FL)  
        
        QF_P = predict[0].detach().numpy()
        DF_P = predict[1].detach().numpy()
        QF_P /= self.params['scaleQF']
        DF_P /= self.params['scaleDF']  

        self.save = 'n'
        
        DF_P = np.reshape(DF_P, (DF_P.shape[0], DF_P.shape[2], DF_P.shape[3]))
        QF_P = np.reshape(QF_P, (QF_P.shape[0], QF_P.shape[2], QF_P.shape[3]))

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

        DF_min = self.get_min(self.DF)
        DFP_min = self.get_min(DF_P)
            
        DF_min = np.array(DF_min)
        DFP_min = np.array(DFP_min)
        #QF_max = np.array(QF_max)
        #QFP_max = np.array(QFP_max)

        #compute absolute mindepth error 
        min_depth_error = np.mean(np.abs(DFP_min - DF_min))
        min_depth_error_std = np.std(np.abs(DFP_min - DF_min))
        print("Average Minimum Depth Error (SD) : {min_depth_error} ({min_depth_error_std})".format(min_depth_error = min_depth_error, min_depth_error_std = min_depth_error_std))

        
        #num_predict_zeros = self.count_predictions_of_zero(DFP_min)
        #print("number of predictions of zero:", num_predict_zeros)
        # SSIM per sample
        DF_ssim =[]
        QF_ssim =[]
        
        ## Plot Correlations
        
        fig, (plt1, plt2) = plt.subplots(1, 2)
        
        plt1.scatter(self.DF[self.indxIncl],DF_P[self.indxIncl],s=1)
        plt1.set_xlim([-5, 15])
        plt1.set_ylim([-5, 15])
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
       


        min_depth_graph = plt.figure()
        
        plt.scatter(DF_min,DFP_min,s=3, label =  "Correct Classification", color = ['blue'])

        DF_min_classify = np.array(DF_min) < 5 
        DFP_min_classify = np.array(DFP_min) < 5
        
        failed_result = DF_min_classify !=DFP_min_classify
        failed_result = np.squeeze(failed_result)
        print(np.shape(failed_result))
        
        plt.scatter(DF_min[failed_result],DFP_min[failed_result],label = "Incorrect Classification", s=3, color = ['red'])
        plt.legend(loc="upper left", prop={'size': 13, 'weight':'bold'})

        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.plot(plt.xlim([0, 10]), plt.ylim([0, 10]),color='k')
        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("Minimum Depth")
        plt.tight_layout()
        font = {'weight': 'bold', 'size':12}
        matplotlib.rc('font', **font)

        min_depth_graph.show()

        num_plot_display = np.shape(self.DF)[0]
        
        num_example_inclusion = [x * 19 for x in range(40)]

        self.save = 'n'
        plot_save_path = './predictions/'

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
                
                print(np.shape(DF_error))
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
            
            for i in range(num_plot_display):#range(num_plot_display):
                print("DF, DF pred: ", DF_min[i], DFP_min[i])
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