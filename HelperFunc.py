import matplotlib.pyplot as plt
import numpy as np, h5py
import os 
import pandas as pd 
import scipy.io
import tensorflow as tf 
import dicttoxml 
from xml.dom.minidom import parseString
import matplotlib.pyplot as plt 
import matplotlib
#from sklearn import metrics
from keras.models import Model, load_model
import io
import os, time, sys
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import StandardScaler

from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout, MaxPool2D, UpSampling2D, ZeroPadding2D, Activation, SpatialDropout2D

from DataImport import Operations
import boto3
import mat73
from os.path import isfile, join
import time
import tempfile
from torch.utils.data import Dataset, DataLoader

import torch

from DataImport import MyDataset
class MonteCarloDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
    

    
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

    
        
    def plot_loss_curve_from_log(self):
        h5_files = []
        loss_vals = []
        for folder in os.listdir("ModelParameters"):
            if not folder.endswith((".h5",".log",".xml")):
                for file in os.listdir("ModelParameters/"+folder):
                    if file.endswith(".log"):
                        filename = "ModelParameters/"+folder+'/'+file
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
    
    
    
    def count_predictions_of_zero(self, DF_P):
        bool_DF_P_equals_zero = DF_P == 0

        return np.sum(bool_DF_P_equals_zero)
    
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
            if file.endswith((".keras", ".pt")):
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
                with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
                    # Write the binary data to the temporary file
                    tmp_file.write(model_data)
                    tmp_file_path = tmp_file.name  # Get the path to the temporary file

                # Load the model from the temporary file

                if not self.run_torch:
                    self.modelD = load_model(tmp_file_path, compile=False)
                else: 
                    self.modelD = torch.load(tmp_file_path, weights_only = False)

                #print(tmp_file_path)

                # Optionally, clean up the temporary file (if delete=False)
                import os
                os.remove(tmp_file_path)  # If you want to delete the temp file manually

                break
                    
            else: # If the modelD attribute does not exist
                print('\nModel is not currently defined - select again.') 
                break

        print("end")
                
    
                
    def predict_uncertainty_with_aleatoric_uncertainty(self): 
        self.load()

        num_dropout_ensembles = 16
        num_examples = self.OP.shape[0]
        image_dim = self.OP.shape[1]

        depth_prediction_ensembles = np.ndarray((num_examples, image_dim, image_dim, num_dropout_ensembles))
        concentration_prediction_ensembles = np.ndarray((num_examples, image_dim, image_dim, num_dropout_ensembles))

        depth_uncertainty_ensembles = np.ndarray((num_examples, image_dim, image_dim, num_dropout_ensembles))
        concentration_uncertainty_ensembles = np.ndarray((num_examples, image_dim, image_dim, num_dropout_ensembles))

        for i in range(num_dropout_ensembles):
     

            tmp = self.modelD.predict([self.OP, self.FL], batch_size = 1)  
            tmp = np.array(tmp)
            depth_prediction_ensembles[:,:,:,i] = tmp[1, :,:,:,0].squeeze()
            concentration_prediction_ensembles[:,:,:,i] = tmp[0, :,:,:,0].squeeze()
            depth_uncertainty_ensembles[:,:,:,i] = tmp[1, :,:,:,1].squeeze()
            concentration_uncertainty_ensembles[:,:,:,i] = tmp[0, :,:,:,1].squeeze()

        depth_prediction_mean = np.zeros((num_examples, image_dim, image_dim))
        concentration_prediction_mean = np.zeros((num_examples, image_dim, image_dim))

        depth_prediction_uncertainty_aleatoric = np.zeros((num_examples, image_dim, image_dim))
        concentration_prediction_uncertainty_aleatoric = np.zeros((num_examples, image_dim, image_dim))
        depth_prediction_uncertainty_epistemic = np.zeros((num_examples, image_dim, image_dim))
        concentration_prediction_uncertainty_epistemic = np.zeros((num_examples, image_dim, image_dim))
        #take the mean of each example 
        for i in range(num_examples):
            depth_prediction_mean[i, :,:] = np.mean(depth_prediction_ensembles[i,:,:,:].squeeze(), axis = 2)
            concentration_prediction_mean[i, :,:] = np.mean(concentration_prediction_ensembles[i,:,:,:].squeeze(), axis = 2)
            #get epistemic uncertainty 
            depth_prediction_uncertainty_epistemic[i, :,:] = np.std(depth_prediction_ensembles[i,:,:,:].squeeze(), axis = 2) 
            concentration_prediction_uncertainty_epistemic[i, :,:] = np.std(concentration_prediction_ensembles[i,:,:,:].squeeze(), axis = 2)
            depth_prediction_uncertainty_aleatoric[i, :,:] = np.mean(depth_uncertainty_ensembles[i,:,:,:].squeeze(), axis = 2) 
            concentration_prediction_uncertainty_aleatoric[i, :,:] = np.mean(concentration_uncertainty_ensembles[i,:,:,:].squeeze(), axis = 2)



        DF_error = depth_prediction_mean - self.DF.squeeze()
        QF_error = concentration_prediction_mean - self.QF.squeeze()
        
        font = {'size'   : 7}
        plt.rc('font', **font)
        for i in range(200):

            fig, axs = plt.subplots(2,5)
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
            plt.colorbar(axs[0,3].imshow(depth_prediction_uncertainty_aleatoric[i,:,:],vmin=0,vmax=2), ax=axs[0, 3],fraction=0.046, pad=0.04)
            axs[0,3].axis('off')
            axs[0,3].set_title('|Data Unc. Pred|')
            plt.colorbar(axs[0,4].imshow(depth_prediction_uncertainty_epistemic[i,:,:],vmin=0,vmax=2), ax=axs[0, 4],fraction=0.046, pad=0.04)
            axs[0,4].axis('off')
            axs[0,4].set_title('|Model Unc. Pred|')
            plt.tight_layout()

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
            plt.colorbar(axs[1,3].imshow(concentration_prediction_uncertainty_aleatoric[i,:,:],vmin=0,vmax=2), ax=axs[1, 3],fraction=0.046, pad=0.04)
            axs[1,3].axis('off')
            axs[1,3].set_title('|Data Unc. Pred|')
            plt.colorbar(axs[1,4].imshow(concentration_prediction_uncertainty_epistemic[i,:,:],vmin=0,vmax=2), ax=axs[1, 4],fraction=0.046, pad=0.04)
            axs[1,4].axis('off')
            axs[1,4].set_title('|Model Unc. Pred|')

            plt.tight_layout()
            
            plt.show()
                
   
    def Analysis_single(self):
        
        self.importData()

        self.indxIncl = np.nonzero(self.temp_DF_pre_conversion)

        #self.Predict()
        predict = self.modelD.predict([self.OP, self.FL], batch_size = 1)  

        print(self.OP)
        print(self.FL)
        
        QF_P = predict[0] 
        DF_P = predict[1]
        
        print(DF_P)

        
        DF_min = self.get_min(self.DF)
        QF_max = self.get_max(self.QF)
        
        print("DF error: ", np.mean(np.abs(DF_min - DF_P)))
        print("QF error: ", np.mean(np.abs(QF_max - QF_P)))

        plt.scatter(DF_min,DF_P, s=3)
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
        
        
        plt.scatter(QF_max,QF_P, s=3)
        plt.legend(loc="upper left", prop={'size': 13, 'weight':'bold'})

        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.plot(plt.xlim([0, 10]), plt.ylim([0, 10]),color='k')
        plt.ylabel("Predicted Concentration (ug/mL)")
        plt.xlabel("True Concentration (ug/mL)")
        plt.title("Maximum Concentration")
        plt.tight_layout()
        font = {'weight': 'bold', 'size':12}
        matplotlib.rc('font', **font)

        
    def visualize_feature_maps(self, num_layer):

        #choose model 
        self.modelD_visualize = Model(inputs=self.modelD.inputs, outputs=self.modelD.layers[num_layer].output)#,outFL])

        print("loaded")


    def Analysis(self):
        
        
        self.import_data_for_testing()

        self.indxIncl = np.nonzero(self.temp_DF_pre_conversion)

        #self.Predict()
        self.OP = np.array(self.OP)
        self.FL = np.array(self.FL) #scale by 2


        if not self.run_torch:
            predict = self.modelD.predict([self.OP, self.FL], batch_size = 32)  
        else:

            #convert the data type 
            #reshape image to have the shape (N, H, W, C) --> (N, C, H, W)
            testing_image_OP = np.transpose(self.OP, (0, 3, 1,2))
            testing_image_FL = np.transpose(self.FL, (0, 3, 1,2))

            testing_image_OP = torch.tensor(testing_image_OP, dtype=torch.float32)
            testing_image_FL = torch.tensor(testing_image_FL, dtype=torch.float32)
            testing_set = MyDataset(testing_image_OP, testing_image_FL,self.QF, self.DF)

            #convert shape of the input to accomodate the expected shape

            # Create data loaders for our datasets; shuffle for training, not for validation
            #testing_loader = DataLoader(testing_set, batch_size=32, shuffle=True)

            #load model 
            predict = self.modelD(testing_image_OP, testing_image_FL)  
        

        
        #predict = self.modelD.predict_on_batch([self.OP, self.FL])
        QF_P = predict[0].detach().numpy()
        DF_P = predict[1].detach().numpy()
       
        QF_P /= self.params['scaleQF']
        DF_P /= self.params['scaleDF']  

        self.save = 'n'
        #if self.save == 'y':
        #    plot_save_path = 'Predictions/' + self.exportName + '/'

        #reshape so that last dimension is removed 

        if not self.run_torch:
            DF_P = np.reshape(DF_P, (DF_P.shape[0], DF_P.shape[1], DF_P.shape[2]))
            QF_P = np.reshape(QF_P, (QF_P.shape[0], QF_P.shape[1], QF_P.shape[2]))
        else: 
            DF_P = np.reshape(DF_P, (DF_P.shape[0], DF_P.shape[2], DF_P.shape[3]))
            QF_P = np.reshape(QF_P, (QF_P.shape[0], QF_P.shape[2], QF_P.shape[3]))



        ## Error Stats
        # Average error
        
        print(np.shape(DF_P), np.shape(self.DF))
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

        print(DF_P_min)
        
        #if self.save in ['Y','y']:
        #    plot_save_path = 'Predictions/' + self.exportName + '/'

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
        '''
        for i in range(DF_P.shape[0]):
            #df_p = np.reshape(DF_P[i,:,:],(DF_P.shape[1],DF_P.shape[2])) # reshape predicted
            #df_t = np.reshape(self.DF[i,:,:],(self.DF.shape[1],self.DF.shape[2])) # reshape true
            df_ssim = ssim(df_p,df_t,data_range=max(df_p.max(),df_t.max())-min(df_p.min(),df_t.min()))
            DF_ssim.append(df_ssim)
            #qf_p = np.reshape(QF_P[i,:,:],(QF_P.shape[1],QF_P.shape[2])) # reshape predicted
            #qf_t = np.reshape(self.QF[i,:,:],(self.QF.shape[1],self.QF.shape[2])) # reshape true
            qf_ssim = ssim(qf_p,qf_t,data_range=max(qf_p.max(),qf_t.max())-min(qf_p.min(),qf_t.min()))
            QF_ssim.append(qf_ssim)
        print('Overall Depth SSIM: {}'.format(float('%.5g' % np.mean(DF_ssim))))
        print('Overall Concentration SSIM: {}'.format(float('%.5g' % np.mean(QF_ssim))))
        '''
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
        '''
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
        
        '''
       

        min_depth_graph = plt.figure()
        #plt.scatter(DF_max[beg:end],DFP_max[beg:end],s=1)
        #plt.scatter(DF_max,DFP_max,s=3, label = "Correct Classification", color = ['blue'])
        plt.scatter(DF_max,DFP_max,s=3, label =  "Correct Classification", color = ['blue'])

        DF_max_classify = np.array(DF_max) < 5 
        DFP_max_classify = np.array(DFP_max) < 5
        
        failed_result = DF_max_classify !=DFP_max_classify
        failed_result = np.squeeze(failed_result)
        print(np.shape(failed_result))
        
        plt.scatter(DF_max[failed_result],DFP_max[failed_result],label = "Incorrect Classification", s=3, color = ['red'])
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

        
        if self.save in ['Y','y']:
            plot_save_path_min_depth = plot_save_path + 'min_depth.png'
            plt.savefig(plot_save_path_min_depth, dpi=100, bbox_inches='tight')
        min_depth_graph.show()

        num_plot_display = np.shape(self.DF)[0]
        
        num_example_inclusion = [x * 19 for x in range(40)]

        self.save = 'n'
        plot_save_path = './predictions4/'

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
                print("DF, DF pred: ", DF_min[i], DF_P_min[i])
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