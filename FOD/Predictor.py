import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter

from PIL import Image

from FOD.FocusOnDepth import FocusOnDepth
from FOD.utils import create_dir
from FOD.dataset import show
import matplotlib

class Predictor(object):
    def __init__(self, config, input_images):
        self.input_images = input_images
        self.config = config
        self.type = self.config['General']['type']

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']
        self.model = FocusOnDepth(
                    image_size  =   (8,resize,resize),
                    emb_dim     =   config['General']['emb_dim'],
                    resample_dim=   config['General']['resample_dim'],
                    read        =   config['General']['read'],
                    nclasses    =   len(config['Dataset']['classes']) + 1,
                    hooks       =   config['General']['hooks'],
                    model_timm  =   config['General']['model_timm'],
                    type        =   self.type,
                    patch_size  =   config['General']['patch_size'],
        )
        path_model = os.path.join('ModelParameters', config['General']['path_model'], self.model.__class__.__name__, 'Model.p')
        self.model.load_state_dict(
            torch.load(path_model, map_location=self.device)['model_state_dict']
        )
        self.model.eval()
        self.transform_image = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.output_dir = self.config['General']['path_predicted_images']
        create_dir(self.output_dir)


    def get_min(self, DF):

        DF_zeros = np.array(DF)

        DF_min_per_case = np.nanmin(DF_zeros, axis = (1,2))

        return DF_min_per_case

    def run(self):
        with torch.no_grad():
            
            #for images in self.input_images:

            #find background regions 
            #self.indxIncl = self.temp_DF_pre_conversion

            Images, self.QF, self.DF = self.input_images

            if Images.ndim == 3:
                Images = Images.unsqueeze(0)  # add batch dimension: [1, 2, 101, 101]
            #pil_im = Image.open(images)
            #original_size = pil_im.size

            self.indxIncl = np.where(self.DF != 10)


            self.model.eval()
            #tensor_im = self.transform_image(pil_im).unsqueeze(0)
            predict = self.model(Images)

            self.DF =  self.DF.detach().numpy()
            self.QF =  self.QF.detach().numpy()
            QF_P = predict[0].detach().numpy()
            DF_P = predict[1].detach().numpy()

            
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
            
            # Max and Min values per sample

            DF_min = self.get_min(self.DF)
            DFP_min = self.get_min(DF_P)
                
            DF_min = np.array(DF_min)
            DFP_min = np.array(DFP_min)

            print("Ground truth:", DF_min)
            print("prediction: ", DFP_min)

            #compute absolute mindepth error 
            min_depth_error = np.mean(np.abs(DFP_min - DF_min))
            min_depth_error_std = np.std(np.abs(DFP_min - DF_min))
            print("Average Minimum Depth Error (SD) : {min_depth_error} ({min_depth_error_std})".format(min_depth_error = min_depth_error, min_depth_error_std = min_depth_error_std))


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
                
                plt.show()
        