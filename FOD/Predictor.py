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

    def run(self):
        with torch.no_grad():
            
            #for images in self.input_images:

            Images, self.QF, self.DF = self.input_images

            if Images.ndim == 3:
                Images = Images.unsqueeze(0)  # add batch dimension: [1, 2, 101, 101]
            #pil_im = Image.open(images)
            #original_size = pil_im.size

            print(Images.size())

            #tensor_im = self.transform_image(pil_im).unsqueeze(0)
            predict = self.model(Images)

            self.DF =  self.DF.detach().numpy()
            self.QF =  self.QF.detach().numpy()
            QF_P = predict[0].detach().numpy()
            DF_P = predict[1].detach().numpy()

            
            DF_P = np.reshape(DF_P, (DF_P.shape[0], DF_P.shape[2], DF_P.shape[3]))
            QF_P = np.reshape(QF_P, (QF_P.shape[0], QF_P.shape[2], QF_P.shape[3]))
        

            QF_error = abs(self.QF - QF_P)
            DF_error = abs(self.DF - DF_P)

            print("DF_P", DF_P.shape)
            print("QF_P", QF_P.shape)


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
        