import json
from glob import glob
from main import DL
import numpy as np 
import torch
import os 
import matplotlib
import matplotlib.pyplot as plt

from FOD.FocusOnDepth import FocusOnDepth

#with open('config.json', 'r') as f:
#    config = json.load(f)

#input_images = glob('input/*.jpg') + glob('input/*.png')


#inport testing data

from DataImport import MyDataset


data_import_op = DL()
data_import_op.importData(isTesting=True,quickTest=False)


testing_image_OP = data_import_op.OP[:, 2:98,2:98,:]

testing_image_FL = data_import_op.FL[:, 2:98,2:98,:]

testing_label_QF = data_import_op.QF[:, 2:98,2:98]

testing_label_DF = data_import_op.DF[:,2:98,2:98]


#reshape image to have the shape (N, H, W, C) --> (N, C, H, W)
testing_image_OP = np.transpose(testing_image_OP, (0, 3, 1,2))

testing_image_FL = np.transpose(testing_image_FL, (0, 3, 1,2))

#concatenate in the channel dimension 
testing_image_concat = np.concatenate((testing_image_OP, testing_image_FL), axis = 1)

print("post concatenation", testing_image_concat.shape)

testing_image_concat = torch.tensor(testing_image_concat, dtype=torch.float32)

testing_label_QF = torch.tensor(testing_label_QF, dtype=torch.float32)
testing_label_DF = torch.tensor(testing_label_DF, dtype=torch.float32)

testing_set =  (testing_image_concat, testing_label_QF, testing_label_DF)# MyDataset(testing_image_concat,testing_label_QF, testing_label_DF)

#select the model to be running 

#display the model parameters available for export 
pt_files = []
for folder in os.listdir("ModelParameters_PT"):
    #if not folder.endswith((".keras")):
    for file in os.listdir("ModelParameters_PT/"+folder):
        if file.endswith((".pt")):
            filename = "ModelParameters_PT/"+folder+'/'+file
            pt_files.append(filename)

print(pt_files)
while True:
    loadFile = input('Enter the general and specific directory pertaining to the .keras (weights) file you would like to load: ')
    break 



checkpoint = torch.load(loadFile, map_location="cuda" if torch.cuda.is_available() else "cpu")  # or 'cuda' if available
config = checkpoint['config']

#reconstruct the model 

resize = config['Dataset']['transforms']['resize']

num_channel_after_concat = 56  # or dynamically compute from input image if needed

model = FocusOnDepth(
    image_size  = (num_channel_after_concat, resize, resize),
    emb_dim     = config['General']['emb_dim'],
    resample_dim= config['General']['resample_dim'],
    read        = config['General']['read'],
    nclasses    = len(config['Dataset']['classes']) + 1,
    hooks       = config['General']['hooks'],
    model_timm  = config['General']['model_timm'],
    type        = config['General']['type'],
    patch_size  = config['General']['patch_size'],
    config      = config
)



model.load_state_dict(checkpoint['model_state_dict'])

def get_min(DF):
    DF_zeros = np.array(DF)
    DF_min_per_case = np.nanmin(DF_zeros, axis=(1, 2))
    return DF_min_per_case

def run(model, input_images):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to correct device
    model = model.to(device)

    with torch.no_grad():
        Images, QF, DF = input_images

        Images = Images.to(device)
        QF = QF.to(device)
        DF = DF.to(device)

        if Images.ndim == 3:
            Images = Images.unsqueeze(0)  # Add batch dimension

        # Index of valid background regions (DF != 10)
        indxIncl = np.where(DF.cpu().numpy() != 10)  # still use CPU values for indexing

        model.eval()
        predict = model(Images)

        DF = DF.cpu().numpy()
        QF = QF.cpu().numpy()
        QF_P = predict[0].detach().cpu().numpy()
        DF_P = predict[1].detach().cpu().numpy()

        DF_P = np.reshape(DF_P, (DF_P.shape[0], DF_P.shape[2], DF_P.shape[3]))
        QF_P = np.reshape(QF_P, (QF_P.shape[0], QF_P.shape[2], QF_P.shape[3]))

        # Average error
        DF_error = DF_P - DF
        QF_error = QF_P - QF
        DF_erroravg = np.mean(abs(DF_error[indxIncl]))
        DF_errorstd = np.std(abs(DF_error[indxIncl]))
        QF_erroravg = np.mean(abs(QF_error[indxIncl]))
        QF_errorstd = np.std(abs(QF_error[indxIncl]))

        print('Average Depth Error (SD): {}({}) mm'.format(
            float('%.5g' % DF_erroravg), float('%.5g' % DF_errorstd)))
        print('Average Concentration Error (SD): {}({}) ug/mL'.format(
            float('%.5g' % QF_erroravg), float('%.5g' % QF_errorstd)))

        # Max and Min values per sample
        DF_min = get_min(DF)
        DFP_min = get_min(DF_P)

        DF_min = np.array(DF_min)
        DFP_min = np.array(DFP_min)

        print("Ground truth:", DF_min)
        print("Prediction: ", DFP_min)

        # Compute absolute minimum depth error
        min_depth_error = np.mean(np.abs(DFP_min - DF_min))
        min_depth_error_std = np.std(np.abs(DFP_min - DF_min))
        print("Average Minimum Depth Error (SD): {} ({})".format(
            min_depth_error, min_depth_error_std))

        # Scatter plot
        min_depth_graph = plt.figure()
        plt.scatter(DF_min, DFP_min, s=3, label="Correct Classification", color='blue')

        DF_min_classify = DF_min < 5
        DFP_min_classify = DFP_min < 5

        failed_result = DF_min_classify != DFP_min_classify
        failed_result = np.squeeze(failed_result)

        plt.scatter(DF_min[failed_result], DFP_min[failed_result], label="Incorrect Classification", s=3, color='red')
        plt.legend(loc="upper left", prop={'size': 13, 'weight': 'bold'})

        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.plot(plt.xlim(), plt.ylim(), color='k')
        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("Minimum Depth")
        plt.tight_layout()
        font = {'weight': 'bold', 'size': 12}
        matplotlib.rc('font', **font)

        min_depth_graph.show()

        for i in range(DF.shape[0]):
            fig, axs = plt.subplots(2, 3)
            plt.set_cmap('jet')

            plt.colorbar(axs[0, 0].imshow(DF[i, :, :], vmin=0, vmax=15), ax=axs[0, 0], fraction=0.046, pad=0.04)
            axs[0, 0].axis('off')
            axs[0, 0].set_title('True Depth (mm)')

            plt.colorbar(axs[0, 1].imshow(DF_P[i, :, :], vmin=0, vmax=15), ax=axs[0, 1], fraction=0.046, pad=0.04)
            axs[0, 1].axis('off')
            axs[0, 1].set_title('Predicted Depth (mm)')

            plt.colorbar(axs[0, 2].imshow(abs(DF_error[i, :, :]), vmin=0, vmax=15), ax=axs[0, 2], fraction=0.046, pad=0.04)
            axs[0, 2].axis('off')
            axs[0, 2].set_title('|Error (mm)|')

            plt.colorbar(axs[1, 0].imshow(QF[i, :, :], vmin=0, vmax=10), ax=axs[1, 0], fraction=0.046, pad=0.04)
            axs[1, 0].axis('off')
            axs[1, 0].set_title('True Conc (ug/mL)')

            plt.colorbar(axs[1, 1].imshow(QF_P[i, :, :], vmin=0, vmax=10), ax=axs[1, 1], fraction=0.046, pad=0.04)
            axs[1, 1].axis('off')
            axs[1, 1].set_title('Predicted Conc (ug/mL)')

            plt.colorbar(axs[1, 2].imshow(abs(QF_error[i, :, :]), vmin=0, vmax=10), ax=axs[1, 2], fraction=0.046, pad=0.04)
            axs[1, 2].axis('off')
            axs[1, 2].set_title('|Error (ug/mL)|')

            plt.tight_layout()
            plt.show()

run(model, testing_set)