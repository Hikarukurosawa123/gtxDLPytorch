import json
from glob import glob
from FOD.Predictor import Predictor
from main import DL
import numpy as np 
import torch
import os 
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

model = torch.load(loadFile, weights_only=False)

#load the config file from the specified path 

model.eval()

