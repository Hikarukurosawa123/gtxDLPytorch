import json
from glob import glob
from FOD.Predictor import Predictor
from main import DL
import numpy as np 
import torch
with open('config.json', 'r') as f:
    config = json.load(f)

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


predictor = Predictor(config, testing_set)
predictor.run()