import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from FOD.Trainer import Trainer
from FOD.dataset import AutoFocusDataset

from main import DL
from DataImport import MyDataset

with open('config.json', 'r') as f:
    config = json.load(f)
np.random.seed(config['General']['seed'])

list_data = config['Dataset']['paths']['list_datasets']

#import training dataset 

data_import_op = DL()
data_import_op.importData(isTesting=False,quickTest=False)

#process the data to match expected shape 
#get 95% of the loaded data for training 

#for each case, crop to size 100 * 100
training_image_OP = data_import_op.OP[0:int(data_import_op.DF.shape[0] * 0.95), 2:98,2:98,:]
validation_image_OP = data_import_op.OP[int(data_import_op.DF.shape[0] * 0.95):, 2:98,2:98,:]

training_image_FL = data_import_op.FL[0:int(data_import_op.DF.shape[0] * 0.95), 2:98,2:98,:]
validation_image_FL = data_import_op.FL[int(data_import_op.DF.shape[0] * 0.95):, 2:98,2:98,:]

training_label_QF = data_import_op.QF[0:int(data_import_op.DF.shape[0] * 0.95), 2:98,2:98]
validation_label_QF = data_import_op.QF[int(data_import_op.DF.shape[0] * 0.95):, 2:98,2:98]

training_label_DF = data_import_op.DF[0:int(data_import_op.DF.shape[0] * 0.95),2:98,2:98]
validation_label_DF = data_import_op.DF[int(data_import_op.DF.shape[0] * 0.95):, 2:98,2:98]


#reshape image to have the shape (N, H, W, C) --> (N, C, H, W)
training_image_OP = np.transpose(training_image_OP, (0, 3, 1,2))
validation_image_OP = np.transpose(validation_image_OP, (0, 3, 1,2))

training_image_FL = np.transpose(training_image_FL, (0, 3, 1,2))
validation_image_FL = np.transpose(validation_image_FL, (0, 3, 1,2))

#concatenate in the channel dimension 
training_image_concat = np.concatenate((training_image_OP, training_image_FL), axis = 1)
validation_image_concat = np.concatenate((validation_image_OP, validation_image_FL), axis = 1)

print("post concatenation", training_image_concat.shape)

training_set = MyDataset(training_image_concat,training_label_QF, training_label_DF)
validation_set = MyDataset(validation_image_concat, validation_label_QF, validation_label_DF)

#convert shape of the input to accomodate the expected shape

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = DataLoader(training_set, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=32, shuffle=True)



## train set
#autofocus_datasets_train = []
#for dataset_name in list_data:
#    autofocus_datasets_train.append(AutoFocusDataset(config, dataset_name, 'train'))
#train_data = ConcatDataset(autofocus_datasets_train)
#train_dataloader = DataLoader(train_data, batch_size=config['General']['batch_size'], shuffle=True)

## validation set
#autofocus_datasets_val = []
#for dataset_name in list_data:
#    autofocus_datasets_val.append(AutoFocusDataset(config, dataset_name, 'val'))
#val_data = ConcatDataset(autofocus_datasets_val)
#val_dataloader = DataLoader(val_data, batch_size=config['General']['batch_size'], shuffle=True)

trainer = Trainer(config)
trainer.train(training_loader, validation_loader)

#data_import_op
