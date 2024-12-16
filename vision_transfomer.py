#!pip install einops
from __future__ import print_function

import math

import six
from einops.layers.tensorflow import Rearrange
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout

from tensorflow.keras import datasets
from keras.models import Model, load_model

import logging
import numpy as np

from fastprogress import master_bar, progress_bar
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
from sklearn import metrics
from csv import writer, reader
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
import numpy as np, h5py
import os, time, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import scipy.io

# Tensorflow / Keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Input, concatenate, Conv2D, add, Conv3D, Reshape, SeparableConv2D, Dropout
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing import image

from gtxDLClassAWSUtils import Utils

import boto3 
import io
import openpyxl
def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_activation(identifier):
    """Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.
    It checks string first and if it is one of customized activation not in TF,
    the corresponding activation will be returned. For non-customized activation
    names and callable identifiers, always fallback to tf.keras.activations.get.
    Args:
        identifier: String name of the activation function or callable.
    Returns:
        A Python function corresponding to the activation function.
    """
    if isinstance(identifier, six.string_types):
        name_to_fn = {"gelu": gelu}
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)


class Residual(tf.keras.Model):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x):
        return self.fn(x) + x


class PreNorm(tf.keras.Model):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.fn = fn

    def call(self, x):
        return self.fn(self.norm(x))


class FeedForward(tf.keras.Model):

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim, activation=get_activation('gelu')),
                                        tf.keras.layers.Dense(dim)])

    def call(self, x):
        return self.net(x)

class Attention(tf.keras.Model):

    def __init__(self, dim, heads = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
        self.to_out = tf.keras.layers.Dense(dim)

        self.rearrange_qkv = Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)
        self.rearrange_out = Rearrange('b h n d -> b n (h d)')

    def call(self, x):
        # print("before self.to_qkv: ", x)
        qkv = self.to_qkv(x)
        # print("qkv", qkv.shape)
        qkv = self.rearrange_qkv(qkv)
        # print("qkv 2", qkv.shape)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        # print("q ", q.shape)
        # print("k ", k.shape)
        # print("v ", v.shape)

        dots = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # print("dots: ", dots.shape)
        attn = tf.nn.softmax(dots,axis=-1)
        # print("attention shape", attn.shape)
        out = tf.einsum('bhij,bhjd->bhid', attn, v)
        # print("out shape 1", out.shape)
        out = self.rearrange_out(out)
        # print("out shape 2", out.shape)
        out =  self.to_out(out)
        # print("out shape 3", out.shape)
        return out

class Transformer(tf.keras.Model):

    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ])
        self.net = tf.keras.Sequential(layers)

    def call(self, x):
        return self.net(x)

class ViT(tf.keras.Model):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=6):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim
        self.pos_embedding = self.add_weight(
                                             [num_patches + 1,
                                                    dim],
                                             tf.keras.initializers.RandomNormal(),
                                             tf.float32)

        
        self.patch_to_embedding = tf.keras.layers.Dense(dim)
        self.cls_token = self.add_weight(
                                         shape=[1,
                                                1,
                                                dim],
                                         initializer=tf.keras.initializers.RandomNormal(),
                                         dtype=tf.float32)

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = tf.identity

        self.mlp_head = tf.keras.Sequential([tf.keras.layers.Dense(mlp_dim, activation=get_activation('gelu')),
                                        tf.keras.layers.Dense(num_classes)])
        


    @tf.function
    def call(self, img):
        shapes = img.shape
        # print("x1: ", img.shape)
        x = self.rearrange(img)
        # print("x2: ", x.shape)
        x = self.patch_to_embedding(x)
        # print("x3: ", x.shape)
        cls_tokens = tf.broadcast_to(self.cls_token,(shapes[0],1,self.dim))
        x = tf.concat((cls_tokens, x), axis=1)
        # print("x4: ", x.shape)

        x += self.pos_embedding
        # print("x5: ", x.shape)

        x = self.transformer(x)
        # print("x6: ", x.shape)

        x = self.to_cls_token(x[:, 0])
        # print("x7: ", x.shape)

        return self.mlp_head(x)
     

    logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 1e-4
    # checkpoint settings
    ckpt_path = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)



class Help(Utils):

    def __init__(self):
        self.bucket = '20240920-matthew'#'20240909-hikaru'
        isCase = input('Choose a case:\n Default\n Default4fx')
        self.case = isCase
        self.params = self.Defaults(self.case) # Define some default parameters based on the selected case
        self.isTesting = False
        
    def importData(self,isTesting=False,quickTest=False): #for training data 
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
        
        

        self.file_key = str(input('Enter the name of the dataset you want to import e.g. matALL.mat '))
        
        self.params["training_file_name"] = self.file_key
        obj = s3_client.get_object(Bucket=self.bucket, Key=self.file_key)
        dataTemp = obj['Body'].read()
        self.dataset = h5py.File(io.BytesIO(dataTemp))
        self.FL = self.dataset['F']
        three_branch = False
        if three_branch:
            self.DF = self.dataset['DF_sub']
        else:
            self.DF = self.dataset['DF']
        self.OP = self.dataset['OP']
        self.QF = self.dataset['QF']
        self.RE = self.dataset['RE']
        self.temp_DF_pre_conversion = self.DF

        
        self.background_val = str(input('Enter the value of the background'))
        if self.background_val == '':
            self.background_val = 0 #default to zero
        
        self.convert_background_val() #convert values of DF background

        # Check whether the user is using the single or multiple MAT format 
        # I.e., looking at individual MAT files (getDims=3) or looking at MAT files with more than one sample (getDim=4)
        getDims = len(np.shape(self.FL))
        print("getDims", getDims)
        if getDims == 4:
            numSets = int(np.shape(self.FL)[3])
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

                sizeFx = int(np.shape(self.FL)[0])
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
            self.FL = self.FL[:,:64,:64,0:numSets]

            self.FL = np.reshape(self.FL[indxFx,:,:,0:numSets],(len(indxFx),self.params['xX'],self.params['yY'],numSets))
            self.RE = self.RE[:,:64,:64,0:numSets]

            self.RE = np.reshape(self.RE[indxFx,:,:,0:numSets],(len(indxFx),self.params['xX'],self.params['yY'],numSets))


            self.OP = self.OP[:,:64,:64,0:numSets]
            self.DF = self.DF[:64,:64,0:numSets]
            self.DF = np.reshape(self.DF,(self.params['xX'],self.params['yY'],numSets))
            self.temp_DF_pre_conversion = self.temp_DF_pre_conversion[:64,:64,0:numSets]
            self.temp_DF_pre_conversion = np.reshape(self.temp_DF_pre_conversion,(self.params['xX'],self.params['yY'],numSets))
            self.QF = self.QF[:64,:64,0:numSets]
            self.QF = np.reshape(self.QF,(self.params['xX'],self.params['yY'],numSets))

            # Reorder data
            #self.RE = np.swapaxes(self.RE,0,3)

            #self.FL = np.swapaxes(self.FL,0,3)
            #self.OP = np.swapaxes(self.OP,0,3)
            #self.DF = np.moveaxis(self.DF,2,0)
            #self.temp_DF_pre_conversion = np.moveaxis(self.temp_DF_pre_conversion,2,0)
            #self.QF = np.moveaxis(self.QF,2,0)

            self.RE = np.transpose(self.RE, (3,0,1,2))
            self.FL = np.transpose(self.FL, (3,0,1,2))
            self.OP = np.transpose(self.OP, (3,0,1,2))
            self.DF = np.transpose(self.DF, (2,0,1))
            self.temp_DF_pre_conversion = np.transpose(self.temp_DF_pre_conversion, (2,0,1))
            self.QF = np.transpose(self.QF, (2,0,1))

            stop = time.perf_counter()
            print('Load time = ' + str(stop-start))

            # Apply scale
            self.OP[:,0,:,:] *= self.params['scaleOP0']
            # self.OP[:,:,:,1] *= self.params['scaleOP1']
            # self.DF *= self.params['scaleDF']
            # self.QF *= self.params['scaleQF']
            # self.RE[:,:,:,0] *= self.params['scaleRE']

            if 'scaleFL0' in self.params:
                for i in range(self.params['nF']):
                    scaleN = 'scaleFL'+ str(i)
                    self.FL[:,:,:,i] *= self.params[scaleN]
                    print(np.mean(self.FL[:,:,:,i]))
            else:
                self.FL *= self.params['scaleFL']


            print("RE shape", self.RE.shape)

    def importDataTest(self,isTesting=True,quickTest=False): #for training data 
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

        # Enter the name of the dataset you want to import
        # Note: To import new data, go to the desired bucket in AWS and upload data
        
        

        self.file_key = str(input('Enter the name of the dataset you want to import e.g. matALL.mat '))
        
        self.params["training_file_name"] = self.file_key
        obj = s3_client.get_object(Bucket=self.bucket, Key=self.file_key)
        dataTemp = obj['Body'].read()
        self.dataset = h5py.File(io.BytesIO(dataTemp))
        # self.FL = self.dataset['F']
        
        self.DF_test = self.dataset['DF']
        # self.QF = self.dataset['QF']
        self.RE_test = self.dataset['RE']
        # self.temp_DF_pre_conversion = self.DF

        
        self.background_val = str(input('Enter the value of the background'))
        if self.background_val == '':
            self.background_val = 0 #default to zero
        
        self.convert_background_val_test() #convert values of DF background

        # Check whether the user is using the single or multiple MAT format 
        # I.e., looking at individual MAT files (getDims=3) or looking at MAT files with more than one sample (getDim=4)
        getDims = len(np.shape(self.RE_test))
        print("getDims", getDims)
        if getDims == 4:
            numSets = int(np.shape(self.RE_test)[3])
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

                sizeFx = int(np.shape(self.RE_test)[0])
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
                numSets = int(np.shape(self.RE_test)[3])
                sizeFx = self.params['nF']
                indxFx = np.arange(0,sizeFx,1)
        
            start = time.perf_counter()
            # self.FL = self.FL[:,:64,:64,0:numSets]

            # self.FL = np.reshape(self.FL[indxFx,:,:,0:numSets],(len(indxFx),self.params['xX'],self.params['yY'],numSets))
            self.RE_test = self.RE_test[:,:64,:64,0:numSets]

            self.RE_test = np.reshape(self.RE_test[indxFx,:,:,0:numSets],(len(indxFx),self.params['xX'],self.params['yY'],numSets))


            # self.OP = self.OP[:,:64,:64,0:numSets]
            self.DF_test = self.DF_test[:64,:64,0:numSets]
            self.DF_test = np.reshape(self.DF_test,(self.params['xX'],self.params['yY'],numSets))
            # self.temp_DF_pre_conversion = self.temp_DF_pre_conversion[:64,:64,0:numSets]
            # self.temp_DF_pre_conversion = np.reshape(self.temp_DF_pre_conversion,(self.params['xX'],self.params['yY'],numSets))
            # self.QF = self.QF[:64,:64,0:numSets]
            # self.QF = np.reshape(self.QF,(self.params['xX'],self.params['yY'],numSets))

            self.RE_test = np.transpose(self.RE_test, (3,0,1,2))
            # self.FL = np.transpose(self.FL, (3,0,1,2))
            # self.OP = np.transpose(self.OP, (3,0,1,2))
            self.DF_test = np.transpose(self.DF_test, (2,0,1))
            # self.temp_DF_pre_conversion = np.transpose(self.temp_DF_pre_conversion, (2,0,1))
            # self.QF = np.transpose(self.QF, (2,0,1))

            stop = time.perf_counter()
            print('Load time = ' + str(stop-start))

            # Apply scale
            # self.OP[:,0,:,:] *= self.params['scaleOP0']
            # self.OP[:,:,:,1] *= self.params['scaleOP1']
            # self.DF *= self.params['scaleDF']
            # self.QF *= self.params['scaleQF']
            # self.RE_test[:,:,:,0] *= self.params['scaleRE']

            if 'scaleFL0' in self.params:
                for i in range(self.params['nF']):
                    scaleN = 'scaleFL'+ str(i)
                    # self.FL[:,:,:,i] *= self.params[scaleN]
                    print(np.mean(self.FL[:,:,:,i]))
            else:
                # self.FL *= self.params['scaleFL']
                pass


            print("RE test shape", self.RE_test.shape)


    def convert_background_val(self):
        self.DF = np.array(self.DF)
        for x in range(self.DF.shape[0]): #for each case 
            for i in range(self.DF.shape[1]):
                    
                DF_zeros_per_column = self.DF[x, i, :] == 0
                self.DF[x, i, DF_zeros_per_column] = self.background_val


    def convert_background_val_test(self):
        self.DF_test = np.array(self.DF_test)
        for x in range(self.DF_test.shape[0]): #for each case 
            for i in range(self.DF_test.shape[1]):
                    
                DF_zeros_per_column = self.DF_test[x, i, :] == 0
                self.DF_test[x, i, DF_zeros_per_column] = self.background_val


    def call_trainer(self):
        self.importDataTest()
        self.importData()

        
        train_images = tf.cast(self.RE, dtype=tf.float32)
        test_images = tf.cast(self.RE_test,dtype=tf.float32)

        DF_max_train = np.max(self.DF, axis = (1,2))
        DF_max_test = np.max(self.DF_test, axis = (1,2))

        DF_train_flattened = np.reshape(self.DF, (-1, self.DF.shape[1] * self.DF.shape[2]))
        DF_test_flattened = np.reshape(self.DF_test, (-1, self.DF_test.shape[1] * self.DF_test.shape[2]))


        train_labels = tf.cast(DF_train_flattened,dtype=tf.float32)
        test_labels = tf.cast(DF_test_flattened,dtype=tf.float32)

        train_x = tf.data.Dataset.from_tensor_slices(train_images,)
        train_y = tf.data.Dataset.from_tensor_slices(train_labels)
        train_dataset = tf.data.Dataset.zip((train_x,train_y))
        test_x = tf.data.Dataset.from_tensor_slices(test_images)
        test_y = tf.data.Dataset.from_tensor_slices(test_labels)
        test_dataset = tf.data.Dataset.zip((test_x,test_y))
     
     
        model_config = {"image_size":64,
                        "patch_size":4,
                        "num_classes":4096,
                        "dim":100,
                        "depth":10,
                        "heads":10,
                        "mlp_dim":1000}
        tconf = TrainerConfig(max_epochs=500, batch_size=64, learning_rate=5e-4)

        trainer = Trainer(ViT, model_config, train_dataset, len(train_images), test_dataset, len(test_images), tconf)
        
        trainer.train()


class Trainer:

    def __init__(self, model, model_config, train_dataset, train_dataset_len, test_dataset, test_dataset_len, config):
        self.train_dataset = train_dataset.batch(config.batch_size)
        self.train_dataset_len = train_dataset_len
        self.test_dataset = test_dataset
        self.test_dataset_len = None
        self.test_dist_dataset = None
        if self.test_dataset:
            self.test_dataset = test_dataset.batch(config.batch_size)
            self.test_dataset_len = test_dataset_len
        self.config = config
        self.tokens = 0
        self.strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        if len(tf.config.list_physical_devices('GPU')) > 1:
            self.strategy = tf.distribute.MirroredStrategy()

        with self.strategy.scope():
            self.model = model(**model_config)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
            self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
            if self.test_dataset:
                self.test_dist_dataset = self.strategy.experimental_distribute_dataset(self.test_dataset)

    def save_checkpoints(self):
        if self.config.ckpt_path is not None:
            self.model.save_weights(self.config.ckpt_path)


    def train(self):

        train_loss_metric = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
        test_loss_metric = tf.keras.metrics.Mean('testing_loss', dtype=tf.float32)

        train_accuracy = tf.keras.metrics.Accuracy('training_accuracy', dtype=tf.float32)
        test_accuracy = tf.keras.metrics.Accuracy('testing_accuracy', dtype=tf.float32)

        @tf.function
        def train_step(dist_inputs):

            def step_fn(inputs):

                X, Y = inputs

                with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                    logits = self.model(X,training=True)
                    num_labels = tf.shape(logits)[-1]
                    label_mask = tf.math.logical_not(Y < 0)
                    #label_mask = tf.reshape(label_mask,(-1,))
                    logits = tf.reshape(logits,(-1,num_labels))
                    #logits_masked = tf.boolean_mask(logits,label_mask)
                    #label_ids = tf.reshape(Y,(-1,))
                    # label_ids_masked = tf.boolean_mask(label_ids,label_mask)
                    # cross_entropy = self.cce(label_ids_masked, logits_masked)
                    # loss = tf.reduce_sum(cross_entropy) * (1.0 / self.config.batch_size)
                    # y_pred = tf.argmax(tf.nn.softmax(logits,axis=-1),axis=-1)

                    loss = tf.math.abs(logits - Y)

                    #train_accuracy.update_state(tf.squeeze(Y),y_pred)

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
                return loss

            per_example_losses = self.strategy.run(step_fn, args=(dist_inputs,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss

        @tf.function
        def test_step(dist_inputs):

            def step_fn(inputs):

                X, Y = inputs
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                logits = self.model(X,training=False)
                num_labels = tf.shape(logits)[-1]
                label_mask = tf.math.logical_not(Y < 0)
                label_mask = tf.reshape(label_mask,(-1,))
                logits = tf.reshape(logits,(-1,num_labels))
                #logits_masked = tf.boolean_mask(logits,label_mask)
                #label_ids = tf.reshape(Y,(-1,))
                #label_ids_masked = tf.boolean_mask(label_ids,label_mask)
                #cross_entropy = self.cce(label_ids_masked, logits_masked)
                loss = tf.math.abs(logits - Y)
                #y_pred = tf.argmax(tf.nn.softmax(logits,axis=-1),axis=-1)
                #test_accuracy.update_state(tf.squeeze(Y),y_pred)

                return loss

            per_example_losses = self.strategy.run(step_fn, args=(dist_inputs,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
            mean_loss = sum_loss / self.config.batch_size
            return mean_loss

        train_pb_max_len = math.ceil(float(self.train_dataset_len)/float(self.config.batch_size))
        test_pb_max_len = math.ceil(float(self.test_dataset_len)/float(self.config.batch_size)) if self.test_dataset else None

        epoch_bar = master_bar(range(self.config.max_epochs))
        with self.strategy.scope():
            for epoch in epoch_bar:
                for inputs in progress_bar(self.train_dist_dataset,total=train_pb_max_len,parent=epoch_bar):
                    loss = train_step(inputs)
                    self.tokens += tf.reduce_sum(tf.cast(inputs[1]>=0,tf.int32)).numpy()
                    train_loss_metric(loss)
                    epoch_bar.child.comment = f'training loss : {train_loss_metric.result()}'
                print(f"epoch {epoch+1}: train loss {train_loss_metric.result():.5f}. train accuracy {train_accuracy.result():.5f}")
                # train_loss_metric.reset_states()
                # train_accuracy.reset_states()

                if self.test_dist_dataset:
                    for inputs in progress_bar(self.test_dist_dataset,total=test_pb_max_len,parent=epoch_bar):
                        loss = test_step(inputs)
                        test_loss_metric(loss)
                        epoch_bar.child.comment = f'testing loss : {test_loss_metric.result()}'
                    print(f"epoch {epoch+1}: test loss {test_loss_metric.result():.5f}. test accuracy {test_accuracy.result():.5f}")
                    # test_loss_metric.reset_states()
                    # test_accuracy.reset_states()

                self.save_checkpoints()
     

