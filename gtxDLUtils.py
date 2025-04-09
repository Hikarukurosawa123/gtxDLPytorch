from csv import writer, reader
import os
# Tensorflow / Keras
import tensorflow as tf
from tensorflow import keras
from keras.layers import BatchNormalization, Conv2D, add, Conv3D

class Utils():
    # Relevant resblock functions (Keras API)
    def resblock_2D(self, num_filters, size_filter, stride_filter, x):
        """Residual block for 2D input excluding batch normalization layers"""
        Fx = Conv2D(filters=num_filters, kernel_size=size_filter, strides=stride_filter,padding='same', activation='relu', 
                    data_format="channels_last")(x)
        Fx = Conv2D(filters=num_filters, kernel_size=size_filter, padding='same', activation='relu', data_format="channels_last")(Fx)
        output = add([Fx, x])
        return output

    def resblock_2D_BN(self,num_filters, size_filter, stride_filter, x):
        """Residual block for 2D input including batch normalization layers"""
        Fx = Conv2D(filters=num_filters, kernel_size=size_filter, strides=stride_filter, padding='same', activation='relu', 
                    data_format="channels_last")(x)
        Fx = BatchNormalization()(Fx)
        Fx = Conv2D(filters=num_filters, kernel_size=size_filter, strides=stride_filter, padding='same', activation='relu', 
                    data_format="channels_last")(Fx)
        Fx = BatchNormalization()(Fx)
        output = add([Fx, x])
        return output

    def resblock_3D(self,num_filters, size_filter, stride_filter, x):
        """Residual block for 3D input excluding batch normalization layers"""
        Fx = Conv3D(filters=num_filters, kernel_size=size_filter, strides=stride_filter, padding='same', activation='relu', 
                    data_format="channels_last")(x)
        Fx = Conv3D(filters=num_filters, kernel_size=size_filter, strides=stride_filter, padding='same', activation='relu', 
                    data_format="channels_last")(Fx)
        output = add([Fx, x])
        return output

    def resblock_3D_BN(self,num_filters, size_filter, stride_filter, x):
        """Residual block for 3D input including batch normalization layers"""
        Fx = Conv3D(filters=num_filters, kernel_size=size_filter, strides=stride_filter, padding='same', activation='relu', 
                    data_format="channels_last")(x)
        Fx = BatchNormalization()(Fx)
        Fx = Conv3D(filters=num_filters, kernel_size=size_filter, strides=stride_filter, padding='same', activation='relu', 
                    data_format="channels_last")(Fx)
        Fx = BatchNormalization()(Fx)
        output = add([Fx, x])
        return output
    
    def conv2Plus1D(self, num_filters, x):
        """(2+1)Conv to replace the 3DConv"""
        Fx = Conv3D(filters=num_filters, kernel_size=(3,3,1), strides=(1,1,1), padding='same', activation='relu', 
                    data_format="channels_last")(x)
        output = Conv3D(filters=num_filters, kernel_size=(1,1,6), strides=(1,1,1), padding='same', activation='relu', 
                    data_format="channels_last")(Fx)
        return output
    
    def scheduler(epoch, lr):
        """Method to define the learning rate schedule used in the LRScheduler callback (as of 2022/08/03 we are not using this)"""
        if epoch != 0:
            if epoch <= 20: # Prior to the 20th epoch we only want to decay the LR every 5 epochs
                if epoch % 5 == 0:
                    return lr/(1+self.params['decayRate']*epoch)
                else:
                    return lr
            elif epoch > 20:
                if epoch % 3 == 0: # After the 20th epoch we start decaying the LR more quickly
                    return lr/(1+self.params['decayRate']*epoch)   
                else:
                    return lr
        else:
            return lr


    def Defaults(self, case):
        """Default values for DL, depending on the case"""
        params = {} # Hyperparameter dictionary for dynamic tuning in gtxDLImplmentation

        # General Parameters
        params['activation'] = 'relu' # Rectified linear unit: new value of pixel = max(0,val), where val is the current value of the pixel
        params['optimizer']  = 'Adam' # Options are SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
        params['epochs'] = 200
        params['learningRate'] = 5e-5 # Initial learning rate, subject to scheduled decay (see DL callbacks)
        # Scaling parameters; goal of scaling is to make our diferently measured inputs (i.e., QF, DF, OP, FL) scale-invariant,
        params['scaleRE'] = 1 # Refl ~ [0.15 - 0.69]
        params['scaleDF'] = 1 # z ~ [1 - 5] (mm)

        if case == 'Default4Fx':
            params['nF'] = 4
            params['xX'] = 65 # Width of input maps
            params['yY'] = 65 # Height of input maps
            params['batch'] = 32
            params['decayRate'] = 0.3
            params['nFilters3D'] = 128 # Need to decrease filters due to OOM error
            params['nFilters2D'] = 128
            params['kernelConv3D'] = (3,3,4) 
            params['strideConv3D'] = (1,1,1)
            params['kernelResBlock3D'] = (3,3,4)
            params['kernelConv2D'] = (3,3)
            params['strideConv2D'] = (1,1)
            params['kernelResBlock2D'] = (3,3)
        elif case == 'Default6Fx_HighRes_SmallFOV':
            params['nF'] = 6
            params['xX'] = 65 # Width of input maps
            params['yY'] = 65 # Height of input maps
            params['batch'] = 32
            params['decayRate'] = 0.3
            params['nFilters3D'] = 128 # Need to decrease filters due to OOM error
            params['nFilters2D'] = 128
            params['kernelConv3D'] = (3,3,6) 
            params['strideConv3D'] = (1,1,1)
            params['kernelResBlock3D'] = (3,3,6)
            params['kernelConv2D'] = (3,3)
            params['strideConv2D'] = (1,1)
            params['kernelResBlock2D'] = (3,3)
        elif case == 'Default6Fx_LowRes_LargeFOV':
            params['nF'] = 6
            params['xX'] = 101 # Width of input maps
            params['yY'] = 101 # Height of input maps
            params['batch'] = 32
            params['decayRate'] = 0.3
            params['nFilters3D'] = 128 # Need to decrease filters due to OOM error
            params['nFilters2D'] = 128
            params['kernelConv3D'] = (3,3,6) 
            params['strideConv3D'] = (1,1,1)
            params['kernelResBlock3D'] = (3,3,6)
            params['kernelConv2D'] = (3,3)
            params['strideConv2D'] = (1,1)
            params['kernelResBlock2D'] = (3,3)
        elif case == 'Default6Fx_64x64':
            params['nF'] = 6
            params['xX'] = 64 # Width of input maps
            params['yY'] = 64 # Height of input maps
            params['batch'] = 32
            params['decayRate'] = 0.3
            params['nFilters3D'] = 128 # Need to decrease filters due to OOM error
            params['nFilters2D'] = 128
            params['kernelConv3D'] = (3,3,2) 
            params['strideConv3D'] = (1,1,1)
            params['kernelResBlock3D'] = (3,3,6)
            params['kernelConv2D'] = (3,3)
            params['strideConv2D'] = (1,1)
            params['kernelResBlock2D'] = (3,3)
        return params