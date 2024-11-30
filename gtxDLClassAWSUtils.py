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
        params['optimizer']  = 'Adam' # Options are SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl: I don't assume results will be much different depending on the optimizer, but I have not tested all of these
        params['epochs'] = 100
        params['nF'] = 6

        # Scaling parameters; goal of scaling is to make our diferently measured inputs (i.e., QF, DF, OP, FL) scale-invariant, i.e., the scale, or units, that each of these values is measured in is irrelevant. But, should OP0 and OP1 not have the same scale? And should the scale for DF not be 1/scaleOP? Since both OPs are measured in 1/mm, and depth is measured in mm, what I just said should be true, otherwise the data in each of these will NOT be scale invariant; it will be relatively variant (i.e., if OP0 = 10*OP1, then the data in OP0 will be ten times closer to the data in OP1 than it is in reality: would this not be harmful to our ability to learn?)
        params['scaleFL'] = 10**4
        # Case1: ua = 4.5e-3 (1/mm); Case2&3: ua ~ [1.5e-3 - 1.5e-2] (1/mm)
        params['scaleOP0'] = 10
        # Case1&2: us' = 1 (1/mm); Case3: us' ~ [0.75 - 2] (1/mm)
        params['scaleOP1'] = 1
        # z ~ [1 - 5] (mm)
        params['scaleDF'] = 1
        # Case1: conc = 5 (ug/mL); Case2&3: conc ~ [1 - 10] (ug/mL)
        params['scaleQF'] = 1
        params['scaleRE'] = 1

        if case == 'Default':
            params['learningRate'] = 5e-4 # Initial learning rate, subject to scheduled decay (see DL callbacks)
            params['xX'] = 101 # Width of input maps
            params['yY'] = 101 # Height of input maps
            params['batch'] = 32
            params['decayRate'] = 0.3
            params['nFilters3D'] = 128 # Need to decrease filters due to OOM error (before - 128)
            params['nFilters2D'] = 128
            params['kernelConv3D'] = (3,3, 3) 
            params['strideConv3D'] = (1,1,1)
            params['kernelResBlock3D'] = (3,3,3)
            params['kernelConv2D'] = (3,3)
            params['strideConv2D'] = (1,1)
            params['kernelResBlock2D'] = (3,3)
            
        elif case == 'Default4fx':
            params['nF']=4
            params['learningRate'] = 5e-5 # Initial learning rate, subject to scheduled decay (see DL callbacks)
            params['scaleOP0'] = 100
            params['xX'] = 101 # Width of input maps
            params['yY'] = 101 # Height of input maps
            params['batch'] = 16
            params['decayRate'] = 0.3
            params['nFilters3D'] = 128 # Need to decrease filters due to OOM error
            params['nFilters2D'] = 128
            params['kernelConv3D'] = (3,3,4) 
            params['strideConv3D'] = (1,1,1)
            params['kernelResBlock3D'] = (3,3,4)
            params['kernelConv2D'] = (3,3)
            params['strideConv2D'] = (1,1)
            params['kernelResBlock2D'] = (3,3)
        
       
        return params