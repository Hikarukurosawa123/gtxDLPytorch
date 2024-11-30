def Model(self):
        """The deep learning architecture gets defined here"""
        ## Input ##
        inputData = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))
        inRE = inputData
        
        ## Reflectance Input Branch ##
        input_shape = inRE.shape
        inRE = Conv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inRE)
        outRE1 = inRE
        
        inRE = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inRE)
        outRE2 = inRE

        inRE = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inRE)
        outRE3 = inRE
        
        inRE = self.resblock_2D(self.params['nFilters2D']//2, self.params['kernelResBlock2D'], self.params['strideConv2D'], inRE)

        ## Reshape ##
        #zReshape = int(((self.params['nFilters3D']//2)*self.params['nF'])/self.params['strideConv3D'][2])
        #inRE = Reshape((self.params['xX'],self.params['yY'],zReshape))(inRE)

        ## Concatenate Branch ##
        concat = SeparableConv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], 
                                 strides=self.params['strideConv2D'], padding='same', activation=self.params['activation'], 
                                 data_format="channels_last")(inRE)

        concat = self.resblock_2D(self.params['nFilters2D'], self.params['kernelResBlock2D'], self.params['strideConv2D'], concat) 

        ## Depth Fluorescence Output Branch ##
        outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(concat)
        
        outDF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(outDF)
       
        outDF = BatchNormalization()(outDF)
        outDF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(outDF)

        
        ## Defining and compiling the model ##
        self.modelD = Model(inputs=[inputData], outputs=[outDF])
        self.modelD.compile(loss='mse',
                      optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                      metrics=['mae'])
        self.modelD.summary()
        ## Outputs for feature maps ##
        self.modelFM = Model(inputs=[inputData], outputs=[inputData, outRE1, outRE2, outRE3]) 
        return None