def Model(self):
        """The deep learning architecture gets defined here"""
        
        # Input Optical Properties ##
        inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],2))
        ## Input Multi-Dimensional Fluorescence ##
        inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF'], 1))

        ## NOTE: Batch normalization can cause instability in the validation loss

        #3D CNN for all layers

        ## Optical Properties Branch ##
        inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)
        inOP = Dropout(0.5)(inOP)

        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        inOP = Dropout(0.5)(inOP)
        
        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        inOP = Dropout(0.5)(inOP)  
        ## Fluorescence Input Branch ##
        #inFL = Reshape((inFL_beg.shape[1], inFL_beg.shape[2], 1,inFL_beg.shape[3]))(inFL_beg)
        input_shape = inFL_beg.shape
        print(input_shape)

        inFL = Conv3D(filters=self.params['nFilters3D']//2, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)
        inFL = Dropout(0.5)(inFL)
        print(inFL.shape)

        inFL = Conv3D(filters=int(self.params['nFilters3D']/2), kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        inFL = Dropout(0.5)(inFL)
        inFL = Conv3D(filters=int(self.params['nFilters3D']/2), kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        inFL = Dropout(0.5)(inFL)

        ## Concatenate Branch ##
        inFL = Reshape((inFL.shape[1], inFL.shape[2], inFL.shape[3] * inFL.shape[4]))(inFL)
        
        ## Concatenate Branch ##
        concat = concatenate([inOP,inFL],axis=-1)

        Max_Pool_1 = MaxPool2D()(concat)

        Max_Pool_1 = Reshape((Max_Pool_1.shape[1], Max_Pool_1.shape[2], 1, Max_Pool_1.shape[3]))(Max_Pool_1)
        print("Max_Pool_1: ", Max_Pool_1.shape)

        Conv_1 = Conv3D(filters=256, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Max_Pool_1)
        
        print("Conv_1: ", Conv_1.shape)

        Conv_1 = Conv3D(filters=256, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_1)
        Conv_1 = Reshape((Conv_1.shape[1], Conv_1.shape[2], Conv_1.shape[4]))(Conv_1)

        Max_Pool_2 = MaxPool2D()(Conv_1)
        Max_Pool_2 = Reshape((Max_Pool_2.shape[1], Max_Pool_2.shape[2], 1, Max_Pool_2.shape[3]))(Max_Pool_2)

        Conv_2 = Conv3D(filters=512, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Max_Pool_2)
        Conv_2 = Conv3D(filters=512, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_2)
        
        Conv_2 = Reshape((Conv_2.shape[1], Conv_2.shape[2], Conv_2.shape[4]))(Conv_2)

        Max_Pool_3 = MaxPool2D()(Conv_2)
        Max_Pool_3 = Reshape((Max_Pool_3.shape[1], Max_Pool_3.shape[2], 1, Max_Pool_3.shape[3]))(Max_Pool_3)

        print("Max_Pool_3: ", Max_Pool_3.shape)

        Conv_3 = Conv3D(filters=1024, kernel_size=(self.params['kernelConv3D']), strides=self.params['strideConv3D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Max_Pool_3)
        Conv_3 = Conv3D(filters=1024, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_3)
        print("Conv_3: ", Conv_3.shape)
        Conv_3 = Reshape((Conv_3.shape[1], Conv_3.shape[2], Conv_3.shape[4]))(Conv_3)
        
        x = Conv_2[:,0:Conv_2.shape[1] - 1, 0:Conv_2.shape[2] - 1, :]
        s = self.attention_gate(x, Conv_3, 512)

        print("x shape", x.shape)

        print("s shape", s.shape)
        #decoder 
        Up_conv_1 = UpSampling2D()(Conv_3)

        #Up_conv_1 = Reshape((Up_conv_1.shape[1], Up_conv_1.shape[2], Up_conv_1.shape[4]))(Up_conv_1)

        Up_conv_1 = Conv2D(filters=512, kernel_size = (2,2), strides=(1,1), padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Up_conv_1)
         #g is coming from encoder path, s is coming from decoder path 

       

        concat_1 = concatenate([Up_conv_1,s],axis=-1)

        Conv_4 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(concat_1)
 
        Conv_4 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                            activation=self.params['activation'], data_format="channels_last")(Conv_4)
        x = Conv_1
        Conv_4_zero_pad = ZeroPadding2D(padding = ((1,0), (1,0)))(Conv_4)
        s = self.attention_gate(x, Conv_4_zero_pad, 256)

        Up_conv_2 = UpSampling2D()(Conv_4)

        Up_conv_2 = Conv2D(filters=256, kernel_size = (2,2), strides=(1,1), padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Up_conv_2)
    

        Up_conv_2 = ZeroPadding2D()(Up_conv_2)

        

        concat_2 = concatenate([Up_conv_2,s],axis=-1)

        Conv_5 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(concat_2)
        Conv_5 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_5)
        x = ZeroPadding2D(padding = ((1,0), (1,0)))(concat)
        Conv_5_zero_pad = ZeroPadding2D(padding = ((1,0), (1,0)))(Conv_5)

        s = self.attention_gate(x, Conv_5_zero_pad, 128)

        Up_conv_3 = UpSampling2D()(Conv_5)
        Up_conv_3 = Conv2D(filters=128, kernel_size = (2,2), strides=(1,1), padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Up_conv_3)
        
                       
        Up_conv_3 = ZeroPadding2D(padding = ((1,0), (1,0)))(Up_conv_3)

        s = s[:,0:s.shape[1] - 1, 0:s.shape[2] - 1, :]
        concat_2 = concatenate([Up_conv_3,s],axis=-1)

        Conv_6 = Conv2D(filters=128, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(concat_2)

        ## Quantitative Fluorescence Output Branch ##
        outQF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Conv_6)

        outQF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(outQF) #outQF
        
        #outQF = BatchNormalization()(outQF)
        
        outQF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        data_format="channels_last")(outQF)

        ## Depth Fluorescence Output Branch ##
        #first DF layer 
        outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Conv_6)

        outDF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(outDF)

        #outDF = BatchNormalization()(outDF)

        
        outDF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                data_format="channels_last")(outDF)

        ## Defining and compiling the model ##
        self.modelD = Model(inputs=[inOP_beg,inFL_beg], outputs=[outQF, outDF])#,outFL])
        self.modelD.compile(loss=['mae', 'mae'],
                optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                metrics=['mae', 'mae'])
        self.modelD.summary()
        return None
