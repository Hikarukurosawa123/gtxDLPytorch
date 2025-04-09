class Model():
    def drop_out(self, x, drop_out = None):
        if drop_out: 
            x = Dropout(drop_out)(x, training = True)
    
        return x 
    def attention_gate(self, g, s, num_filters):
        print("g in shape: ", g.shape)
        Wg = Conv2D(num_filters, 1, strides = (2,2), padding="valid")(g)
        print("Wg shape: ", Wg.shape)
        #Wg = BatchNormalization()(Wg)
        print("s in shape: ", s.shape)

        Ws = Conv2D(num_filters, 1, padding="same")(s)
        #Ws = BatchNormalization()(Ws)
        print("Ws shape: ", Ws.shape)

        out = Activation("relu")(Wg + Ws)
        out = Conv2D(1, 1, padding="same")(out)
        out = Activation("sigmoid")(out)
        out = UpSampling2D()(out)
        print("out shape: ", out.shape)
        print("g shape: ", g.shape)

        return out * g
    
    def Model(self):
        """The deep learning architecture gets defined here"""

        # Input Optical Properties ##
        inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],2))
        ## Input Multi-Dimensional Fluorescence ##
        inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF'], 1))
        
        ## Optical Properties Branch ##
        inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)
        inOP = Dropout(0.5)(inOP)
        
        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        inOP = Dropout(0.5)(inOP)  

        ## Fluorescence Input Branch ##
        #inFL = Reshape((inFL_beg.shape[1], inFL_beg.shape[2], 1,inFL_beg.shape[3]))(inFL_beg)
        input_shape = inFL_beg.shape

        inFL = Conv3D(filters=self.params['nFilters3D']//2, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)
        
        inFL = Dropout(0.5)(inFL)

        inFL = Conv3D(filters=int(self.params['nFilters3D']/2), kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], 
                padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        
        inFL = Dropout(0.5)(inFL)
        inFL = Reshape((inFL.shape[1], inFL.shape[2], inFL.shape[3] * inFL.shape[4]))(inFL)

        ## Concatenate Branch ##
        concat = concatenate([inOP,inFL],axis=-1)

        Max_Pool_1 = MaxPool2D()(concat)

        Max_Pool_1 = Reshape((Max_Pool_1.shape[1], Max_Pool_1.shape[2], 1, Max_Pool_1.shape[3]))(Max_Pool_1)

        Conv_1 = Conv3D(filters=256, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Max_Pool_1)
        
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

        Conv_3 = Conv3D(filters=1024, kernel_size=(self.params['kernelConv3D']), strides=self.params['strideConv3D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Max_Pool_3)
        #Conv_3 = Conv3D(filters=1024, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                #activation=self.params['activation'], data_format="channels_last")(Conv_3)
        Conv_3 = Reshape((Conv_3.shape[1], Conv_3.shape[2], Conv_3.shape[4]))(Conv_3)
        
        x = Conv_2[:,0:Conv_2.shape[1] - 1, 0:Conv_2.shape[2] - 1, :]

        print("Conv_3 shape: ", Conv_3)

        #decoder 
        Up_conv_1 = UpSampling2D()(Conv_3)

        #Up_conv_1 = Reshape((Up_conv_1.shape[1], Up_conv_1.shape[2], Up_conv_1.shape[4]))(Up_conv_1)

        #Up_conv_1 = Conv2D(filters=512, kernel_size = (2,2), strides=(1,1), padding='same', 
        #               activation=self.params['activation'], data_format="channels_last")(Up_conv_1)
        #g is coming from encoder path, s is coming from decoder path 

        concat_1 = concatenate([Up_conv_1,x],axis=-1)

        concat_1 = Reshape((concat_1.shape[1], concat_1.shape[2], 1, concat_1.shape[3]))(concat_1)

        Conv_4 = Conv3D(filters=512, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(concat_1)

        Conv_4 = Conv3D(filters=512, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                        activation=self.params['activation'], data_format="channels_last")(Conv_4)
        x = Conv_1
        Conv_4 = Reshape((Conv_4.shape[1], Conv_4.shape[2], Conv_4.shape[4]))(Conv_4)

        Up_conv_2 = UpSampling2D()(Conv_4)

        Up_conv_2 = ZeroPadding2D()(Up_conv_2)

        concat_2 = concatenate([Up_conv_2,x],axis=-1)

        concat_2 = Reshape((concat_2.shape[1], concat_2.shape[2], 1, concat_2.shape[3]))(concat_2)

        Conv_5 = Conv3D(filters=256, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(concat_2)
        Conv_5 = Conv3D(filters=256, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Conv_5)
        
        Conv_5 = Reshape((Conv_5.shape[1], Conv_5.shape[2], Conv_5.shape[4]))(Conv_5)

        #x = ZeroPadding2D(padding = ((1,0), (1,0)))(concat)

        x = concat
        Up_conv_3 = UpSampling2D()(Conv_5)
        
        Up_conv_3 = ZeroPadding2D(padding = ((1,0), (1,0)))(Up_conv_3)

        concat_2 = concatenate([Up_conv_3,x],axis=-1)    

        concat_2 = Reshape((concat_2.shape[1], concat_2.shape[2], 1,concat_2.shape[3]))(concat_2)


        Conv_6 = Conv3D(filters=128, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(concat_2)

        ## Quantitative Fluorescence Output Branch ##
        outQF = Conv3D(filters=64, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Conv_6)

        outQF = Conv3D(filters=32, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(outQF) #outQF
        
        #outQF = BatchNormalization()(outQF)
        
        outQF = Conv3D(filters=1, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                        data_format="channels_last")(outQF)
        outQF = Reshape((outQF.shape[1], outQF.shape[2], outQF.shape[4]))(outQF)

        ## Depth Fluorescence Output Branch ##
        #first DF layer 
        outDF = Conv3D(filters=64, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(Conv_6)

        outDF = Conv3D(filters=32, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                activation=self.params['activation'], data_format="channels_last")(outDF)

        #outDF = BatchNormalization()(outDF)
        
        outDF = Conv3D(filters=1, kernel_size=self.params['kernelConv3D'], strides=self.params['strideConv3D'], padding='same', 
                data_format="channels_last")(outDF)
        outDF = Reshape((outDF.shape[1], outDF.shape[2], outDF.shape[4]))(outDF)

        ## Defining and compiling the model ##
        self.modelD = Model(inputs=[inOP_beg,inFL_beg], outputs=[outQF, outDF])#,outFL])
        self.modelD.compile(loss=['mae', 'mae'],
                optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                metrics=['mae', 'mae'])
        self.modelD.summary()
        return None