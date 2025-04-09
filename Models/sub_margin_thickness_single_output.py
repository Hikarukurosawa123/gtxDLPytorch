def Model(self):
    

    ## Input Optical Properties ##
    inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],2))
    ## Input Multi-Dimensional Fluorescence ##
    inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))

    ## NOTE: Batch normalization can cause instability in the validation loss

    ## Optical Properties Branch ##
    inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                    padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)

    #inOP = Dropout(0.5)(inOP)
    inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                    padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
    #inOP = Dropout(0.5)(inOP)

    ## Fluorescence Input Branch ##
    input_shape = inFL_beg.shape
    inFL = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                    padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)

    #inFL = Dropout(0.5)(inFL)

    inFL = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                    padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
    #inFL = Dropout(0.5)(inFL)

    ## Concatenate Branch ##
    concat = concatenate([inOP,inFL],axis=-1)
    print("pre Max_Pool_1: ", Max_Pool_1.shape)

    Max_Pool_1 = MaxPool3D()(concat)
    print("Max_Pool_1: ", Max_Pool_1.shape)

    Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                    activation=self.params['activation'], data_format="channels_last")(Max_Pool_1)
    Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                    activation=self.params['activation'], data_format="channels_last")(Conv_1)


    Max_Pool_2 = MaxPool3D()(Conv_1)

    Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                    activation=self.params['activation'], data_format="channels_last")(Max_Pool_2)
    Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                    activation=self.params['activation'], data_format="channels_last")(Conv_2)
    print("Conv_2: ", Conv_2.shape)

    Max_Pool_3 = MaxPool3D()(Conv_2)


    Conv_3 = Conv2D(filters=1024, kernel_size=(self.params['kernelConv2D']), strides=self.params['strideConv2D'], padding='same', 
                    activation=self.params['activation'], data_format="channels_last")(Max_Pool_3)
    Conv_3 = Conv2D(filters=1024, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                    activation=self.params['activation'], data_format="channels_last")(Conv_3)

    #flatten outDF 

    print("Conv_3", Conv_3.shape)
    Conv_3 = Flatten()(Conv_3)
    print("outDF", outDF.shape)
    outDF = Dense(4096)(Conv_3)
    outDF = Dropout(0.5)(outDF)

    outDF = Dense(4096)(outDF)
    outDF = Dropout(0.5)(outDF)

    outDF = Dense(1)(outDF)

    #flatten outQF
    outQF = Dense(4096)(Conv_3)
    outQF = Dropout(0.5)(outQF)

    outQF = Dense(4096)(outDF)
    outQF = Dropout(0.5)(outQF)
    outQF = Dense(1)(outDF)



    ## Defining and compiling the model ##
    self.modelD = Model(inputs=[inOP_beg,inFL_beg], outputs=[outQF, outDF])#,outFL])
    self.modelD.compile(loss=['mae', 'mae'],
                    optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                    metrics=['mae', 'mae'])
    self.modelD.summary()
    return None