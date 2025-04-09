
def Model(self):
    """The deep learning architecture gets defined here"""
    

    ## Input Optical Properties ##
    inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))
    ## Input Multi-Dimensional Fluorescence ##
    inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))

    ## NOTE: Batch normalization can cause instability in the validation loss

    ## Optical Properties Branch ##
    inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                    padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)

    inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                    padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
    
    print("inOP1: ", inOP.shape)
    
    ## Fluorescence Input Branch ##
    input_shape = inFL_beg.shape
    inFL = Conv2D(filters=self.params['nFilters3D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                    padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)
    

    inFL = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                    padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)

    ## Concatenate Branch ##
    concat = concatenate([inOP,inFL],axis=-1)

    Max_Pool_1 = MaxPool2D()(concat)

    Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                    activation=self.params['activation'], data_format="channels_last")(Max_Pool_1)
    Conv_1 = Conv2D(filters=256, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                    activation=self.params['activation'], data_format="channels_last")(Conv_1)
    
    Max_Pool_2 = MaxPool2D()(Conv_1)

    Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                    activation=self.params['activation'], data_format="channels_last")(Max_Pool_2)
    Conv_2 = Conv2D(filters=512, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                    activation=self.params['activation'], data_format="channels_last")(Conv_2)

    tfk = tf.keras
    tfkl = tfk.layers
    tfm = tf.math
    hidden_size = Conv_2.shape[-1]
    n_layers = 12
    n_heads = 16
    mlp_dim = 3072
    dropout = 0.1
    patch_size = 1

    #linear embeddings 

    y = tfkl.Conv2D(
    filters=hidden_size,
    kernel_size=patch_size,
    strides=patch_size,
    padding="valid",
    name="embedding",
    trainable=True
    )(Conv_2)

    #flattening out 
    y = tfkl.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
    
    #create the layers 
    y = encoder_layers.AddPositionEmbs(trainable=True)(y)

    y = tfkl.Dropout(0.1)(y)

    # Transformer/Encoder
    for n in range(n_layers):
        y, _ = encoder_layers.TransformerBlock(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            trainable=True
        )(y)
    y = tfkl.LayerNormalization(
        epsilon=1e-6
    )(y)

    transformed = Flatten()(y)

    #flatten outQF

    Dense1 = Dense(128)(transformed)
    Dense1 = Dropout(0.5)(Dense1)
    Dense2 = Dense(128)(Dense1)
    Dense2 = Dropout(0.5)(Dense2)
    Dense3 = Dense(2)(Dense2)

    outQF = Dense3[:][0]
    outDF = Dense3[:][1]

    ## Defining and compiling the model ##
    self.modelD = Model(inputs=[inOP_beg,inFL_beg], outputs=[outQF, outDF])#,outFL])
    self.modelD.compile(loss=['mae', 'mae'],
                    optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                    metrics=['mae', 'mae'])
    self.modelD.summary()
    return None
