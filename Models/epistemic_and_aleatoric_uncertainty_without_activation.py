class MonteCarloDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
    

def Model(self):
        """The deep learning architecture gets defined here"""

        drop_out = 0.5

        print("drop_out", drop_out)
        inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],2))
        ## Input Multi-Dimensional Fluorescence ##
        inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))
        #inFL = FlData
        
        ## NOTE: Batch normalization can cause instability in the validation loss

        ## Optical Properties Branch ##
        inOP = Conv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)
        #outOP1 = inOP
        inOP = BatchNormalization()(inOP)  

        inOP = self.drop_out(inOP, drop_out) #drop out 1
        inOP = BatchNormalization()(inOP)

        inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        #outOP2 = inOP

        inOP = BatchNormalization()(inOP)

        inOP = self.drop_out(inOP, drop_out) #drop out 1

        inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        #outOP3 = inOP
        inOP = self.drop_out(inOP, drop_out) #drop out 1

        inOP = self.resblock_2D(self.params['nFilters2D']//2, self.params['kernelResBlock2D'], self.params['strideConv2D'], inOP, drop_out)

        ## Fluorescence Input Branch ##
        input_shape = inFL_beg.shape

        inFL = Conv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], input_shape=input_shape, data_format="channels_last")(inFL_beg)
        
        inFL = BatchNormalization()(inFL)

        inFL = self.drop_out(inFL, drop_out) #drop out 1

        inFL = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        #outFL2 = inFL
        inFL = BatchNormalization()(inFL)

        inFL = self.drop_out(inFL, drop_out) #drop out 1

        inFL = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        #outFL3 = inFL
        inFL = BatchNormalization()(inFL)

        inFL = self.drop_out(inFL, drop_out) #drop out 1

        inFL = self.resblock_2D(self.params['nFilters2D']//2, self.params['kernelResBlock2D'], self.params['strideConv2D'], inFL, drop_out)
        
        ## Concatenate Branch ##
        concat = concatenate([inOP,inFL],axis=-1)
        concat = self.drop_out(concat, drop_out) #drop out 3

        concat = SeparableConv2D(filters=self.params['nFilters2D'], kernel_size=self.params['kernelConv2D'], 
                                 strides=self.params['strideConv2D'], padding='same', activation=self.params['activation'], 
                                 data_format="channels_last")(concat)
        concat = BatchNormalization()(concat)

        concat = self.drop_out(concat, drop_out) #drop out 3

        #concat = BatchNormalization()(concat)
        concat = self.resblock_2D(self.params['nFilters2D'], self.params['kernelResBlock2D'], self.params['strideConv2D'], concat, drop_out) 
        concat = self.drop_out(concat, drop_out) #drop out 3

        ## Quantitative Fluorescence Output Branch ##
        outQF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                        data_format="channels_last")(concat)
        outQF = BatchNormalization()(outQF)

        outQF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       data_format="channels_last")(outQF)
        
        outQF = BatchNormalization()(outQF)
     
        outQF = Conv2D(filters=2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       data_format="channels_last")(outQF)

        ## Depth Fluorescence Output Branch ##
        outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       data_format="channels_last")(concat)
        
        outDF = BatchNormalization()(outDF)
   

        outDF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       data_format="channels_last")(outDF)
        outDF = BatchNormalization()(outDF)

        outDF = Conv2D(filters=2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       data_format="channels_last")(outDF)

    
        self.modelD = Model(inputs=[inOP_beg,inFL_beg], outputs=[outQF, outDF])#,outFL])
        self.modelD.compile(loss=[self.laplacian_loss, self.laplacian_loss],
                      optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                      metrics=[self.mae_on_first_channel, self.mae_on_first_channel])
        self.modelD.summary()
        
        return self.modelD
def laplacian_loss(self, y_true, y_pred):
    mean_true = y_true[:, :, :, 0]
    mean_pred = y_pred[:, :, :, 0]
    log_var = y_pred[:, :, :, 1]
    #loss = tf.abs(tf.math.divide(tf.math.abs(mean_true - mean_pred), scale_pred + 1e-2) + tf.math.log(scale_pred + 1e-2))
    
    loss = tf.reduce_mean( tf.exp(log_var) *tf.square( (mean_pred-mean_true) ) )

    return loss

def mae_on_first_channel(self, y_true, y_pred):
    mean_true = y_true[:, :, :, 0]
    mean_pred = y_pred[:, :, :, 0]
    log_var = y_pred[:, :, :, 1]

    loss = tf.reduce_mean(log_var)
    return loss 
