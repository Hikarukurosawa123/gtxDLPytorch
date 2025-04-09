def Model(self):
        """The deep learning architecture gets defined here"""
      

        ## Input Optical Properties ##
        inOP_beg = Input(shape=(self.params['xX'],self.params['yY'],2))
        ## Input Multi-Dimensional Fluorescence ##
        inFL_beg = Input(shape=(self.params['xX'],self.params['yY'],self.params['nF']))

        inQF_beg = Input(shape=(self.params['xX'],self.params['yY'], 1))

        ## NOTE: Batch normalization can cause instability in the validation loss

        ## Optical Properties Branch ##
        inOP = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP_beg)

        inOP = Dropout(0.5)(inOP)
        inOP = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inOP)
        inOP = Dropout(0.5)(inOP)
      
        ## Fluorescence Input Branch ##
        input_shape = inFL_beg.shape
        inFL = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inFL_beg)
        
        inFL = Dropout(0.5)(inFL)

        inFL = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inFL)
        inFL = Dropout(0.5)(inFL)

        #fluorophore concentration branch 
        input_shape = inQF_beg.shape
        inQF = Conv2D(filters=self.params['nFilters2D']//2, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], input_shape=input_shape[1:], data_format="channels_last")(inQF)
        
        inQF = Dropout(0.5)(inQF)

        inQF = Conv2D(filters=int(self.params['nFilters2D']/2), kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], 
                      padding='same', activation=self.params['activation'], data_format="channels_last")(inQF)
        inQF = Dropout(0.5)(inQF)


        ## Concatenate Branch ##
        concat = concatenate([inOP,inFL, inQF],axis=-1)

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
        print("Conv_2: ", Conv_2.shape)

        Max_Pool_3 = MaxPool2D()(Conv_2)


        Conv_3 = Conv2D(filters=1024, kernel_size=(self.params['kernelConv2D']), strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Max_Pool_3)
        Conv_3 = Conv2D(filters=1024, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_3)
        print("Conv_3: ", Conv_3.shape)

        #decoder 

        x = Conv_2[:,0:Conv_2.shape[1] - 1, 0:Conv_2.shape[2] - 1, :]
        s = self.attention_gate(x, Conv_3, 512)

        print("x shape", x.shape)

        print("s shape", s.shape)
        Up_conv_1 = UpSampling2D()(Conv_3)

        
        Up_conv_1 = Conv2D(filters=512, kernel_size = (2,2), strides=(1,1), padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Up_conv_1)

        #attention block 
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

        print("Up_conv shape", Up_conv_3.shape)
        print("s shape", s.shape)
        s = s[:,0:s.shape[1] - 1, 0:s.shape[2] - 1, :]
        concat_2 = concatenate([Up_conv_3,s],axis=-1)

        Conv_6 = Conv2D(filters=128, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(concat_2)

        ## Depth Fluorescence Output Branch ##
        #first DF layer 
        outDF = Conv2D(filters=64, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(Conv_6)
        #outDF = Dropout(outDF)

        outDF = Conv2D(filters=32, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                       activation=self.params['activation'], data_format="channels_last")(outDF)
        #outDF = BatchNormalization()(outDF)
        #outDF = Dropout(outDF)
     
        
        outDF = Conv2D(filters=1, kernel_size=self.params['kernelConv2D'], strides=self.params['strideConv2D'], padding='same', 
                     data_format="channels_last")(outDF)

        ## Defining and compiling the model ##
        self.modelD = Model(inputs=[inOP_beg,inFL_beg, inQF_beg], outputs=[outDF])#,outFL])
        self.modelD.compile(loss=['mae'],
                      optimizer=getattr(keras.optimizers,self.params['optimizer'])(learning_rate=self.params['learningRate']),
                      metrics=['mae', 'mae'])
        self.modelD.summary()
        return None



 def Analysis(self):


        self.import_data_for_testing()

        self.indxIncl = np.nonzero(self.temp_DF_pre_conversion)

        #self.Predict()
        predict = self.modelD.predict([self.OP, self.FL, self.QF], batch_size = 1)  

        DF_P = predict
        DF_P /= self.params['scaleDF']  



        DF_P = np.reshape(DF_P, (DF_P.shape[0], DF_P.shape[1], DF_P.shape[2], 1))

        ## Error Stats
        # Average error
        DF_error = DF_P - self.DF
        DF_erroravg = np.mean(abs(DF_error[self.indxIncl]))
        DF_errorstd = np.std(abs(DF_error[self.indxIncl]))

        print('Average Depth Error (SD): {}({}) mm'.format(float('%.5g' % DF_erroravg),float('%.5g' % DF_errorstd)))
        # Overall  mean squared error
        DF_mse = np.sum((DF_P - self.DF) ** 2)
        DF_mse /= float(DF_P.shape[0] * DF_P.shape[1] * DF_P.shape[2])
        print('Depth Mean Squared Error: {} mm'.format(float('%.5g' % DF_mse)))
        # Max and Min values per sample
        ua_max = []
        us_max = []
        DF_max = []
        DFP_max = []

        DF_min = self.get_min(self.DF)
        DF_P_min = self.get_min(DF_P)

        for i in range(DF_P.shape[0]):
            ua_max.append(self.OP[i,:,:,0].max())
            us_max.append(self.OP[i,:,:,1].max())
            #DF_max.append(self.DF[i,:,:].max())
            #DFP_max.append(DF_P[i,:,:].max())



        DF_max = DF_min
        DFP_max = DF_P_min

        DF_max = np.array(DF_max)
        DFP_max = np.array(DFP_max)

        #compute absolute mindepth error 
        min_depth_error = np.mean(np.abs(DFP_max - DF_max))
        min_depth_error_std = np.std(np.abs(DFP_max - DF_max))
        print("Average Minimum Depth Error (SD) : {min_depth_error} ({min_depth_error_std})".format(min_depth_error = min_depth_error, min_depth_error_std = min_depth_error_std))
        num_predict_zeros = self.count_predictions_of_zero(DFP_max)
        print("number of predictions of zero:", num_predict_zeros)
        # SSIM per sample


        min_depth_graph = plt.figure()
        #plt.scatter(DF_max[beg:end],DFP_max[beg:end],s=1)
        #plt.scatter(DF_max,DFP_max,s=3, label = "Correct Classification", color = ['blue'])
        plt.scatter(DF_max,DFP_max,s=3, label =  "Correct Classification", color = ['blue'])

        DF_max_classify = np.array(DF_max) < 5 
        DFP_max_classify = np.array(DFP_max) < 5

        failed_result = DF_max_classify !=DFP_max_classify

        plt.scatter(DF_max[failed_result],DFP_max[failed_result],label = "Incorrect Classification", s=3, color = ['red'])
        plt.legend(loc="upper left", prop={'size': 13, 'weight':'bold'})

        plt.xlim([0, 10])
        plt.ylim([0, 10])
        plt.plot(plt.xlim([0, 10]), plt.ylim([0, 10]),color='k')
        plt.ylabel("Predicted Depth (mm)")
        plt.xlabel("True Depth (mm)")
        plt.title("Minimum Depth")
        plt.tight_layout()
        font = {'weight': 'bold', 'size':12}
        matplotlib.rc('font', **font)

        num_plot_display = 100
        if self.DF.shape[0] < 10:
            for i in range(self.DF.shape[0]):
                fig, axs = plt.subplots(2,3)
                plt.set_cmap('jet')
                plt.colorbar(axs[0,0].imshow(self.DF[i,:,:],vmin=0,vmax=15), ax=axs[0, 0],fraction=0.046, pad=0.04)


                axs[0,0].axis('off')
                axs[0,0].set_title('True Depth (mm)')
                plt.colorbar(axs[0,1].imshow(DF_P[i,:,:],vmin=0,vmax=15), ax=axs[0, 1],fraction=0.046, pad=0.04)
                axs[0,1].axis('off')
                axs[0,1].set_title('Predicted Depth (mm)')
                plt.colorbar(axs[0,2].imshow(abs(DF_error[i,:,:]),vmin=0,vmax=15), ax=axs[0, 2],fraction=0.046, pad=0.04)
                axs[0,2].axis('off')
                axs[0,2].set_title('|Error (mm)|')

        else:

            for i in range(num_plot_display):#range(num_plot_display):
                fig, axs = plt.subplots(2,3)
                plt.set_cmap('jet')
                plt.colorbar(axs[0,0].imshow(self.DF[i,:,:],vmin=0,vmax=15), ax=axs[0, 0],fraction=0.046, pad=0.04)


                axs[0,0].axis('off')
                axs[0,0].set_title('True Depth (mm)')


                plt.colorbar(axs[0,1].imshow(DF_P[i,:,:],vmin=0,vmax=15), ax=axs[0, 1],fraction=0.046, pad=0.04)
                axs[0,1].axis('off')
                axs[0,1].set_title('Predicted Depth (mm)')
                plt.colorbar(axs[0,2].imshow(abs(DF_error[i,:,:]),vmin=0,vmax=15), ax=axs[0, 2],fraction=0.046, pad=0.04)
                axs[0,2].axis('off')
                axs[0,2].set_title('|Error (mm)|')