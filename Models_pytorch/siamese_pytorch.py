import torch
import torch.functional as F
import torch.nn as nn 
#implement siamese architecture 
class TinyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv_fluorescence_1 = nn.Conv3d(1, 64, kernel_size=3, padding='same')
        self.conv_fluorescence_2 = nn.Conv3d(64, 64, kernel_size=3, padding='same')
        self.conv_fluorescence_3 = nn.Conv3d(64, 64, kernel_size=3, padding='same')
        

        self.conv_op_1 = nn.Conv2d(2, 64, kernel_size=3, padding='same')
        self.conv_op_2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.conv_op_3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')

        #intermediate branch 
        self.intermediate1 = None#nn.Conv2d(448, 128, kernel_size=3, padding='same')
        self.intermediate2 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.intermediate3 = nn.Conv2d(128, 128, kernel_size=3, padding='same')

        #resnet 

        #output branch 

        self.depth_branch1 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        self.depth_branch2 = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        self.depth_branch3 = nn.Conv2d(32, 1, kernel_size=3, padding='same')

        self.concentration_branch1 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        self.concentration_branch2 = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        self.concentration_branch3 = nn.Conv2d(32, 1, kernel_size=3, padding='same')

    def fluorescence_branch(self,x):
        #define input pytorch tensor 
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]))

        #N, Cout, Dout, Hout, Wout 
        x = self.conv_fluorescence_1(x)
        x = self.conv_fluorescence_2(x)
        x = self.conv_fluorescence_3(x)

        #reshape 
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        return x 

    def op_branch(self,x):
        #define input pytorch tensor 
        #N, Cout, Dout, Hout, Wout 
        x = self.conv_op_1(x)
        print(x.shape)
        x = self.conv_op_2(x)
        print(x.shape)

        x = self.conv_op_3(x)
        print(x.shape)

        return x 

    def intermediate(self,x,y):
        self.concat = torch.concat((x,y),1)

        concat_shape =  self.concat.shape[1] 


        x = nn.Conv2d(concat_shape, 128, kernel_size=3, padding='same')(self.concat)
        
        #resnet 1
        temp = x
        x = self.intermediate2(x)
        x = self.intermediate3(x)

        x = torch.add(x, temp)

        #resnet 2
        temp = x
        x = self.intermediate2(x)
        x = self.intermediate3(x)

        x = torch.add(x, temp)

        return x 
    
    def output(self,intermediate):

        x = self.concentration_branch1(intermediate)
        x = self.concentration_branch2(x)
        x = self.concentration_branch3(x)

        y = self.depth_branch1(intermediate)
        y = self.depth_branch2(y)
        y = self.depth_branch3(y)

        return x,y
    def forward(self,x,y):

        x = self.op_branch(x)
        y = self.fluorescence_branch(y)

        post_intermediate = self.intermediate(x,y)
        QF, DF = self.output(post_intermediate)
        return QF, DF
    