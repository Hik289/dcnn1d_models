import torch
from torch import nn

class bottleneck_1(nn.Module):
    def __init__(self,input_dim=24, hidden_dim = 24,output_dim = 24):
        super(bottleneck_1,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.cnn1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=1,stride = 1,bias = False)
        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, 
                              kernel_size=3,stride = 1,padding = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm2 = nn.BatchNorm1d(self.hidden_dim)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.output_dim, kernel_size=1,stride = 1,bias =False)
        self.batchnorm3 = nn.BatchNorm1d(self.output_dim)
        self.act3 = nn.ReLU()

        self.cnn4 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.output_dim, kernel_size=1,stride = 1,bias = False)
        self.batchnorm4 = nn.BatchNorm1d(self.output_dim)
        self.act4 = nn.ReLU()        

        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act3(x_signal)

        if self.input_dim == self.output_dim:
            x_temp = self.cnn4(x)
            x_temp = self.batchnorm4(x_temp)
            x_temp = self.act4(x_temp)
            x_signal = x_signal + x_temp
        else:
            x_signal = x_signal

        out = self.act(x_signal)

        return out


class bottleneck_1_cut2(nn.Module):
    def __init__(self,input_dim=24, hidden_dim = 24,output_dim = 24):
        super(bottleneck_1_cut2,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.cnn1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=1,stride = 1,bias = False)
        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, 
                              kernel_size=3,stride = 2,padding = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm2 = nn.BatchNorm1d(self.hidden_dim)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.output_dim, kernel_size=1,stride = 1,bias = False)
        self.batchnorm3 = nn.BatchNorm1d(self.output_dim)
        self.act3 = nn.ReLU()

        self.cnn4 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.output_dim, kernel_size=1,stride = 2,bias = False)
        self.batchnorm4 = nn.BatchNorm1d(self.output_dim)
        self.act4 = nn.ReLU()        

        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act3(x_signal)

        if self.input_dim == self.output_dim:
            x_temp = self.cnn4(x)
            x_temp = self.batchnorm4(x_temp)
            x_temp = self.act4(x_temp)
            x_signal = x_signal + x_temp
        else:
            x_signal = x_signal

        out = self.act(x_signal)

        return out


class bottleneck_2(nn.Module):
    def __init__(self,input_dim=24, hidden_dim = 24,output_dim = 24):
        super(bottleneck_2,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.cnn1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=1,stride = 1,bias = False)
        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, 
                              kernel_size=3,stride = 1,padding = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm2 = nn.BatchNorm1d(self.hidden_dim)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.output_dim, kernel_size=1,stride = 1,bias = False)
        self.batchnorm3 = nn.BatchNorm1d(self.output_dim)
        self.act3 = nn.ReLU()

        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act3(x_signal)

        if self.input_dim == self.output_dim:
            x_signal = x_signal + x
        else:
            x_signal = x_signal

        out = self.act(x_signal)

        return out


class bottleneck_3(nn.Module):
    def __init__(self,input_dim=24, hidden_dim = 24, output_dim = 24):
        super(bottleneck_3,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.cnn1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, 
                              kernel_size=3,stride = 1,padding = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.output_dim,
                              kernel_size=3,stride = 1,padding = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm2 = nn.BatchNorm1d(self.output_dim)
        self.act2 = nn.ReLU()

        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        if self.input_dim == self.output_dim:
            x_signal = x_signal + x
        else:
            x_signal = x_signal

        out = self.act(x_signal)

        return out


class bottleneck_4(nn.Module):
    def __init__(self,input_dim=24, hidden_dim = 24, output_dim = 24):
        super(bottleneck_4,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.cnn1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, 
                              kernel_size=3,stride = 1,padding = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.output_dim,
                              kernel_size=3,stride = 1,padding = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm2 = nn.BatchNorm1d(self.output_dim)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.output_dim,
                              kernel_size=1,stride = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm3 = nn.BatchNorm1d(self.output_dim)
        self.act3 = nn.ReLU()

        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        if self.input_dim == self.output_dim:
            x_temp = self.cnn3(x)
            x_temp = self.batchnorm3(x_temp)
            x_temp = self.act3(x_temp)
            x_signal = x_signal + x_temp
        else:
            x_signal = x_signal

        out = self.act(x_signal)

        return out


class bottleneck_4_cut2(nn.Module):
    def __init__(self,input_dim=24, hidden_dim = 24, output_dim = 24):
        super(bottleneck_4_cut2,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.cnn1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, 
                              kernel_size=3,stride = 2,padding = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.output_dim,
                              kernel_size=3,stride = 1,padding = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm2 = nn.BatchNorm1d(self.output_dim)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.output_dim,
                              kernel_size=1,stride = 2,padding_mode = 'replicate',bias = False)
        self.batchnorm3 = nn.BatchNorm1d(self.output_dim)
        self.act3 = nn.ReLU()

        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        if self.input_dim == self.output_dim:
            x_temp = self.cnn3(x)
            x_temp = self.batchnorm3(x_temp)
            x_temp = self.act3(x_temp)
            x_signal = x_signal + x_temp
        else:
            x_signal = x_signal

        out = self.act(x_signal)

        return out


class bottleneck_5(nn.Module):
    def __init__(self,input_dim=24, hidden_dim = 24,output_dim = 24):
        super(bottleneck_5,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.cnn1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=1,stride = 1,bias = False)
        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, groups = hidden_dim//4,
                              kernel_size=3,stride = 1,padding = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm2 = nn.BatchNorm1d(self.hidden_dim)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.output_dim, kernel_size=1,stride = 1,bias = False)
        self.batchnorm3 = nn.BatchNorm1d(self.output_dim)
        self.act3 = nn.ReLU()

        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act3(x_signal)

        if self.input_dim == self.output_dim:
            x_signal = x_signal + x
        else:
            x_signal = x_signal

        out = self.act(x_signal)

        return out


class bottleneck_6(nn.Module):
    def __init__(self,input_dim=24, hidden_dim = 24, output_dim = 24):
        super(bottleneck_6,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.cnn1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, groups = hidden_dim//4,
                              kernel_size=3,stride = 1,padding = 1,padding_mode = 'replicate',bias = False)
        self.batchnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.output_dim, groups = hidden_dim//4,
                              kernel_size=3,stride = 1,padding = 1,padding_mode = 'replicate', bias = False)
        self.batchnorm2 = nn.BatchNorm1d(self.output_dim)
        self.act2 = nn.ReLU()

        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        if self.input_dim == self.output_dim:
            x_signal = x_signal + x
        else:
            x_signal = x_signal

        out = self.act(x_signal)

        return out























