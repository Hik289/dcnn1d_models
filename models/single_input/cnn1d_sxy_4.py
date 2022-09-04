from ctypes import Structure
from models.BasicModule import BasicModule
import torch.nn as nn
import torch
from .cnn_model_sxy.bottleneck1d import bottleneck_1,bottleneck_1_cut2,bottleneck_2,bottleneck_3,bottleneck_4,\
                                        bottleneck_4_cut2,bottleneck_5,bottleneck_6


class cnn1d_sxy_4_1(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_1,self).__init__()
        self.model_name = 'cnn1d_sxy_4_1: resnet'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)

        x_signal = x_signal + shortcut

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_2,self).__init__()
        self.model_name = 'cnn1d_sxy_4_2: resnet2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_3(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_3,self).__init__()
        self.model_name = 'cnn1d_sxy_4_3: resnet3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_4(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_4,self).__init__()
        self.model_name = 'cnn1d_sxy_4_4: resnet3*2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)        
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_5(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_5,self).__init__()
        self.model_name = 'cnn1d_sxy_4_5: resnet3*3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_6(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_6,self).__init__()
        self.model_name = 'cnn1d_sxy_4_6: resnet3*4'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_7(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_7,self).__init__()
        self.model_name = 'cnn1d_sxy_4_7: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_8(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_8,self).__init__()
        self.model_name = 'cnn1d_sxy_4_8: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        # x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_9(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_9,self).__init__()
        self.model_name = 'cnn1d_sxy_4_9: ordinary_3_ordinary_ordinary1_allpointwise32'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size= 5, stride = 1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 512, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)
    
        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)
        x_signal = self.avgpool(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_10(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_10,self).__init__()
        self.model_name = 'cnn1d_sxy_4_10: best_edition2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()
        self.batchnorm5 = nn.BatchNorm1d(512)

        self.avgpool = nn.AvgPool1d(kernel_size= 5, stride = 1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 512, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)
        x_singal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_singal = self.batchnorm2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_singal = self.batchnorm3(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)
        x_singal = self.batchnorm4(x_signal)
    
        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)
        x_singal = self.batchnorm5(x_signal)
        x_signal = self.avgpool(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        # x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_11(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_11,self).__init__()
        self.model_name = 'cnn1d_sxy_4_11: best_edition1'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size= 5, stride = 1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 1024, out_features= 128)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)
    
        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)
        x_signal = self.avgpool(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_12(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_12,self).__init__()
        self.model_name = 'cnn1d_sxy_4_12: best_edition3'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()
        self.batchnorm5 = nn.BatchNorm1d(1024)

        self.avgpool = nn.AvgPool1d(kernel_size= 5, stride = 1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 1024, out_features= 128)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)
        x_singal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_singal = self.batchnorm2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_singal = self.batchnorm3(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)
        x_singal = self.batchnorm4(x_signal)
    
        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)
        x_singal = self.batchnorm5(x_signal)
        x_signal = self.avgpool(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        # x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_13(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_13,self).__init__()
        self.model_name = 'cnn1d_sxy_4_13: best_edition2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()
        self.batchnorm5 = nn.BatchNorm1d(512)

        self.avgpool = nn.AvgPool1d(kernel_size= 5, stride = 1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 512, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)
        x_signal = self.batchnorm4(x_signal)
    
        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.avgpool(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        # x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_14(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_14,self).__init__()
        self.model_name = 'cnn1d_sxy_4_14: best_edition3'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()
        self.batchnorm5 = nn.BatchNorm1d(1024)

        self.avgpool = nn.AvgPool1d(kernel_size= 5, stride = 1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 1024, out_features= 128)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)
        x_signal = self.batchnorm4(x_signal)
    
        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.avgpool(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        # x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)
        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_15(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_15,self).__init__()
        self.model_name = 'cnn1d_sxy_4_15: resnet3*3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1,stride = 1)
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1)
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1,stride = 1)
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1)
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1, stride = 1)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1)
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1)
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride = 1)
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1)
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1)
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1)
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn2(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        # x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)
        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_16(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_16,self).__init__()
        self.model_name = 'cnn1d_sxy_4_16: resnet3*3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn2(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        # x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        # x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_17(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_17,self).__init__()
        self.model_name = 'cnn1d_sxy_4_17: ordinary_3_ordinary_ordinary1_allpointwise32'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride = 1,padding = 2,count_include_pad = False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size= 5, stride = 1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 512, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        # torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)
    
        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)
        x_signal = self.avgpool(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_18(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_18,self).__init__()
        self.model_name = 'cnn1d_sxy_4_18: resnet'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 7, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)

        x_signal = x_signal + shortcut

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_19(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_19,self).__init__()
        self.model_name = 'cnn1d_sxy_4_19: resnet2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_20(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_20,self).__init__()
        self.model_name = 'cnn1d_sxy_4_20: resnet3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_21(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_21,self).__init__()
        self.model_name = 'cnn1d_sxy_4_21: resnet3*2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)        
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        #self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_22(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_22,self).__init__()
        self.model_name = 'cnn1d_sxy_4_22: resnet3*3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_23(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_23,self).__init__()
        self.model_name = 'cnn1d_sxy_4_23: resnet3*4'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_24(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_24,self).__init__()
        self.model_name = 'cnn1d_sxy_4_24: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_25(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_25,self).__init__()
        self.model_name = 'cnn1d_sxy_4_25: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_26(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_26,self).__init__()
        self.model_name = 'cnn1d_sxy_4_26: resnet'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 7, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)

        x_signal = x_signal + shortcut

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_27(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_27,self).__init__()
        self.model_name = 'cnn1d_sxy_4_27: resnet2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_28(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_28,self).__init__()
        self.model_name = 'cnn1d_sxy_4_28: resnet2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_29(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_29,self).__init__()
        self.model_name = 'cnn1d_sxy_4_29: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_30(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_30,self).__init__()
        self.model_name = 'cnn1d_sxy_4_30: resnet3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_31(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_31,self).__init__()
        self.model_name = 'cnn1d_sxy_4_31: resnet3*2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)        
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        #self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_32(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_32,self).__init__()
        self.model_name = 'cnn1d_sxy_4_32: resnet3*3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_33(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_33,self).__init__()
        self.model_name = 'cnn1d_sxy_4_33: resnet3*4'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_34(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_34,self).__init__()
        self.model_name = 'cnn1d_sxy_4_34: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_35(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_35,self).__init__()
        self.model_name = 'cnn1d_sxy_4_35: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_36(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_36,self).__init__()
        self.model_name = 'cnn1d_sxy_4_36: resnet'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 7, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)

        x_signal = x_signal + shortcut

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_37(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_37,self).__init__()
        self.model_name = 'cnn1d_sxy_4_37: resnet2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_38(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_38,self).__init__()
        self.model_name = 'cnn1d_sxy_4_38: resnet2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_39(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_39,self).__init__()
        self.model_name = 'cnn1d_sxy_4_39: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)
        
        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_40(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_40,self).__init__()
        self.model_name = 'cnn1d_sxy_4_40: resnet3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_41(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_41,self).__init__()
        self.model_name = 'cnn1d_sxy_4_41: resnet3*2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)        
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        #self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_42(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_42,self).__init__()
        self.model_name = 'cnn1d_sxy_4_42: resnet3*3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_43(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_43,self).__init__()
        self.model_name = 'cnn1d_sxy_4_43: resnet3*4'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_44(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_44,self).__init__()
        self.model_name = 'cnn1d_sxy_4_44: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_45(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_45,self).__init__()
        self.model_name = 'cnn1d_sxy_4_45: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn20 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_46(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_46,self).__init__()
        self.model_name = 'cnn1d_sxy_4_46: resnet'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 7, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_47(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_47,self).__init__()
        self.model_name = 'cnn1d_sxy_4_47: resnet2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.cnn_3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.cnn_5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn_3(x_signal)
        x_signal = self.batchnorm_3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn_5(x_signal)
        x_signal = self.batchnorm_5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_48(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_48,self).__init__()
        self.model_name = 'cnn1d_sxy_4_48: resnet2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.cnn_3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.cnn_5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn_3(x_signal)
        x_signal = self.batchnorm_3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn_5(x_signal)
        x_signal = self.batchnorm_5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_49(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_49,self).__init__()
        self.model_name = 'cnn1d_sxy_4_49: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn_15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_15 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)

        self.cnn_17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_17 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)

        self.cnn_19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_19 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)

        self.cnn_21 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_21 = nn.BatchNorm1d(128)
        self.cnn20 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)

        self.cnn_23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_23 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)

        self.cnn_25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_25 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)

        self.cnn_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_3 = nn.BatchNorm1d(256)
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.cnn_5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_5 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.cnn_7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_7 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn_27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_27 = nn.BatchNorm1d(256)
        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)

        self.cnn_29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_29 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)

        self.cnn_31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_31 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)

        self.cnn_9 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_9 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)

        self.cnn_11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_11 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)

        self.cnn_13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_13 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn_15(x_signal)
        x_signal = self.batchnorm_15(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_17(x_signal)
        x_signal = self.batchnorm_17(x_signal)
        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_19(x_signal)
        x_signal = self.batchnorm_19(x_signal)
        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn_21(x_signal)
        x_signal = self.batchnorm_21(x_signal)
        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_23(x_signal)
        x_signal = self.batchnorm_23(x_signal)
        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_25(x_signal)
        x_signal = self.batchnorm_25(x_signal)
        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_3(x_signal)
        x_signal = self.batchnorm_3(x_signal)
        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_5(x_signal)
        x_signal = self.batchnorm_5(x_signal)
        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_7(x_signal)
        x_signal = self.batchnorm_7(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_27(x_signal)
        x_signal = self.batchnorm_27(x_signal)
        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_29(x_signal)
        x_signal = self.batchnorm_29(x_signal)
        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_31(x_signal)
        x_signal = self.batchnorm_31(x_signal)
        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn_9(x_signal)
        x_signal = self.batchnorm_9(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_11(x_signal)
        x_signal = self.batchnorm_11(x_signal)
        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_13(x_signal)
        x_signal = self.batchnorm_13(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)
        
        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_50(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_50,self).__init__()
        self.model_name = 'cnn1d_sxy_4_50: resnet3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.cnn_3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_3 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.cnn_5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_5 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)

        self.cnn_7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_7 = nn.BatchNorm1d(256)

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn_3(x_signal)
        x_signal = self.batchnorm_3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn_5(x_signal)
        x_signal = self.batchnorm_5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = self.cnn_7(x_signal)
        x_signal = self.batchnorm_7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_51(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_51,self).__init__()
        self.model_name = 'cnn1d_sxy_4_51: resnet3*2'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)        
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn_3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_3 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn_5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_5 = nn.BatchNorm1d(256)

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)
        self.cnn_7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_7 = nn.BatchNorm1d(256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn8 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn_9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm_9 = nn.BatchNorm1d(512)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn_11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_11 = nn.BatchNorm1d(512)

        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)
        self.cnn_13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_13 = nn.BatchNorm1d(512)

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        #self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn_3(x_signal)
        x_signal = self.batchnorm_3(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn_5(x_signal)
        x_signal = self.batchnorm_5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = self.cnn_7(x_signal)
        x_signal = self.batchnorm_7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn_9(x_signal)
        x_signal = self.batchnorm_9(x_signal)
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn_11(x_signal)
        x_signal = self.batchnorm_11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = self.cnn_13(x_signal)
        x_signal = self.batchnorm_13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_52(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_52,self).__init__()
        self.model_name = 'cnn1d_sxy_4_52: resnet3*3'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn_15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_15 = nn.BatchNorm1d(64)

        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn_17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_17 = nn.BatchNorm1d(64)

        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn_19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_19 = nn.BatchNorm1d(64)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn_3 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_3 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn_5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_5 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn_7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_7 = nn.BatchNorm1d(256)

        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn_9 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_9 = nn.BatchNorm1d(512)

        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn_11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_11 = nn.BatchNorm1d(512)

        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn_13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_15(x_signal)
        x_signal = self.batchnorm_15(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_17(x_signal)
        x_signal = self.batchnorm_17(x_signal)
        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn_19(x_signal)
        x_signal = self.batchnorm_19(x_signal)
        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn_3(x_signal)
        x_signal = self.batchnorm_3(x_signal)
        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_5(x_signal)
        x_signal = self.batchnorm_5(x_signal)
        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_7(x_signal)
        x_signal = self.batchnorm_7(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut

        x_signal = self.cnn_9(x_signal)
        x_signal = self.batchnorm_9(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_11(x_signal)
        x_signal = self.batchnorm_11(x_signal)
        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn_13(x_signal)
        x_signal = self.batchnorm_13(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_53(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_53,self).__init__()
        self.model_name = 'cnn1d_sxy_4_53: resnet3*4'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn_15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_15 = nn.BatchNorm1d(64)

        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn_17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_17 = nn.BatchNorm1d(64)

        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn_19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_19 = nn.BatchNorm1d(64)

        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn20 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn_21 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_21 = nn.BatchNorm1d(128)

        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn_23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_23 = nn.BatchNorm1d(128)

        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn_25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_25 = nn.BatchNorm1d(128)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_3 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn_5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_5 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn_7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_7 = nn.BatchNorm1d(256)

        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn_9 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_9 = nn.BatchNorm1d(512)

        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn_11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_11 = nn.BatchNorm1d(512)

        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn_13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn_15(x_signal)
        x_signal = self.batchnorm_15(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_17(x_signal)
        x_signal = self.batchnorm_17(x_signal)
        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_19(x_signal)
        x_signal = self.batchnorm_19(x_signal)
        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn_21(x_signal)
        x_signal = self.batchnorm_21(x_signal)
        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_23(x_signal)
        x_signal = self.batchnorm_23(x_signal)
        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_25(x_signal)
        x_signal = self.batchnorm_25(x_signal)
        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_3(x_signal)
        x_signal = self.batchnorm_3(x_signal)
        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_5(x_signal)
        x_signal = self.batchnorm_5(x_signal)
        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_7(x_signal)
        x_signal = self.batchnorm_7(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn_9(x_signal)
        x_signal = self.batchnorm_9(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_11(x_signal)
        x_signal = self.batchnorm_11(x_signal)
        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_13(x_signal)
        x_signal = self.batchnorm_13(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_54(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_54,self).__init__()
        self.model_name = 'cnn1d_sxy_4_54: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn_15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_15 = nn.BatchNorm1d(64)

        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn_17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_17 = nn.BatchNorm1d(64)

        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn_19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_19 = nn.BatchNorm1d(64)

        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn20 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn_21 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_21 = nn.BatchNorm1d(128)

        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn_23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_23 = nn.BatchNorm1d(128)

        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn_25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_25 = nn.BatchNorm1d(128)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_3 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn_5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_5 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn_7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_7 = nn.BatchNorm1d(256)

        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn_27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_27 = nn.BatchNorm1d(256)

        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn_29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_29 = nn.BatchNorm1d(256)

        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn_31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_31 = nn.BatchNorm1d(256)

        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn_9 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_9 = nn.BatchNorm1d(512)

        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn_11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_11 = nn.BatchNorm1d(512)

        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn_13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn_15(x_signal)
        x_signal = self.batchnorm_15(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_17(x_signal)
        x_signal = self.batchnorm_17(x_signal)
        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_19(x_signal)
        x_signal = self.batchnorm_19(x_signal)
        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn_21(x_signal)
        x_signal = self.batchnorm_21(x_signal)
        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_23(x_signal)
        x_signal = self.batchnorm_23(x_signal)
        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_25(x_signal)
        x_signal = self.batchnorm_25(x_signal)
        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_3(x_signal)
        x_signal = self.batchnorm_3(x_signal)
        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_5(x_signal)
        x_signal = self.batchnorm_5(x_signal)
        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_7(x_signal)
        x_signal = self.batchnorm_7(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn_27(x_signal)
        x_signal = self.batchnorm_27(x_signal)
        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_29(x_signal)
        x_signal = self.batchnorm_29(x_signal)
        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_31(x_signal)
        x_signal = self.batchnorm_31(x_signal)
        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_9(x_signal)
        x_signal = self.batchnorm_9(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_11(x_signal)
        x_signal = self.batchnorm_11(x_signal)
        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_13(x_signal)
        x_signal = self.batchnorm_13(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_55(BasicModule):
    def __init__(self,input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_55,self).__init__()
        self.model_name = 'cnn1d_sxy_4_55: resnet3*5:RESNET'

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.cnn15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm15 = nn.BatchNorm1d(64)
        self.cnn14 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm14 = nn.BatchNorm1d(64)
        self.cnn_15 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_15 = nn.BatchNorm1d(64)

        self.cnn17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm17 = nn.BatchNorm1d(64)
        self.cnn16 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm16 = nn.BatchNorm1d(64)
        self.cnn_17 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_17 = nn.BatchNorm1d(64)

        self.cnn19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm19 = nn.BatchNorm1d(64)
        self.cnn18 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm18 = nn.BatchNorm1d(64)
        self.cnn_19 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_19 = nn.BatchNorm1d(64)

        self.cnn21 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm21 = nn.BatchNorm1d(128)
        self.cnn20 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm20 = nn.BatchNorm1d(128)
        self.cnn_21 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_21 = nn.BatchNorm1d(128)

        self.cnn23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm23 = nn.BatchNorm1d(128)
        self.cnn22 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm22 = nn.BatchNorm1d(128)
        self.cnn_23 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_23 = nn.BatchNorm1d(128)

        self.cnn25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm25 = nn.BatchNorm1d(128)
        self.cnn24 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm24 = nn.BatchNorm1d(128)
        self.cnn_25 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_25 = nn.BatchNorm1d(128)

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_3 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn_5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_5 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.cnn_7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_7 = nn.BatchNorm1d(256)

        self.cnn27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm27 = nn.BatchNorm1d(256)
        self.cnn26 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm26 = nn.BatchNorm1d(256)
        self.cnn_27 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_27 = nn.BatchNorm1d(256)

        self.cnn29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm29 = nn.BatchNorm1d(256)
        self.cnn28 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm28 = nn.BatchNorm1d(256)
        self.cnn_29 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_29 = nn.BatchNorm1d(256)

        self.cnn31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm31 = nn.BatchNorm1d(256)
        self.cnn30 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm30 = nn.BatchNorm1d(256)
        self.cnn_31 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_31 = nn.BatchNorm1d(256)

        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(512)
        self.cnn_9 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_9 = nn.BatchNorm1d(512)

        self.cnn11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(512)
        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(512)
        self.cnn_11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_11 = nn.BatchNorm1d(512)

        self.cnn13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(512)
        self.cnn12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(512)
        self.cnn_13= nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding_mode='replicate')
        self.batchnorm_13 = nn.BatchNorm1d(512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        shortcut = x_signal

        x_signal = self.cnn_15(x_signal)
        x_signal = self.batchnorm_15(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm14(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm15(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_17(x_signal)
        x_signal = self.batchnorm_17(x_signal)
        x_signal = self.cnn16(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm16(x_signal)
        x_signal = self.cnn17(x_signal)
        x_signal = self.batchnorm17(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_19(x_signal)
        x_signal = self.batchnorm_19(x_signal)
        x_signal = self.cnn18(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm18(x_signal)
        x_signal = self.cnn19(x_signal)
        x_signal = self.batchnorm19(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn_21(x_signal)
        x_signal = self.batchnorm_21(x_signal)
        x_signal = self.cnn20(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm20(x_signal)
        x_signal = self.cnn21(x_signal)
        x_signal = self.batchnorm21(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_23(x_signal)
        x_signal = self.batchnorm_23(x_signal)
        x_signal = self.cnn22(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm22(x_signal)
        x_signal = self.cnn23(x_signal)
        x_signal = self.batchnorm23(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_25(x_signal)
        x_signal = self.batchnorm_25(x_signal)
        x_signal = self.cnn24(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm24(x_signal)
        x_signal = self.cnn25(x_signal)
        x_signal = self.batchnorm25(x_signal)
        x_signal = x_signal + shortcut
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_3(x_signal)
        x_signal = self.batchnorm_3(x_signal)
        x_signal = self.cnn2(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_5(x_signal)
        x_signal = self.batchnorm_5(x_signal)
        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_7(x_signal)
        x_signal = self.batchnorm_7(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_27(x_signal)
        x_signal = self.batchnorm_27(x_signal)
        x_signal = self.cnn26(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm26(x_signal)
        x_signal = self.cnn27(x_signal)
        x_signal = self.batchnorm27(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_29(x_signal)
        x_signal = self.batchnorm_29(x_signal)
        x_signal = self.cnn28(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm28(x_signal)
        x_signal = self.cnn29(x_signal)
        x_signal = self.batchnorm29(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_31(x_signal)
        x_signal = self.batchnorm_31(x_signal)
        x_signal = self.cnn30(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm30(x_signal)
        x_signal = self.cnn31(x_signal)
        x_signal = self.batchnorm31(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn_9(x_signal)
        x_signal = self.batchnorm_9(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm9(x_signal)
        shortcut = x_signal

        x_signal = self.cnn_11(x_signal)
        x_signal = self.batchnorm_11(x_signal)
        x_signal = self.cnn10(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.cnn_13(x_signal)
        x_signal = self.batchnorm_13(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm13(x_signal)
        x_signal = x_signal + shortcut
        shortcut = x_signal

        x_signal = self.avgpool4(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_56(BasicModule):
    def __init__(self,structure = 'None', input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_56,self).__init__()
        self.model_name = 'cnn1d_sxy_4_56: resnet' + structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.module1 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_57(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_57,self).__init__()
        self.model_name = 'cnn1d_sxy_4_57: resnet2'+ structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.module1 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.module2 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_58(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_58,self).__init__()
        self.model_name = 'cnn1d_sxy_4_58: resnet2'+ structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.module1 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module2 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module3 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.module3(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_59(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_59,self).__init__()
        self.model_name = 'cnn1d_sxy_4_59: resnet3*2'+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)  

        self.module1 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module2 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module3 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 512)

        self.module4 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.module5 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.module6 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.module3(x_signal)

        x_signal = self.module4(x_signal)

        x_signal = self.module5(x_signal)

        x_signal = self.module6(x_signal)        

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_60(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_60,self).__init__()
        self.model_name = 'cnn1d_sxy_4_60: resnet3*3'+ structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64, hidden_dim = 256, output_dim = 64)

        self.module2 = self.structure(input_dim = 64, hidden_dim = 256, output_dim = 64)

        self.module3 = self.structure(input_dim = 64, hidden_dim = 256, output_dim = 256)

        self.module4 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module5 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module6 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 512)

        self.module7 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.module8 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.module9 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,100)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.module3(x_signal)

        x_signal = self.module4(x_signal)

        x_signal = self.module5(x_signal)

        x_signal = self.module6(x_signal)  

        x_signal = self.module7(x_signal)

        x_signal = self.module8(x_signal)

        x_signal = self.module9(x_signal)  

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        # x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_61(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_61,self).__init__()
        self.model_name = 'cnn1d_sxy_4_61: resnet3*4'+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64, hidden_dim = 128, output_dim = 64)

        self.module2 = self.structure(input_dim = 64, hidden_dim = 128, output_dim = 64)

        self.module3 = self.structure(input_dim = 64, hidden_dim = 128, output_dim = 128)

        self.module4 = self.structure(input_dim = 128, hidden_dim = 256, output_dim = 128)

        self.module5 = self.structure(input_dim = 128, hidden_dim = 256, output_dim = 128)

        self.module6 = self.structure(input_dim = 128, hidden_dim = 256, output_dim = 256)

        self.module7 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module8 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module9 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 512)

        self.module10 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.module11 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.module12 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,100)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.module3(x_signal)

        x_signal = self.module4(x_signal)

        x_signal = self.module5(x_signal)

        x_signal = self.module6(x_signal)  

        x_signal = self.module7(x_signal)

        x_signal = self.module8(x_signal)

        x_signal = self.module9(x_signal)  

        x_signal = self.module10(x_signal)

        x_signal = self.module11(x_signal)

        x_signal = self.module12(x_signal)  

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_62(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_62,self).__init__()
        self.model_name = 'cnn1d_sxy_4_62: resnet3*5:RESNET'+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64, hidden_dim = 128, output_dim = 64)

        self.module2 = self.structure(input_dim = 64, hidden_dim = 128, output_dim = 64)

        self.module3 = self.structure(input_dim = 64, hidden_dim = 128, output_dim = 128)

        self.module4 = self.structure(input_dim = 128, hidden_dim = 256, output_dim = 128)

        self.module5 = self.structure(input_dim = 128, hidden_dim = 256, output_dim = 128)

        self.module6 = self.structure(input_dim = 128, hidden_dim = 256, output_dim = 256)

        self.module7 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module8 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module9 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module10 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module11 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module12 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 512)

        self.module13 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.module14 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.module15 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,100)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.module3(x_signal)

        x_signal = self.module4(x_signal)

        x_signal = self.module5(x_signal)

        x_signal = self.module6(x_signal)  

        x_signal = self.module7(x_signal)

        x_signal = self.module8(x_signal)

        x_signal = self.module9(x_signal)  

        x_signal = self.module10(x_signal)

        x_signal = self.module11(x_signal)

        x_signal = self.module12(x_signal)  

        x_signal = self.module13(x_signal)

        x_signal = self.module14(x_signal)

        x_signal = self.module15(x_signal)  

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_63(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_63,self).__init__()
        self.model_name = 'cnn1d_sxy_4_63: resnet3*5:RESNET'+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64, hidden_dim = 128, output_dim = 64)

        self.module2 = self.structure(input_dim = 64, hidden_dim = 128, output_dim = 64)

        self.module3 = self.structure(input_dim = 64, hidden_dim = 128, output_dim = 128)

        self.module4 = self.structure(input_dim = 128, hidden_dim = 256, output_dim = 128)

        self.module5 = self.structure(input_dim = 128, hidden_dim = 256, output_dim = 128)

        self.module6 = self.structure(input_dim = 128, hidden_dim = 256, output_dim = 256)

        self.module7 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module8 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module9 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module10 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module11 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 256)

        self.module12 = self.structure(input_dim = 256, hidden_dim = 512, output_dim = 512)

        self.module13 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.module14 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.module15 = self.structure(input_dim = 512, hidden_dim = 1024, output_dim = 512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,100)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.module3(x_signal)

        x_signal = self.module4(x_signal)

        x_signal = self.module5(x_signal)

        x_signal = self.module6(x_signal)  

        x_signal = self.module7(x_signal)

        x_signal = self.module8(x_signal)

        x_signal = self.module9(x_signal)  

        x_signal = self.module10(x_signal)

        x_signal = self.module11(x_signal)

        x_signal = self.module12(x_signal)  

        x_signal = self.module13(x_signal)

        x_signal = self.module14(x_signal)

        x_signal = self.module15(x_signal)  

        # x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_64(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_64,self).__init__()
        self.model_name = 'cnn1d_sxy_4_64: resnet + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.module1 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 7, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_65(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_65,self).__init__()
        self.model_name = 'cnn1d_sxy_4_65: resnet2 + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.module1 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.module2 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_66(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_66,self).__init__()
        self.model_name = 'cnn1d_sxy_4_66: resnet2 + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.module1 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.module2 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_67(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_67,self).__init__()
        self.model_name = 'cnn1d_sxy_4_67: resnet3*5:RESNET + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128,output_dim = 64)

        self.module2 = self.structure(input_dim = 64,hidden_dim = 128,output_dim = 64)

        self.module3 = self.structure(input_dim = 64,hidden_dim = 128,output_dim = 64)

        self.module4 = self.structure(input_dim = 64,hidden_dim = 128,output_dim = 128)

        self.module5 = self.structure(input_dim = 128,hidden_dim = 256,output_dim = 128)

        self.module6 = self.structure(input_dim = 128,hidden_dim = 256,output_dim = 128)

        self.module7 = self.structure(input_dim = 128,hidden_dim = 256,output_dim = 256)

        self.module8 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 256)

        self.module9 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 256)

        self.module10 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 256)

        self.module11 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 256)

        self.module12 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 256)

        self.module13 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 512)

        self.module14 = self.structure(input_dim = 512,hidden_dim = 1024,output_dim = 512)

        self.module15 = self.structure(input_dim = 512,hidden_dim = 1024,output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)
        x_signal = self.module2(x_signal)
        x_signal = self.module3(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module4(x_signal)
        x_signal = self.module5(x_signal)
        x_signal = self.module6(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.module7(x_signal)
        x_signal = self.module8(x_signal)
        x_signal = self.module9(x_signal)

        x_signal = self.module10(x_signal)
        x_signal = self.module11(x_signal)
        x_signal = self.module12(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.module13(x_signal)
        x_signal = self.module14(x_signal)
        x_signal = self.module15(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)
        
        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_68(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_68,self).__init__()
        self.model_name = 'cnn1d_sxy_4_68: resnet3 + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.module1 = self.structure(input_dim = 256, hidden_dim = 512,output_dim = 256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.module2 = self.structure(input_dim = 256, hidden_dim = 512,output_dim = 256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.module3 = self.structure(input_dim = 256, hidden_dim = 512,output_dim = 256)

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.module3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_69(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_69,self).__init__()
        self.model_name = 'cnn1d_sxy_4_69: resnet3*2 + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)       

        self.module1 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module2 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module3 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module4 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module5 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)
        self.module6 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        #self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)
        x_signal = self.module2(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module3(x_signal)
        x_signal = self.module4(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.module5(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.module6(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_70(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_70,self).__init__()
        self.model_name = 'cnn1d_sxy_4_70: resnet3*3 + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 256, output_dim = 64)
        self.module2 = self.structure(input_dim = 64,hidden_dim = 256, output_dim = 64)
        self.module3 = self.structure(input_dim = 64,hidden_dim = 256, output_dim = 64)
        self.module4 = self.structure(input_dim = 64,hidden_dim = 256, output_dim = 256)
        self.module5 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module6 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module7 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module8 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)
        self.module9 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)
        x_signal = self.module2(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module3(x_signal)
        x_signal = self.module4(x_signal)
        x_signal = self.module5(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.module6(x_signal)
        x_signal = self.module7(x_signal)
        x_signal = self.module8(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.module9(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_71(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_71,self).__init__()
        self.model_name = 'cnn1d_sxy_4_71: resnet3*4 + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 64)
        self.module2 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 64)
        self.module3 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 64)
        self.module4 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 128)
        self.module5 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 128)
        self.module6 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 128)
        self.module7 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 256)
        self.module8 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module9 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module10 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module11 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)
        self.module12 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)
        x_signal = self.module2(x_signal)
        x_signal = self.module3(x_signal)  

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module4(x_signal)
        x_signal = self.module5(x_signal)
        x_signal = self.module6(x_signal)  

        x_signal = self.avgpool2(x_signal)

        x_signal = self.module7(x_signal)
        x_signal = self.module8(x_signal)
        x_signal = self.module9(x_signal) 

        x_signal = self.avgpool3(x_signal)

        x_signal = self.module10(x_signal)
        x_signal = self.module11(x_signal)
        x_signal = self.module12(x_signal)  

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_72(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_72,self).__init__()
        self.model_name = 'cnn1d_sxy_4_72: resnet3*5:RESNET + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 64)
        self.module2 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 64)
        self.module3 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 64)
        self.module4 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 128)
        self.module5 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 128)
        self.module6 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 128)
        self.module7 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 256)
        self.module8 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module9 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module10 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module11 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module12 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module13 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module14 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)
        self.module15 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)
        x_signal = self.module2(x_signal)
        x_signal = self.module3(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.module4(x_signal)
        x_signal = self.module5(x_signal)
        x_signal = self.module6(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.module7(x_signal)
        x_signal = self.module8(x_signal)
        x_signal = self.module9(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.module10(x_signal)
        x_signal = self.module11(x_signal)
        x_signal = self.module12(x_signal)
        x_signal = self.module13(x_signal)
        x_signal = self.module14(x_signal)
        x_signal = self.module15(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_73(BasicModule):
    def __init__(self,structure = 'None',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_73,self).__init__()
        self.model_name = 'cnn1d_sxy_4_73: resnet3*5:RESNET + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 64)
        self.module2 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 64)
        self.module3 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 64)
        self.module4 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 128)
        self.module5 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 128)
        self.module6 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 128)
        self.module7 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 256)
        self.module8 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module9 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module10 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module11 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module12 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module13 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module14 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)
        self.module15 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)
        x_signal = self.module2(x_signal)
        x_signal = self.module3(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module4(x_signal)
        x_signal = self.module5(x_signal)
        x_signal = self.module6(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.module7(x_signal)
        x_signal = self.module8(x_signal)
        x_signal = self.module9(x_signal)

        x_signal = self.module10(x_signal)
        x_signal = self.module11(x_signal)
        x_signal = self.module12(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.module13(x_signal)
        x_signal = self.module14(x_signal)
        x_signal = self.module15(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_74(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_74,self).__init__()
        self.model_name = 'cnn1d_sxy_4_74: resnet_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.module1 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 4, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_75(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_75,self).__init__()
        self.model_name = 'cnn1d_sxy_4_75: resnet2_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)
               
        self.module1 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.module2 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_76(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_76,self).__init__()
        self.model_name = 'cnn1d_sxy_4_76: resnet3*5:RESNET_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128,output_dim = 128)

        self.module2 = self.structure(input_dim = 128,hidden_dim = 256,output_dim = 256)

        self.module3 = self.structure(input_dim = 256,hidden_dim = 512,output_dim = 512)

        self.module4 = self.structure(input_dim = 512,hidden_dim = 1024,output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)
        x_signal = self.module2(x_signal)
        x_signal = self.module3(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module4(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)
        
        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_77(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_77,self).__init__()
        self.model_name = 'cnn1d_sxy_4_77: resnet3_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.module1 = self.structure(input_dim = 256, hidden_dim = 512,output_dim = 256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.module2 = self.structure(input_dim = 256, hidden_dim = 512,output_dim = 256)

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.module3 = self.structure(input_dim = 256, hidden_dim = 512,output_dim = 256)

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn_timeline = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(256,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.module3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_78(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_78,self).__init__()
        self.model_name = 'cnn1d_sxy_4_78: resnet3*2_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)       

        self.module1 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module2 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module3 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module4 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module5 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        #self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)
        x_signal = self.module2(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module3(x_signal)
        x_signal = self.module4(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.module5(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_79(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_79,self).__init__()
        self.model_name = 'cnn1d_sxy_4_79: resnet3*3_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 128)
        self.module2 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 256)
        self.module3 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module4 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.module3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.module4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_80(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_80,self).__init__()
        self.model_name = 'cnn1d_sxy_4_80: resnet3*4_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 128)
        self.module2 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 256)
        self.module3 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module4 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.module3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.module4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_81(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_81,self).__init__()
        self.model_name = 'cnn1d_sxy_4_81: resnet3*4_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 128)
        self.module2 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 256)
        self.module3 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module4 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.module1(x_signal)
        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.module2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.module3(x_signal)
        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.module4(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_82(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_82,self).__init__()
        self.model_name = 'cnn1d_sxy_4_82: resnet3*4_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 128)
        self.module2 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 256)
        self.module3 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module4 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.module3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.module4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_83(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_83,self).__init__()
        self.model_name = 'cnn1d_sxy_4_83: resnet3*4_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 128)
        self.module2 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 256)
        self.module3 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module4 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,128)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.module3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.module4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_84(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_84,self).__init__()
        self.model_name = 'cnn1d_sxy_4_84: resnet3*4_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 128)
        self.module2 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 256)
        self.module3 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)
        self.module4 = self.structure(input_dim = 512,hidden_dim = 1024, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        # self.linear = nn.Linear(512,128)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.module3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.module4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        # x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_85(BasicModule):
    def __init__(self,structure = 'bottleneck_1_cut2',input_dim=24,drop_rate=0.1):
        super(cnn1d_sxy_4_85,self).__init__()
        self.model_name = 'cnn1d_sxy_4_85: resnet3*4_cut + '+structure
        self.structure = eval(structure)

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1,padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.module1 = self.structure(input_dim = 64,hidden_dim = 128, output_dim = 128)
        self.module2 = self.structure(input_dim = 128,hidden_dim = 256, output_dim = 256)
        self.module3 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 256)
        self.module4 = self.structure(input_dim = 256,hidden_dim = 512, output_dim = 512)

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        # self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(512,1024)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(1024,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.module1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.module2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.module3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.module4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out