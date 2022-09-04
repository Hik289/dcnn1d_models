from models.BasicModule import BasicModule
import torch.nn as nn
import torch

from .cnn_model_sxy.bottleneck1d import bottleneck_1
from .cnn_model_sxy.bottleneck1d import bottleneck_2

class cnn1d_sxy_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_1,self).__init__()
        self.model_name = 'cnn1d_sxy_1: ordinary_2_pointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,stride = 1,padding = 1,padding_mode='replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,padding_mode='replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,padding_mode='replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.pointwise = nn.Conv1d(in_channels= 15, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 128, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


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

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise(x_signal[:,:,-15:].permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_1_false(BasicModule):
    def __init__(self,kernel_size = 3,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_1_false,self).__init__()
        self.model_name = 'cnn1d_sxy_1: ordinary_2_pointwise_12avg_3kernel'

        self.avgpool = nn.AvgPool1d(kernel_size= 12, stride = 1)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=kernel_size,stride = 1,padding = kernel_size//2,padding_mode='replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=12,stride=1)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size,padding = kernel_size//2,padding_mode='replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=12,stride=1)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size,padding = kernel_size//2,padding_mode='replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=12,stride=1)

        self.pointwise = nn.Conv1d(in_channels= 15, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 128, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):
        
        x_signal = x.permute(0,2,1)
        x_signal = self.avgpool(x_signal)

        x_signal = self.cnn1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise(x_signal[:,:,-15:].permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_1_false12(BasicModule):
    def __init__(self,kernel_size = 12,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_1_false12,self).__init__()
        self.model_name = 'cnn1d_sxy_1: ordinary_2_pointwise_12avg_12kernel'

        self.avgpool = nn.AvgPool1d(kernel_size= 12, stride = 1)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=kernel_size,stride = 1,padding = kernel_size//2,padding_mode='replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=12,stride=1)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size,padding = kernel_size//2,padding_mode='replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=12,stride=1)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size,padding = kernel_size//2,padding_mode='replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=12,stride=1)

        self.pointwise = nn.Conv1d(in_channels= 15, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 128, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):
        
        x_signal = x.permute(0,2,1)
        x_signal = self.avgpool(x_signal)

        x_signal = self.cnn1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise(x_signal[:,:,-15:].permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_2(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_2,self).__init__()
        self.model_name = 'cnn1d_sxy_2: ordinary_3_ordinary_ordinary1'
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=12,stride=1,padding = 6,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=12,stride = 1,padding = 6,count_include_pad = False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=12,stride = 1,padding = 6,count_include_pad = False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size= 12, stride = 1,padding = 6,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 15, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 512, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()

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

        x_signal = self.pointwise1(x_signal[:,:,-15:].permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_2_false(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_2_false,self).__init__()
        self.model_name = 'cnn1d_sxy_2: ordinary_3_ordinary_ordinary1_allpointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=12,stride=1,padding = 6,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=12,stride = 1,padding = 6,count_include_pad = False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=12,stride = 1,padding = 6,count_include_pad = False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size= 12, stride = 1,padding = 6,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 512, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()

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

        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_2_false490(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_2_false,self).__init__()
        self.model_name = 'cnn1d_sxy_2: ordinary_3_ordinary_ordinary1_allpointwise490'
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=12,stride=1,padding = 6,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=12,stride = 1,padding = 6,count_include_pad = False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=12,stride = 1,padding = 6,count_include_pad = False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size= 12, stride = 1,padding = 6,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 490, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 512, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()

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

        x_signal = self.pointwise1(x_signal[:,:,-490:].permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_3,self).__init__()
        self.model_name = 'cnn1d_sxy_3: ordinary_3_ordinary_ordinary1_pointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=12,stride=1,padding = 6,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=12,stride=1,padding = 6,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()

        self.pointwise = nn.Conv1d(in_channels= 490, out_channels=1, kernel_size=1, stride = 1)

        self.linear = nn.Linear(in_features= 512, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)

        x_signal = self.pointwise(x_signal[:,:,-490:].permute(0,2,1)).squeeze(1)
        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_4,self).__init__()
        self.model_name = 'cnn1d_sxy_4: ordinary_3_avg_ordinary_3  -- largeFOV'
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()

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

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        x_signal = self.avgpool(x_signal)

        x_day = x_signal.reshape((x_signal.shape[0], x_signal.shape[1], 10, x_signal.shape[-1]// 10))
        x_day = torch.mean(x_day, axis = 3) 

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)
        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_5(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_5,self).__init__()
        self.model_name = 'cnn1d_sxy_5: ordinary_3_avg_ordinary_3  -- largeFOV'

        self.side1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)) 

        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()

    def forward(self,x):

        x_side1 = self.side1(x[:,-49:,:].permute(0,2,1)).squeeze(-1)
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

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        x_signal = self.avgpool(x_signal)

        x_day = x_signal.reshape((x_signal.shape[0], x_signal.shape[1], 10, x_signal.shape[-1]// 10))
        x_day = torch.mean(x_day, axis = 3) 

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = x_signal[:,:,-49:] + x_side1
        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)


        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_6(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_6,self).__init__()
        self.model_name = 'cnn1d_sxy_6: -- deeplabV0'

        self.side1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)) 

        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.side2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)) 

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()

    def forward(self,x):

        x_side1 = self.side1(x.permute(0,2,1)).squeeze(-1)
        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_side2 = self.side2(x_signal).squeeze(-1)

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

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        x_signal = self.avgpool(x_signal)

        x_day = x_signal.reshape((x_signal.shape[0], x_signal.shape[1], 10, x_signal.shape[-1]// 10))
        x_day = torch.mean(x_day, axis = 3) 

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = x_signal+ x_side1 + x_side2

        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)


        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_7(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_7,self).__init__()
        self.model_name = 'cnn1d_sxy_7: -- deeplabV0'

        self.side1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)) 

        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.side2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)) 

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.side3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)) 

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()

    def forward(self,x):

        x_side1 = self.side1(x.permute(0,2,1)).squeeze(-1)
        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_side2 = self.side2(x_signal).squeeze(-1)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_side3 = self.side3(x_signal).squeeze(-1)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        x_signal = self.avgpool(x_signal)

        x_day = x_signal.reshape((x_signal.shape[0], x_signal.shape[1], 10, x_signal.shape[-1]// 10))
        x_day = torch.mean(x_day, axis = 3) 

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = x_signal+ x_side1 + x_side2 + x_side3

        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)


        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_8(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_8,self).__init__()
        self.model_name = 'cnn1d_sxy_8: -- deeplabV0'

        self.side1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)) 

        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.side2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)) 

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.side3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)) 

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.side4 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)) 

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()

    def forward(self,x):

        x_side1 = self.side1(x.permute(0,2,1)).squeeze(-1)
        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_side2 = self.side2(x_signal).squeeze(-1)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_side3 = self.side3(x_signal).squeeze(-1)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_side4 = self.side4(x_signal).squeeze(-1)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        x_signal = self.avgpool(x_signal)

        x_day = x_signal.reshape((x_signal.shape[0], x_signal.shape[1], 10, x_signal.shape[-1]// 10))
        x_day = torch.mean(x_day, axis = 3) 

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = x_signal+ x_side1 + x_side2 + x_side3 + x_side4

        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)


        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_9_false(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_9_false,self).__init__()
        self.model_name = 'cnn1d_sxy_9_false: bottleneck_1_pointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1,stride = 1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=32, out_channels=6, kernel_size=1)
        self.batchnorm3 = nn.BatchNorm1d(6)
        self.act3 = nn.ReLU()

        self.shortcut = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=6, kernel_size=1, stride = 1),
                                      nn.BatchNorm1d(6))

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.linear = nn.Linear(in_features= 6, out_features= 10)

        self.end_layer = nn.Linear(10,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act3(x_signal)
    
        shortcut = self.shortcut(x.permute(0,2,1))

        x_signal = x_signal + shortcut

        x_day = x_signal.reshape((x_signal.shape[0], x_signal.shape[1], 10, x_signal.shape[-1]// 10))
        x_day = torch.mean(x_day, axis = 3) 

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_9(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_9,self).__init__()
        self.model_name = 'cnn1d_sxy_9: bottleneck_2_pointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=32, out_channels=3, kernel_size=1)
        self.batchnorm3 = nn.BatchNorm1d(3)
        self.act3 = nn.ReLU()

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.linear = nn.Linear(in_features= 3, out_features= 10)

        self.end_layer = nn.Linear(10,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act3(x_signal)

        x_signal = x_signal + (x.permute(0,2,1))

        x_day = x_signal.reshape((x_signal.shape[0], x_signal.shape[1], 10, x_signal.shape[-1]// 10))
        x_day = torch.mean(x_day, axis = 3) 

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_10(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_10,self).__init__()
        self.model_name = 'cnn1d_sxy_10: -- DeeplabV2:0 -- 20'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(128)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.dropout(x_signal)

        x_day = x_signal.reshape((x_signal.shape[0], x_signal.shape[1], 10, x_signal.shape[-1]// 10))
        x_day = torch.mean(x_day, axis = 3) 

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_11(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_11,self).__init__()
        self.model_name = 'cnn1d_sxy_11: -- DeeplabV2:1'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(128)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim=128, output_dim=256)
        self.bottleneck1_day = bottleneck_1(input_dim = 128, output_dim= 256)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 256, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day).squeeze(-1)
        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_12(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_12,self).__init__()
        self.model_name = 'cnn1d_sxy_12: -- DeeplabV2:1'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(128)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_2(input_dim=128, output_dim=256)
        self.bottleneck1_day = bottleneck_2(input_dim = 128, output_dim= 256)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day).squeeze(-1)
        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_13(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_13,self).__init__()
        self.model_name = 'cnn1d_sxy_13: -- DeeplabV2:1'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(64)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2 = bottleneck_2(input_dim=128, output_dim=256)
        self.bottleneck1_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2_day = bottleneck_2(input_dim = 128, output_dim= 256)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day).squeeze(-1)

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_14(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_14,self).__init__()
        self.model_name = 'cnn1d_sxy_14: -- DeeplabV2:1 no avgpool'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(64)
        self.act0 = nn.ReLU()
        # self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        # self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        # self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2 = bottleneck_2(input_dim=128, output_dim=256)
        self.bottleneck1_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2_day = bottleneck_2(input_dim = 128, output_dim= 256)

        self.pointwise1 = nn.Conv1d(in_channels= 200, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        # x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day).squeeze(-1)

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        # x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        # x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal[:,:,-200:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_14_cat(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_14_cat,self).__init__()
        self.model_name = 'cnn1d_sxy_14_cat: -- DeeplabV2:1 no avgpool cat'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(64)
        self.act0 = nn.ReLU()
        # self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        # self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        # self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2 = bottleneck_2(input_dim=128, output_dim=256)
        self.bottleneck1_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2_day = bottleneck_2(input_dim = 128, output_dim= 256)

        self.pointwise1 = nn.Conv1d(in_channels= 200, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        # x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day).squeeze(-1)

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        # x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        # x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal[:,:,-200:].permute(0,2,1)).squeeze(1)
        
        x_signal = torch.cat([x_signal, x_day], axis = 1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_14_2(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_14_2,self).__init__()
        self.model_name = 'cnn1d_sxy_14_2: -- DeeplabV2:1+'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(64)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2 = bottleneck_2(input_dim=128, output_dim=128)
        self.bottleneck1_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2_day = bottleneck_2(input_dim = 128, output_dim= 128)

        self.bottleneck3 = bottleneck_1(input_dim = 128, output_dim = 256)
        self.bottleneck4 = bottleneck_2(input_dim=256, output_dim=256)
        self.bottleneck3_day = bottleneck_1(input_dim = 128, output_dim = 256)
        self.bottleneck4_day = bottleneck_2(input_dim = 256, output_dim= 256)

        self.pointwise1 = nn.Conv1d(in_channels= 200, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day)
        x_day = self.bottleneck3_day(x_day)
        x_day = self.bottleneck4_day(x_day).squeeze(-1)

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)
        x_signal = self.bottleneck3(x_signal)
        x_signal = self.bottleneck4(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal[:,:,-200:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_15(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_15,self).__init__()
        self.model_name = 'cnn1d_sxy_15: -- DeeplabV2:FINAL'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(32)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2 = bottleneck_2(input_dim=64, output_dim=64)
        self.bottleneck1_day = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2_day = bottleneck_2(input_dim = 64, output_dim= 64)

        self.bottleneck3 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4 = bottleneck_2(input_dim=128, output_dim=128)
        self.bottleneck3_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4_day = bottleneck_2(input_dim = 128, output_dim= 128)

        self.assp_frac1 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1)
        self.assp_frac2 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2)
        self.assp_frac3 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4)
        self.assp_frac4 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8)

        self.assp_frac_day1 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1)
        self.assp_frac_day2 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)
        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day)
        x_day = self.bottleneck3_day(x_day)
        x_day = self.bottleneck4_day(x_day).squeeze(-1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)
        x_signal = self.bottleneck3(x_signal)
        x_signal = self.bottleneck4(x_signal)

        assp1 = self.assp_frac1(x_signal)
        assp2 = self.assp_frac2(x_signal) 
        assp3 = self.assp_frac3(x_signal) 
        assp4 = self.assp_frac4(x_signal)

        assp1 = nn.functional.interpolate(assp1, size = 490)
        assp2 = nn.functional.interpolate(assp2, size = 490)
        assp3 = nn.functional.interpolate(assp3, size = 490)
        assp4 = nn.functional.interpolate(assp4, size = 490)
        x_signal = assp1 + assp2 + assp3 + assp4

        assp_day1 = self.assp_frac_day1(x_day)
        assp_day2 = self.assp_frac_day2(x_day) 
    
        x_day1 = nn.functional.interpolate(assp_day1, size = 10)
        x_day2 = nn.functional.interpolate(assp_day2, size = 10)
        x_day =  x_day1 +x_day2       

        x_signal = self.dropout(x_signal)
        x_day = self.dropout(x_day)

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)
        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_16(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_16,self).__init__()
        self.model_name = 'cnn1d_sxy_16: -- DeeplabV2:assp'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(32)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2 = bottleneck_2(input_dim=64, output_dim=64)
        self.bottleneck1_day = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2_day = bottleneck_2(input_dim = 64, output_dim= 64)

        self.bottleneck3 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4 = bottleneck_2(input_dim=128, output_dim=128)
        self.bottleneck3_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4_day = bottleneck_2(input_dim = 128, output_dim= 128)

        self.assp_frac1 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1)
        self.assp_frac2 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2)
        self.assp_frac3 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4)
        self.assp_frac4 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8)

        self.assp_frac_day1 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1)
        self.assp_frac_day2 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2)
        self.assp_frac_day3 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4)
        self.assp_frac_day4 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8)


        self.pointwise1 = nn.Conv1d(in_channels= 200, out_channels=1, kernel_size=1, stride = 1)
        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day)
        x_day = self.bottleneck3_day(x_day)
        x_day = self.bottleneck4_day(x_day).squeeze(-1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)
        x_signal = self.bottleneck3(x_signal)
        x_signal = self.bottleneck4(x_signal)

        assp1 = self.assp_frac1(x_signal)
        assp2 = self.assp_frac2(x_signal) 
        assp3 = self.assp_frac3(x_signal) 
        assp4 = self.assp_frac4(x_signal)

        assp1 = nn.functional.interpolate(assp1, size = 490)
        assp2 = nn.functional.interpolate(assp2, size = 490)
        assp3 = nn.functional.interpolate(assp3, size = 490)
        assp4 = nn.functional.interpolate(assp4, size = 490)
        x_signal = assp1 + assp2 + assp3 + assp4

        assp_day1 = self.assp_frac_day1(x_day)
        assp_day2 = self.assp_frac_day2(x_day) 
        assp_day3 = self.assp_frac_day3(x_day)

        x_day1 = nn.functional.interpolate(assp_day1, size = 10)
        x_day2 = nn.functional.interpolate(assp_day2, size = 10)
        x_day3 = nn.functional.interpolate(assp_day3, size = 10)

        x_day =  x_day1 + x_day2 + x_day3     

        x_signal = self.dropout(x_signal)
        x_day = self.dropout(x_day)

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)
        x_signal = self.pointwise1(x_signal[:,:,-200:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_17(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_17,self).__init__()
        self.model_name = 'cnn1d_sxy_17: -- DeeplabV2:ASSP+interpolate no pointwise'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(32)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2 = bottleneck_2(input_dim=64, output_dim=64)
        self.bottleneck1_day = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2_day = bottleneck_2(input_dim = 64, output_dim= 64)

        self.bottleneck3 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4 = bottleneck_2(input_dim=128, output_dim=128)
        self.bottleneck3_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4_day = bottleneck_2(input_dim = 128, output_dim= 128)

        self.assp_frac1 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1)
        self.assp_frac2 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2)
        self.assp_frac3 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4)
        self.assp_frac4 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8)

        self.assp_frac_day1 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1)
        self.assp_frac_day2 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2)
        self.assp_frac_day3 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4)
        self.assp_frac_day4 = nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day)
        x_day = self.bottleneck3_day(x_day)
        x_day = self.bottleneck4_day(x_day).squeeze(-1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)
        x_signal = self.bottleneck3(x_signal)
        x_signal = self.bottleneck4(x_signal)

        assp1 = self.assp_frac1(x_signal)[:,:,-1]
        assp2 = self.assp_frac2(x_signal)[:,:,-1] 
        assp3 = self.assp_frac3(x_signal)[:,:,-1] 
        assp4 = self.assp_frac4(x_signal)[:,:,-1]

        x_signal = assp1 + assp2 + assp3 + assp4

        assp_day1 = self.assp_frac_day1(x_day)[:,:,-1]
        assp_day2 = self.assp_frac_day2(x_day)[:,:,-1] 
        assp_day3 = self.assp_frac_day3(x_day)[:,:,-1]

        x_day =  assp_day1 + assp_day2 + assp_day3     

        x_signal = self.dropout(x_signal)
        x_day = self.dropout(x_day)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_18(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_18,self).__init__()
        self.model_name = 'cnn1d_sxy_18: -- DeeplabV2:ASSP+interpolate no pointwise'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(32)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2 = bottleneck_2(input_dim=64, output_dim=64)
        self.bottleneck1_day = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2_day = bottleneck_2(input_dim = 64, output_dim= 64)

        self.bottleneck3 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4 = bottleneck_2(input_dim=128, output_dim=128)
        self.bottleneck3_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4_day = bottleneck_2(input_dim = 128, output_dim= 128)

        self.assp_frac1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())
        
        self.assp_frac2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac4 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU())

        self.assp_frac_day1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())
        
        self.assp_frac_day2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_frac_day3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_day_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                             nn.BatchNorm1d(256),
                                             nn.ReLU())

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day)
        x_day = self.bottleneck3_day(x_day)
        x_day = self.bottleneck4_day(x_day).squeeze(-1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)
        x_signal = self.bottleneck3(x_signal)
        x_signal = self.bottleneck4(x_signal)

        assp1 = self.assp_frac1(x_signal)[:,:,-1]
        assp2 = self.assp_frac2(x_signal)[:,:,-1] 
        assp3 = self.assp_frac3(x_signal)[:,:,-1] 
        assp4 = self.assp_frac4(x_signal)[:,:,-1]
        assp_origin = self.assp_origin(x_signal)[:,:,-1]

        x_signal = assp1 + assp2 + assp3 + assp4 + assp_origin

        assp_day1 = self.assp_frac_day1(x_day)[:,:,-1]
        assp_day2 = self.assp_frac_day2(x_day)[:,:,-1] 
        assp_day3 = self.assp_frac_day3(x_day)[:,:,-1]
        assp_day_origin = self.assp_frac_day3(x_day)[:,:,-1]
        x_day =  assp_day1 + assp_day2 + assp_day3 + assp_day_origin     

        x_signal = self.dropout(x_signal)
        x_day = self.dropout(x_day)
        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_19(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_19,self).__init__()
        self.model_name = 'cnn1d_sxy_19: -- DeeplabV2:FINAL'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(32)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2 = bottleneck_2(input_dim=64, output_dim=64)
        self.bottleneck1_day = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2_day = bottleneck_2(input_dim = 64, output_dim= 64)

        self.bottleneck3 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4 = bottleneck_2(input_dim=128, output_dim=128)
        self.bottleneck3_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4_day = bottleneck_2(input_dim = 128, output_dim= 128)

        self.assp_frac1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())
        
        self.assp_frac2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac4 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU())

        self.assp_frac_day1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())
        
        self.assp_frac_day2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_frac_day3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_frac_day_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                             nn.BatchNorm1d(256),
                                             nn.ReLU())

        self.pointwise1 = nn.Conv1d(in_channels= 490, out_channels=1, kernel_size=1, stride = 1)
        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day)
        x_day = self.bottleneck3_day(x_day)
        x_day = self.bottleneck4_day(x_day).squeeze(-1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)
        x_signal = self.bottleneck3(x_signal)
        x_signal = self.bottleneck4(x_signal)

        assp1 = self.assp_frac1(x_signal)
        assp2 = self.assp_frac2(x_signal) 
        assp3 = self.assp_frac3(x_signal)
        assp4 = self.assp_frac4(x_signal)
        assp_origin = self.assp_origin(x_signal)
        assp1 = nn.functional.interpolate(assp1, size = 490)
        assp2 = nn.functional.interpolate(assp2, size = 490)
        assp3 = nn.functional.interpolate(assp3, size = 490)
        assp4 = nn.functional.interpolate(assp4, size = 490)
        assp_origin = nn.functional.interpolate(assp_origin, size = 490)
        x_signal = assp1 + assp2 + assp3 + assp4 + assp_origin

        assp_day1 = self.assp_frac_day1(x_day)
        assp_day2 = self.assp_frac_day2(x_day)
        assp_day3 = self.assp_frac_day3(x_day)
        assp_origin_day = self.assp_frac_day_origin(x_day)
        assp_day1 = nn.functional.interpolate(assp_day1, size = 10)
        assp_day2 = nn.functional.interpolate(assp_day2, size = 10)
        assp_day3 = nn.functional.interpolate(assp_day3, size = 10)
        assp_origin_day = nn.functional.interpolate(assp_origin_day, size = 10)


        x_day =  assp_day1 + assp_day2 + assp_day3 + assp_origin_day    

        x_signal = self.dropout(x_signal)
        x_day = self.dropout(x_day)

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)
        x_signal = self.pointwise1(x_signal[:,:,-490:].permute(0,2,1)).squeeze(1)        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_20(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_20,self).__init__()
        self.model_name = 'cnn1d_sxy_20: -- DeeplabV3:1'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(32)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2 = bottleneck_2(input_dim=64, output_dim=64)
        self.bottleneck1_day = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2_day = bottleneck_2(input_dim = 64, output_dim= 64)

        self.bottleneck3 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4 = bottleneck_2(input_dim=128, output_dim=128)
        self.bottleneck3_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4_day = bottleneck_2(input_dim = 128, output_dim= 128)

        self.assp_frac1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())
        
        self.assp_frac2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac4 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU())

        self.down_sample = nn.Sequential(nn.Conv1d(in_channels=1280, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU())

        self.assp_frac_day1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())
        
        self.assp_frac_day2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_frac_day3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_frac_day_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                                  nn.BatchNorm1d(256),
                                                  nn.ReLU())

        self.down_sample_day = nn.Sequential(nn.Conv1d(in_channels=256*4, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                             nn.BatchNorm1d(256),
                                             nn.ReLU())

        self.pointwise1 = nn.Conv1d(in_channels= 490, out_channels=1, kernel_size=1, stride = 1)
        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day)
        x_day = self.bottleneck3_day(x_day)
        x_day = self.bottleneck4_day(x_day).squeeze(-1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)
        x_signal = self.bottleneck3(x_signal)
        x_signal = self.bottleneck4(x_signal)

        assp1 = self.assp_frac1(x_signal)
        assp2 = self.assp_frac2(x_signal) 
        assp3 = self.assp_frac3(x_signal)
        assp4 = self.assp_frac4(x_signal)
        assp_origin = self.assp_origin(x_signal)
        assp1 = nn.functional.interpolate(assp1, size = 490)
        assp2 = nn.functional.interpolate(assp2, size = 490)
        assp3 = nn.functional.interpolate(assp3, size = 490)
        assp4 = nn.functional.interpolate(assp4, size = 490)
        assp_origin = nn.functional.interpolate(assp_origin, size = 490)

        x_signal = torch.cat([assp1,assp2,assp3,assp4,assp_origin],axis = 1)
        x_signal = self.down_sample(x_signal)

        assp_day1 = self.assp_frac_day1(x_day)
        assp_day2 = self.assp_frac_day2(x_day)
        assp_day3 = self.assp_frac_day3(x_day)
        assp_origin_day = self.assp_frac_day_origin(x_day)
        assp_day1 = nn.functional.interpolate(assp_day1, size = 10)
        assp_day2 = nn.functional.interpolate(assp_day2, size = 10)
        assp_day3 = nn.functional.interpolate(assp_day3, size = 10)
        assp_origin_day = nn.functional.interpolate(assp_origin_day, size = 10)


        x_day = torch.cat([assp_day1,assp_day2,assp_day3,assp_origin_day],axis = 1)
        x_day = self.down_sample_day(x_day)

        x_signal = self.dropout(x_signal)
        x_day = self.dropout(x_day)

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)
        x_signal = self.pointwise1(x_signal[:,:,-490:].permute(0,2,1)).squeeze(1)        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_21(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_21,self).__init__()
        self.model_name = 'cnn1d_sxy_21: -- DeeplabV3:1 no pointwise'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(32)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2 = bottleneck_2(input_dim=64, output_dim=64)
        self.bottleneck1_day = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2_day = bottleneck_2(input_dim = 64, output_dim= 64)

        self.bottleneck3 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4 = bottleneck_2(input_dim=128, output_dim=128)
        self.bottleneck3_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4_day = bottleneck_2(input_dim = 128, output_dim= 128)

        self.assp_frac1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())
        
        self.assp_frac2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac4 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU())

        self.down_sample = nn.Sequential(nn.Conv1d(in_channels=1280, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU())

        self.assp_frac_day1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())
        
        self.assp_frac_day2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_frac_day3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_frac_day_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                                  nn.BatchNorm1d(256),
                                                  nn.ReLU())

        self.down_sample_day = nn.Sequential(nn.Conv1d(in_channels=256*4, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                             nn.BatchNorm1d(256),
                                             nn.ReLU())

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day)
        x_day = self.bottleneck3_day(x_day)
        x_day = self.bottleneck4_day(x_day).squeeze(-1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)
        x_signal = self.bottleneck3(x_signal)
        x_signal = self.bottleneck4(x_signal)

        assp1 = self.assp_frac1(x_signal)[:,:,-1]
        assp2 = self.assp_frac2(x_signal)[:,:,-1]
        assp3 = self.assp_frac3(x_signal)[:,:,-1]
        assp4 = self.assp_frac4(x_signal)[:,:,-1]
        assp_origin = self.assp_origin(x_signal)[:,:,-1]

        x_signal = torch.cat([assp1,assp2,assp3,assp4,assp_origin],axis = 1).unsqueeze(-1)
        x_signal = self.down_sample(x_signal).squeeze(-1)

        assp_day1 = self.assp_frac_day1(x_day)[:,:,-1]
        assp_day2 = self.assp_frac_day2(x_day)[:,:,-1]
        assp_day3 = self.assp_frac_day3(x_day)[:,:,-1]
        assp_origin_day = self.assp_frac_day_origin(x_day)[:,:,-1]

        x_day = torch.cat([assp_day1,assp_day2,assp_day3,assp_origin_day],axis = 1).unsqueeze(-1)
        x_day = self.down_sample_day(x_day).squeeze(-1)

        x_signal = self.dropout(x_signal)
        x_day = self.dropout(x_day)
      
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_22(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_22,self).__init__()
        self.model_name = 'cnn1d_sxy_22: -- assp:0 no pointwise -- light model'
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3,padding = 1, stride = 1, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1, stride = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()

        self.assp_frac1 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU())
        
        self.assp_frac2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU())

        self.assp_frac3 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU())

        self.assp_frac4 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU())

        self.assp_origin = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64,kernel_size=1,stride = 1),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU()
                                         )

        self.down_sample = nn.Sequential(nn.Conv1d(in_channels=320, out_channels=64,kernel_size=1),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU())

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 64, out_features= 10)

        self.end_layer = nn.Linear(10,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.dropout(x_signal)

        assp1 = self.assp_frac1(x_signal)[:,:,-1]
        assp2 =  self.assp_frac2(x_signal)[:,:,-1]
        assp3 = self.assp_frac3(x_signal)[:,:,-1]
        assp4 = self.assp_frac4(x_signal)[:,:,-1]
        assp_origin = self.assp_origin(x_signal)[:,:,-1]

        x_signal = torch.cat([assp1,assp2,assp3,assp4,assp_origin],axis = 1).unsqueeze(-1)

        x_signal = self.down_sample(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.dropout(x_signal).squeeze(-1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_23(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_23,self).__init__()
        self.model_name = 'cnn1d_sxy_23: -- DeeplabV3:3'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(32)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2 = bottleneck_2(input_dim=64, output_dim=64)
        self.bottleneck1_day = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2_day = bottleneck_2(input_dim = 64, output_dim= 64)

        self.bottleneck3 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4 = bottleneck_2(input_dim=128, output_dim=128)
        self.bottleneck3_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4_day = bottleneck_2(input_dim = 128, output_dim= 128)

        self.assp_frac1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())
        
        self.assp_frac2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac4 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU())

        self.down_sample = nn.Sequential(nn.Conv1d(in_channels=1280, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU())

        self.up_sample = nn.Sequential(nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size=3,padding = 1, stride = 1, padding_mode='replicate'),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(),
                                       nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size=3,padding = 1, stride = 1, padding_mode='replicate'))

        self.assp_frac_day1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())
        
        self.assp_frac_day2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_frac_day3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_frac_day_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                                  nn.BatchNorm1d(256),
                                                  nn.ReLU())

        self.down_sample_day = nn.Sequential(nn.Conv1d(in_channels=256*4, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                             nn.BatchNorm1d(256),
                                             nn.ReLU())

        self.up_sample_day = nn.Sequential(nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size=3,padding = 1, stride = 1, padding_mode='replicate'),
                                           nn.BatchNorm1d(256),
                                           nn.ReLU(),
                                           nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size=3,padding = 1, stride = 1, padding_mode='replicate'))

        self.pointwise1 = nn.Conv1d(in_channels= 490, out_channels=1, kernel_size=1, stride = 1)
        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day)
        x_day = self.bottleneck3_day(x_day)
        x_day = self.bottleneck4_day(x_day).squeeze(-1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)
        x_signal = self.bottleneck3(x_signal)
        x_signal = self.bottleneck4(x_signal)

        assp1 = self.assp_frac1(x_signal)
        assp2 = self.assp_frac2(x_signal) 
        assp3 = self.assp_frac3(x_signal)
        assp4 = self.assp_frac4(x_signal)
        assp_origin = self.assp_origin(x_signal)
        assp1 = nn.functional.interpolate(assp1, size = 490)
        assp2 = nn.functional.interpolate(assp2, size = 490)
        assp3 = nn.functional.interpolate(assp3, size = 490)
        assp4 = nn.functional.interpolate(assp4, size = 490)
        assp_origin = nn.functional.interpolate(assp_origin, size = 490)

        x_signal = torch.cat([assp1,assp2,assp3,assp4,assp_origin],axis = 1)
        x_signal = self.down_sample(x_signal)

        assp_day1 = self.assp_frac_day1(x_day)
        assp_day2 = self.assp_frac_day2(x_day)
        assp_day3 = self.assp_frac_day3(x_day)
        assp_origin_day = self.assp_frac_day_origin(x_day)
        assp_day1 = nn.functional.interpolate(assp_day1, size = 10)
        assp_day2 = nn.functional.interpolate(assp_day2, size = 10)
        assp_day3 = nn.functional.interpolate(assp_day3, size = 10)
        assp_origin_day = nn.functional.interpolate(assp_origin_day, size = 10)


        x_day = torch.cat([assp_day1,assp_day2,assp_day3,assp_origin_day],axis = 1)
        x_day = self.down_sample_day(x_day)

        x_signal = self.dropout(x_signal)
        x_signal = self.up_sample(x_signal)
        x_day = self.dropout(x_day)
        x_day = self.up_sample_day(x_day)

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)
        x_signal = self.pointwise1(x_signal[:,:,-490:].permute(0,2,1)).squeeze(1)        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_24(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_24,self).__init__()
        self.model_name = 'cnn1d_sxy_24: -- DeeplabV3:FINAL'

        self.cnn0 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(32)
        self.act0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2 = bottleneck_2(input_dim=64, output_dim=64)
        self.bottleneck1_day = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2_day = bottleneck_2(input_dim = 64, output_dim= 64)

        self.shortcut = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                      nn.BatchNorm1d(256),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'))

        self.shortcut_day = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                          nn.BatchNorm1d(256),
                                          nn.ReLU(),
                                          nn.Dropout(0.1),
                                          nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'))

        self.bottleneck3 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4 = bottleneck_2(input_dim=128, output_dim=128)
        self.bottleneck3_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck4_day = bottleneck_2(input_dim = 128, output_dim= 128)

        self.assp_frac1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())
        
        self.assp_frac2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_frac4 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=8),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU())

        self.assp_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU())

        self.down_sample = nn.Sequential(nn.Conv1d(in_channels=1280, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU())

        self.up_sample = nn.Sequential(nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size=3,padding = 1, stride = 1, padding_mode='replicate'),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(),
                                       nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size=3,padding = 1, stride = 1, padding_mode='replicate'))

        self.assp_frac_day1 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=1),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())
        
        self.assp_frac_day2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=2),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_frac_day3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=3,padding = 1, stride = 1, padding_mode='replicate',dilation=4),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU())

        self.assp_frac_day_origin = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                                  nn.BatchNorm1d(256),
                                                  nn.ReLU())

        self.down_sample_day = nn.Sequential(nn.Conv1d(in_channels=256*4, out_channels=256,kernel_size=1, stride = 1, padding_mode='replicate',dilation=1),
                                             nn.BatchNorm1d(256),
                                             nn.ReLU())

        self.up_sample_day = nn.Sequential(nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size=3,padding = 1, stride = 1, padding_mode='replicate'),
                                           nn.BatchNorm1d(256),
                                           nn.ReLU(),
                                           nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size=3,padding = 1, stride = 1, padding_mode='replicate'))

        self.pointwise1 = nn.Conv1d(in_channels= 490, out_channels=1, kernel_size=1, stride = 1)
        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2) 

        x_day = self.cnn0(x_day.permute(0,2,1))
        x_day = self.batchnorm0(x_day)
        x_day = self.act0(x_day)
        x_day = self.avgpool0(x_day)
        x_day = self.bottleneck1_day(x_day)
        x_day = self.bottleneck2_day(x_day)

        x_shortcut_day = self.shortcut_day(x_day)

        x_day = self.bottleneck3_day(x_day)
        x_day = self.bottleneck4_day(x_day).squeeze(-1)

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)
        x_signal = self.bottleneck2(x_signal)

        x_shortcut = self.shortcut(x_signal)

        x_signal = self.bottleneck3(x_signal)
        x_signal = self.bottleneck4(x_signal)

        assp1 = self.assp_frac1(x_signal)
        assp2 = self.assp_frac2(x_signal) 
        assp3 = self.assp_frac3(x_signal)
        assp4 = self.assp_frac4(x_signal)
        assp_origin = self.assp_origin(x_signal)
        assp1 = nn.functional.interpolate(assp1, size = 490)
        assp2 = nn.functional.interpolate(assp2, size = 490)
        assp3 = nn.functional.interpolate(assp3, size = 490)
        assp4 = nn.functional.interpolate(assp4, size = 490)
        assp_origin = nn.functional.interpolate(assp_origin, size = 490)

        x_signal = torch.cat([assp1,assp2,assp3,assp4,assp_origin],axis = 1)
        x_signal = self.down_sample(x_signal)

        assp_day1 = self.assp_frac_day1(x_day)
        assp_day2 = self.assp_frac_day2(x_day)
        assp_day3 = self.assp_frac_day3(x_day)
        assp_origin_day = self.assp_frac_day_origin(x_day)
        assp_day1 = nn.functional.interpolate(assp_day1, size = 10)
        assp_day2 = nn.functional.interpolate(assp_day2, size = 10)
        assp_day3 = nn.functional.interpolate(assp_day3, size = 10)
        assp_origin_day = nn.functional.interpolate(assp_origin_day, size = 10)


        x_day = torch.cat([assp_day1,assp_day2,assp_day3,assp_origin_day],axis = 1)
        x_day = self.down_sample_day(x_day)

        x_signal = self.dropout(x_signal)
        x_signal = self.up_sample(x_signal)
        x_day = self.dropout(x_day)
        x_day = self.up_sample_day(x_day)

        x_day = x_day + x_shortcut_day
        x_signal = x_signal + x_shortcut

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)
        x_signal = self.pointwise1(x_signal[:,:,-490:].permute(0,2,1)).squeeze(1)        
        x_signal = x_signal + x_day

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_25(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn1d_sxy_25,self).__init__()
        self.model_name = 'cnn1d_sxy_25: ordinary_3_avg_ordinary_3  -- largeFOV'
        
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5,padding = 2,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 49, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1, stride=1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()

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

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        x_signal = self.avgpool(x_signal)

        x_day = x_signal.reshape((x_signal.shape[0], x_signal.shape[1], 10, x_signal.shape[-1]// 10))
        x_day = torch.mean(x_day, axis = 3) 

        x_day = self.pointwise2(x_day.permute(0,2,1)).squeeze(1)
        x_signal = self.pointwise1(x_signal[:,:,-49:].permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out