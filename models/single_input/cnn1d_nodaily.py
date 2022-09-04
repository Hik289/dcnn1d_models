from models.BasicModule import BasicModule
import torch.nn as nn
import torch

from .cnn_model_sxy.bottleneck1d import bottleneck_1
from .cnn_model_sxy.bottleneck1d import bottleneck_2


class cnn1d_sxy_1_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_1_2,self).__init__()
        self.model_name = 'cnn1d_sxy_1_2: ordinary_2_pointwise'
        
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

        self.pointwise = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 128, out_features= 100)
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

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise(x_signal.permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_1_false_2(BasicModule):
    def __init__(self,kernel_size = 3,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_1_false_2,self).__init__()
        self.model_name = 'cnn1d_sxy_1_false_2: ordinary_2_pointwise_12avg_3kernel'

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=kernel_size,stride = 1,padding = kernel_size//2,padding_mode='replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size,padding = kernel_size//2,padding_mode='replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size,padding = kernel_size//2,padding_mode='replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.pointwise = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 128, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

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

        x_signal = self.pointwise(x_signal.permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_1_false12_2(BasicModule):
    def __init__(self,kernel_size = 5,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_1_false12_2,self).__init__()
        self.model_name = 'cnn1d_sxy_1_2: ordinary_2_pointwise_12avg_12kernel'

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=kernel_size,stride = 1,padding = kernel_size//2,padding_mode='replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size,padding = kernel_size//2,padding_mode='replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size,padding = kernel_size//2,padding_mode='replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.pointwise = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 128, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

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

        x_signal = self.pointwise(x_signal.permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_2_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_2_2,self).__init__()
        self.model_name = 'cnn1d_sxy_2_2: ordinary_3_ordinary_ordinary1'
        
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


class cnn1d_sxy_2_false_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_2_false_2,self).__init__()
        self.model_name = 'cnn1d_sxy_2_false_2: ordinary_3_ordinary_ordinary1_allpointwise'
        
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


class cnn1d_sxy_2_false32_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_2_false32_2,self).__init__()
        self.model_name = 'cnn1d_sxy_2_false32_2: ordinary_3_ordinary_ordinary1_allpointwise32'
        
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


class cnn1d_sxy_3_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_2,self).__init__()
        self.model_name = 'cnn1d_sxy_3_2: ordinary_3_ordinary_ordinary1_pointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1, stride = 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size= 1, stride = 1)
        self.act5 = nn.ReLU()

        self.pointwise = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

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

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)

        x_signal = self.pointwise(x_signal.permute(0,2,1)).squeeze(1)
        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_4_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_4_2,self).__init__()
        self.model_name = 'cnn1d_sxy_4_2: ordinary_3_avg_ordinary_3  -- largeFOV'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1, stride=1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
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

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        x_signal = self.avgpool(x_signal)

        x_day = self.pointwise2(x_signal.permute(0,2,1)).squeeze(1)
        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_5_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_5_2,self).__init__()
        self.model_name = 'cnn1d_sxy_5_2: ordinary_3_avg_ordinary_3  -- largeFOV'

        self.side1 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)) 

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1, stride=1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_side1 = self.side1(x.permute(0,2,1)).squeeze(-1)
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

        x_day = self.pointwise2(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = x_signal + x_side1
        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)
        
        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)


        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_6_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_6_2,self).__init__()
        self.model_name = 'cnn1d_sxy_6_2: -- deeplabV0'

        self.side1 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)) 

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.side2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)) 

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

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

        x_signal = x_signal+ x_side1 + x_side2

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)
        
        x_signal = self.dropout(x_signal)


        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out  


class cnn1d_sxy_7_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_7_2,self).__init__()
        self.model_name = 'cnn1d_sxy_7_2: -- deeplabV0'

        self.side1 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)) 

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.side2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)) 

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.side3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)) 

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

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

        x_signal = x_signal+ x_side1 + x_side2 + x_side3

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)
        
        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_8_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_8_2,self).__init__()
        self.model_name = 'cnn1d_sxy_8_2: -- deeplabV0'

        self.side1 = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)) 

        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.side2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)) 

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.side3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)) 

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.side4 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)) 

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

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

        x_signal = x_signal+ x_side1 + x_side2 + x_side3 + x_side4

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)
        
        x_signal = self.dropout(x_signal)


        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_9_false_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_9_false_2,self).__init__()
        self.model_name = 'cnn1d_sxy_9_false_2: bottleneck_1_pointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=1,stride = 1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=32, out_channels=6, kernel_size=1)
        self.batchnorm3 = nn.BatchNorm1d(6)
        self.act3 = nn.ReLU()

        self.shortcut = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=6, kernel_size=1, stride = 1),
                                      nn.BatchNorm1d(6))

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.linear = nn.Linear(in_features= 6, out_features= 10)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(10,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

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

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)
        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_9_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_9_2,self).__init__()
        self.model_name = 'cnn1d_sxy_9_2: bottleneck_2_pointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()

        self.cnn3 = nn.Conv1d(in_channels=32, out_channels=input_dim, kernel_size=1)
        self.batchnorm3 = nn.BatchNorm1d(input_dim)
        self.act3 = nn.ReLU()

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.linear = nn.Linear(in_features= input_dim, out_features= 10)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(10,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

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

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        x_signal = self.dropout(x_signal)
        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_10_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_10_2,self).__init__()
        self.model_name = 'cnn1d_sxy_10_2: -- DeeplabV2:0 -- 20'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_11_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_11_2,self).__init__()
        self.model_name = 'cnn1d_sxy_11_2: -- DeeplabV2:1'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim=128, output_dim=256)
        self.bottleneck1_day = bottleneck_1(input_dim = 128, output_dim= 256)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 256, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_12_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_12_2,self).__init__()
        self.model_name = 'cnn1d_sxy_12_2: -- DeeplabV2:1'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.bottleneck1 = bottleneck_2(input_dim=128, output_dim=256)
        self.bottleneck1_day = bottleneck_2(input_dim = 128, output_dim= 256)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.bottleneck1(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)
        
        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out   


class cnn1d_sxy_13_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_13_2,self).__init__()
        self.model_name = 'cnn1d_sxy_13_2: -- DeeplabV2:1'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2 = bottleneck_2(input_dim=128, output_dim=256)
        self.bottleneck1_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2_day = bottleneck_2(input_dim = 128, output_dim= 256)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_14_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_14_2,self).__init__()
        self.model_name = 'cnn1d_sxy_14_2: -- DeeplabV2:1 no avgpool'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
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

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_14_cat_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_14_cat_2,self).__init__()
        self.model_name = 'cnn1d_sxy_14_cat_2: -- DeeplabV2:1 no avgpool cat'

        self.cnn0 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm0 = nn.BatchNorm1d(64)
        self.act0 = nn.ReLU()
        # self.avgpool0 = nn.AvgPool1d(kernel_size=11,stride=1,padding = 5,count_include_pad= False)
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
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

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.pointwise2 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_day = x.reshape((x.shape[0], 8, x.shape[1]//8, x.shape[-1]))
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

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)
        
        x_signal = torch.cat([x_signal, x_day], axis = 1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_14_2_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_14_2_2,self).__init__()
        self.model_name = 'cnn1d_sxy_14_2_2: -- DeeplabV2:1+'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2 = bottleneck_2(input_dim=128, output_dim=128)
        self.bottleneck1_day = bottleneck_1(input_dim = 64, output_dim = 128)
        self.bottleneck2_day = bottleneck_2(input_dim = 128, output_dim= 128)

        self.bottleneck3 = bottleneck_1(input_dim = 128, output_dim = 256)
        self.bottleneck4 = bottleneck_2(input_dim=256, output_dim=256)
        self.bottleneck3_day = bottleneck_1(input_dim = 128, output_dim = 256)
        self.bottleneck4_day = bottleneck_2(input_dim = 256, output_dim= 256)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_15_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_15_2,self).__init__()
        self.model_name = 'cnn1d_sxy_15_2: -- DeeplabV2:FINAL'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

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

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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

        assp1 = nn.functional.interpolate(assp1, size = 32)
        assp2 = nn.functional.interpolate(assp2, size = 32)
        assp3 = nn.functional.interpolate(assp3, size = 32)
        assp4 = nn.functional.interpolate(assp4, size = 32)
        x_signal = assp1 + assp2 + assp3 + assp4  

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_16_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_16_2,self).__init__()
        self.model_name = 'cnn1d_sxy_16_2: -- DeeplabV2:assp'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

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

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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

        assp1 = nn.functional.interpolate(assp1, size = 32)
        assp2 = nn.functional.interpolate(assp2, size = 32)
        assp3 = nn.functional.interpolate(assp3, size = 32)
        assp4 = nn.functional.interpolate(assp4, size = 32)
        x_signal = assp1 + assp2 + assp3 + assp4  

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_17_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_17_2,self).__init__()
        self.model_name = 'cnn1d_sxy_17_2: -- DeeplabV2:ASSP+interpolate no pointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

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

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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

        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_18_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_18_2,self).__init__()
        self.model_name = 'cnn1d_sxy_18_2: -- DeeplabV2:ASSP+interpolate no pointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

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

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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

        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_19_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_19_2,self).__init__()
        self.model_name = 'cnn1d_sxy_19_2: -- DeeplabV2:FINAL'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

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

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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
        assp1 = nn.functional.interpolate(assp1, size = 32)
        assp2 = nn.functional.interpolate(assp2, size = 32)
        assp3 = nn.functional.interpolate(assp3, size = 32)
        assp4 = nn.functional.interpolate(assp4, size = 32)
        assp_origin = nn.functional.interpolate(assp_origin, size = 32)
        x_signal = assp1 + assp2 + assp3 + assp4 + assp_origin 

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)        

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_20_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_20_2,self).__init__()
        self.model_name = 'cnn1d_sxy_20: -- DeeplabV3:1'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

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

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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
        assp1 = nn.functional.interpolate(assp1, size = 32)
        assp2 = nn.functional.interpolate(assp2, size = 32)
        assp3 = nn.functional.interpolate(assp3, size = 32)
        assp4 = nn.functional.interpolate(assp4, size = 32)
        assp_origin = nn.functional.interpolate(assp_origin, size = 32)

        x_signal = torch.cat([assp1,assp2,assp3,assp4,assp_origin],axis = 1)
        x_signal = self.down_sample(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)        

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_21_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_21_2,self).__init__()
        self.model_name = 'cnn1d_sxy_21_2: -- DeeplabV3:1 no pointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

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

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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

        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_22_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_22_2,self).__init__()
        self.model_name = 'cnn1d_sxy_22_2: -- assp:0 no pointwise -- light model'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3,padding = 1, stride = 1, padding_mode='replicate')
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

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 64, out_features= 10)

        self.end_layer = nn.Linear(10,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

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


class cnn1d_sxy_23_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_23_2,self).__init__()
        self.model_name = 'cnn1d_sxy_23_2: -- DeeplabV3:3'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

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

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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
        assp1 = nn.functional.interpolate(assp1, size = 32)
        assp2 = nn.functional.interpolate(assp2, size = 32)
        assp3 = nn.functional.interpolate(assp3, size = 32)
        assp4 = nn.functional.interpolate(assp4, size = 32)
        assp_origin = nn.functional.interpolate(assp_origin, size = 32)

        x_signal = torch.cat([assp1,assp2,assp3,assp4,assp_origin],axis = 1)
        x_signal = self.down_sample(x_signal)

        x_signal = self.dropout(x_signal)
        x_signal = self.up_sample(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)        

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_24_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_24_2,self).__init__()
        self.model_name = 'cnn1d_sxy_24_2: -- DeeplabV3:FINAL'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.bottleneck1 = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2 = bottleneck_2(input_dim=64, output_dim=64)
        self.bottleneck1_day = bottleneck_1(input_dim = 32, output_dim = 64)
        self.bottleneck2_day = bottleneck_2(input_dim = 64, output_dim= 64)

        self.shortcut = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate'),
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

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features= 128*2, out_features= 100)

        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

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
        assp1 = nn.functional.interpolate(assp1, size = 32)
        assp2 = nn.functional.interpolate(assp2, size = 32)
        assp3 = nn.functional.interpolate(assp3, size = 32)
        assp4 = nn.functional.interpolate(assp4, size = 32)
        assp_origin = nn.functional.interpolate(assp_origin, size = 32)

        x_signal = torch.cat([assp1,assp2,assp3,assp4,assp_origin],axis = 1)
        x_signal = self.down_sample(x_signal)

        x_signal = self.dropout(x_signal)
        x_signal = self.up_sample(x_signal)

        x_signal = x_signal + x_shortcut

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)        

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_25_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_25_2,self).__init__()
        self.model_name = 'cnn1d_sxy_25_2: ordinary_3_avg_ordinary_3  -- largeFOV'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        self.linear = nn.Linear(in_features= 256, out_features= 100)
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

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        x_signal = self.avgpool(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)
        
        x_signal = self.dropout(x_signal)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out