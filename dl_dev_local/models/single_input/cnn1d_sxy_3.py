from models.BasicModule import BasicModule
import torch.nn as nn
import torch

class cnn1d_sxy_3_1(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_1,self).__init__()
        self.model_name = 'cnn1d_sxy_3_1: 5kernelbase'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5,stride = 1,padding = 2, padding_mode='replicate')

        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn_timeline = nn.Conv1d(in_channels= 15, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))

        x_signal = self.cnn2(x_signal)

        x_signal = self.cnn3(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn_timeline(x_signal[:,:,-15:].permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_2,self).__init__()
        self.model_name = 'cnn1d_sxy_3_2: 5kernelbase+CNN1d*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))

        x_signal = self.cnn2(x_signal)

        x_signal = self.cnn3(x_signal)

        x_signal = self.cnn4(x_signal)

        x_signal = self.cnn5(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_3(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_3,self).__init__()
        self.model_name = 'cnn1d_sxy_3_3: 5kernelbase+CNN1d*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))

        x_signal = self.cnn2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_4(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_4,self).__init__()
        self.model_name = 'cnn1d_sxy_3_4: 5kernelbase+CNN1d*2(stride = kernel)'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 3, padding = 2, padding_mode='replicate')

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 3)

        self.cnn_timeline = nn.Conv1d(in_channels= 4, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))

        x_signal = self.cnn2(x_signal)

        x_signal = self.cnn3(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_5(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.5):
        super(cnn1d_sxy_3_5,self).__init__()
        self.model_name = 'cnn1d_sxy_3_5: ordinary_2_pointwise'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5,stride = 1,padding = 2,padding_mode='replicate')
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,padding = 2,padding_mode='replicate')
        self.act2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,padding_mode='replicate')
        self.act3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad=False)

        self.pointwise = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.linear = nn.Linear(in_features= 128, out_features= 100)
        self.dropout = nn.Dropout(0.1)
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


class cnn1d_sxy_3_6(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_6,self).__init__()
        self.model_name = 'cnn1d_sxy_3_6: 5kernelbase+CNN1d*2(stride = kernel)'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 3, padding = 2, padding_mode='replicate')

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 3)

        self.cnn_timeline = nn.Conv1d(in_channels= 4, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)


        x_signal = self.cnn4(x_signal)


        x_signal = self.cnn5(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_7(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_7,self).__init__()
        self.model_name = 'cnn1d_sxy_3_7: 5kernelbase+CNN1d*2+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 1, stride=1, padding=0)
        self.cnn_timeline_2 = nn.Conv1d(in_channels=32, out_channels= 32, kernel_size= 1, stride=1, padding= 0)
        self.cnn_timeline_3 = nn.Conv1d(in_channels=32, out_channels= 1, kernel_size= 1, stride=1, padding= 0)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))

        x_signal = self.cnn2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1))
        x_signal = self.cnn_timeline_2(x_signal)
        x_signal = self.cnn_timeline_3(x_signal).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_8(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_8,self).__init__()
        self.model_name = 'cnn1d_sxy_3_8: 5kernelbase+CNN1d*2+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 1, stride=1, padding=0)
        self.cnn_timeline_2 = nn.Conv1d(in_channels=32, out_channels= 32, kernel_size= 1, stride=1, padding= 0)
        self.cnn_timeline_3 = nn.Conv1d(in_channels=32, out_channels= 1, kernel_size= 1, stride=1, padding= 0)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))

        x_signal = self.cnn2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm3(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1))
        x_signal = self.cnn_timeline_2(x_signal)
        x_signal = self.cnn_timeline_3(x_signal).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_9(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_9,self).__init__()
        self.model_name = 'cnn1d_sxy_3_9: 5kernelbase+CNN1d*2+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 1, stride=1, padding=0)
        self.cnn_timeline_2 = nn.Conv1d(in_channels=32, out_channels= 32, kernel_size= 1, stride=1, padding= 0)
        self.cnn_timeline_3 = nn.Conv1d(in_channels=32, out_channels= 1, kernel_size= 1, stride=1, padding= 0)

        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))

        x_signal = self.cnn2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm3(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1))
        x_signal = self.cnn_timeline_2(x_signal)
        x_signal = self.cnn_timeline_3(x_signal).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_10(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_10,self).__init__()
        self.model_name = 'cnn1d_sxy_3_10: 5kernelbase+CNN1d*2+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 1, stride=1, padding=0)
        self.cnn_timeline_2 = nn.Conv1d(in_channels=32, out_channels= 32, kernel_size= 1, stride=1, padding= 0)
        self.cnn_timeline_3 = nn.Conv1d(in_channels=32, out_channels= 1, kernel_size= 1, stride=1, padding= 0)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))

        x_signal = self.cnn2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm3(x_signal)

        # x_signal = self.cnn_timeline(x_signal.permute(0,2,1))
        # x_signal = self.cnn_timeline_2(x_signal)
        x_signal = self.cnn_timeline_3(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_11(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_11,self).__init__()
        self.model_name = 'cnn1d_sxy_3_11: 5kernelbase+CNN1d*2+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)

        self.cnn_timeline = nn.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 1, stride=1, padding=0)
        self.cnn_timeline_2 = nn.Conv1d(in_channels=32, out_channels= 32, kernel_size= 1, stride=1, padding= 0)
        self.cnn_timeline_3 = nn.Conv1d(in_channels=32, out_channels= 1, kernel_size= 1, stride=1, padding= 0)

        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))

        x_signal = self.cnn2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline(x_signal.permute(0,2,1))
        x_signal = self.cnn_timeline_2(x_signal)
        x_signal = self.cnn_timeline_3(x_signal).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_12(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_12,self).__init__()
        self.model_name = 'cnn1d_sxy_3_12: 5kernelbase+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))

        x_signal = self.cnn2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm2(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm3(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_13(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_13,self).__init__()
        self.model_name = 'cnn1d_sxy_3_13: 5kernelbase+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_14(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_14,self).__init__()
        self.model_name = 'cnn1d_sxy_3_14: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_15(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_15,self).__init__()
        self.model_name = 'cnn1d_sxy_3_15: 5kernelbase+(CNN1d*3)*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_16(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_16,self).__init__()
        self.model_name = 'cnn1d_sxy_3_16: 5kernelbase+(CNN1d*3)*2+down*2+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256) 

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128)           

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_2,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_2: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_3(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_3,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_3: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.batchnorm1 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.avgpool1(x_signal)
        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_4(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_4,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_4: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.batchnorm1 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)

        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.avgpool1(x_signal)
        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)

        x_signal = self.avgpool2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)

        x_signal = self.avgpool3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_5(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_5,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_5: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')

        self.batchnorm1 = nn.BatchNorm1d(256)

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)

        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.drop = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.avgpool1(x_signal)
        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)

        x_signal = self.avgpool2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)

        x_signal = self.avgpool3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_18(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_18,self).__init__()
        self.model_name = 'cnn1d_sxy_3_18: ordinary_3_avg_ordinary_3  -- largeFOV'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        #self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        #self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        #self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

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

        #x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        #x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        #x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        x_signal = self.avgpool(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_19(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_19,self).__init__()
        self.model_name = 'cnn1d_sxy_3_19: ordinary_3_avg_ordinary_3  -- largeFOV'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        #self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        #self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        #self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        #self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.act1(x_signal)

        #x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        x_signal = self.act2(x_signal)
        #x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        #x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.act4(x_signal)

        x_signal = self.cnn5(x_signal)
        x_signal = self.act5(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        x_signal = self.avgpool(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)
        
        #x_signal = self.dropout(x_signal)

        #out = self.linear(x_signal)
        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_19_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_19_2,self).__init__()
        self.model_name = 'cnn1d_sxy_3_19_2: ordinary_3_avg_ordinary_3  -- largeFOV'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act1 = nn.ReLU()
        #self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act2 = nn.ReLU()
        #self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.act3 = nn.ReLU()
        #self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size= 1)
        self.act4 = nn.ReLU()

        self.cnn5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size= 1)
        self.act5 = nn.ReLU()

        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size= 1)
        self.act6 = nn.ReLU()

        #self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.pointwise1 = nn.Conv1d(in_channels= 32, out_channels=1, kernel_size=1, stride = 1)

        #self.linear = nn.Linear(in_features= 256, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        #x_signal = self.act1(x_signal)

        #x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn2(x_signal)
        #x_signal = self.act2(x_signal)
        #x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn3(x_signal)
        x_signal = self.act3(x_signal)
        #x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn4(x_signal)
        #x_signal = self.act4(x_signal)

        x_signal = self.cnn5(x_signal)
        #x_signal = self.act5(x_signal)

        #x_signal = self.dropout(x_signal)

        x_signal = self.cnn6(x_signal)
        x_signal = self.act6(x_signal)

        #x_signal = self.avgpool(x_signal)

        x_signal = self.pointwise1(x_signal.permute(0,2,1)).squeeze(1)
        
        #x_signal = self.dropout(x_signal)

        #out = self.linear(x_signal)
        # out = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_20(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_20,self).__init__()
        self.model_name = 'cnn1d_sxy_3_20: 5kernelbase+(CNN1d*3)*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)    

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.avgpool(x_signal)
        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_21(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_21,self).__init__()
        self.model_name = 'cnn1d_sxy_3_21: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.avgpool = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.avgpool(x_signal)
        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out

        
class cnn1d_sxy_3_22(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_22,self).__init__()
        self.model_name = 'cnn1d_sxy_3_22: 5kernelbase+(CNN1d*3)*2+down*2+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256) 

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128)           

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_23(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_23,self).__init__()
        self.model_name = 'cnn1d_sxy_3_23: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_24(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_24,self).__init__()
        self.model_name = 'cnn1d_sxy_3_24: 5kernelbase+(CNN1d*3)*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_25(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_25,self).__init__()
        self.model_name = 'cnn1d_sxy_3_25: 5kernelbase+(CNN1d*3)*2+down*2+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256) 

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128)           

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_normal(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_normal,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_normal'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()

        for p in self.cnn1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.cnn2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.cnn3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.cnn4.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.cnn5.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.cnn6.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.cnn_timeline_1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_uniform(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_uniform,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_uniform'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()

        for p in self.cnn1.parameters():
            nn.init.uniform_(p)
        for p in self.cnn2.parameters():
            nn.init.uniform_(p)
        for p in self.cnn3.parameters():
            nn.init.uniform_(p)
        for p in self.cnn4.parameters():
            nn.init.uniform_(p)
        for p in self.cnn5.parameters():
            nn.init.uniform_(p)
        for p in self.cnn6.parameters():
            nn.init.uniform_(p)

        for p in self.cnn_timeline_1.parameters():
            nn.init.uniform_(p)
        for p in self.end_layer.parameters():
            nn.init.uniform_(p)

        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_xavier_uniform(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_xavier_uniform,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_xavier_uniform'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()

        for p in self.cnn1.parameters():
            nn.init.xavier_uniform_(p)
        for p in self.cnn2.parameters():
            nn.init.xavier_uniform_(p)
        for p in self.cnn3.parameters():
            nn.init.xavier_uniform_(p)
        for p in self.cnn4.parameters():
            nn.init.xavier_uniform_(p)
        for p in self.cnn5.parameters():
            nn.init.xavier_uniform_(p)
        for p in self.cnn6.parameters():
            nn.init.xavier_uniform_(p)

        for p in self.cnn_timeline_1.parameters():
            nn.init.xavier_uniform_(p)
        for p in self.end_layer.parameters():
            nn.init.xavier_uniform_(p)

        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_xavier_normal(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_xavier_normal,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_xavier_normal'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()

        for p in self.cnn1.parameters():
            nn.init.xavier_normal_(p)
        for p in self.cnn2.parameters():
            nn.init.xavier_normal_(p)
        for p in self.cnn3.parameters():
            nn.init.xavier_normal_(p)
        for p in self.cnn4.parameters():
            nn.init.xavier_normal_(p)
        for p in self.cnn5.parameters():
            nn.init.xavier_normal_(p)
        for p in self.cnn6.parameters():
            nn.init.xavier_normal_(p)

        for p in self.cnn_timeline_1.parameters():
            nn.init.xavier_normal_(p)
        for p in self.end_layer.parameters():
            nn.init.xavier_normal_(p)

        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_kaiming_uniform(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_kaiming_uniform,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_kaiming_uniform'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate',bias=False)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate',bias = False)
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate',bias = False)
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate',bias = False)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate',bias = False)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate',bias = False)
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0,bias = False)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()

        for p in self.cnn1.parameters():
            nn.init.kaiming_uniform_(p)
        for p in self.cnn2.parameters():
            nn.init.kaiming_uniform_(p)
        for p in self.cnn3.parameters():
            nn.init.kaiming_uniform_(p)
        for p in self.cnn4.parameters():
            nn.init.kaiming_uniform_(p)
        for p in self.cnn5.parameters():
            nn.init.kaiming_uniform_(p)
        for p in self.cnn6.parameters():
            nn.init.kaiming_uniform_(p)

        for p in self.cnn_timeline_1.parameters():
            nn.init.kaiming_uniform_(p)
        # for p in self.end_layer.parameters():
        #     nn.init.kaiming_uniform_(p)

        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_kaiming_normal(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_kaiming_normal,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_kaiming_normal'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate',bias = False)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate',bias = False)
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate',bias = False)
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate',bias = False)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate',bias = False)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate',bias = False)
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0,bias = False)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()

        for p in self.cnn1.parameters():
            nn.init.kaiming_normal_(p)
        for p in self.cnn2.parameters():
            nn.init.kaiming_normal_(p)
        for p in self.cnn3.parameters():
            nn.init.kaiming_normal_(p)
        for p in self.cnn4.parameters():
            nn.init.kaiming_normal_(p)
        for p in self.cnn5.parameters():
            nn.init.kaiming_normal_(p)
        for p in self.cnn6.parameters():
            nn.init.kaiming_normal_(p)

        for p in self.cnn_timeline_1.parameters():
            nn.init.kaiming_normal_(p)
        # for p in self.end_layer.parameters():
        #     nn.init.kaiming_normal_(p)

        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_orthogonal(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_orthogonal,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_orthogonal'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()

        for p in self.cnn1.parameters():
            nn.init.orthogonal_(p)
        for p in self.cnn2.parameters():
            nn.init.orthogonal_(p)
        for p in self.cnn3.parameters():
            nn.init.orthogonal_(p)
        for p in self.cnn4.parameters():
            nn.init.orthogonal_(p)
        for p in self.cnn5.parameters():
            nn.init.orthogonal_(p)
        for p in self.cnn6.parameters():
            nn.init.orthogonal_(p)

        for p in self.cnn_timeline_1.parameters():
            nn.init.orthogonal_(p)
        for p in self.end_layer.parameters():
            nn.init.orthogonal_(p)

        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_constant(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_constant,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_constant'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()

        for p in self.cnn1.parameters():
            nn.init.constant_(p,1.2)
        for p in self.cnn2.parameters():
            nn.init.constant_(p,1.2)
        for p in self.cnn3.parameters():
            nn.init.constant_(p,1.2)
        for p in self.cnn4.parameters():
            nn.init.constant_(p,1.2)
        for p in self.cnn5.parameters():
            nn.init.constant_(p,1.2)
        for p in self.cnn6.parameters():
            nn.init.constant_(p,1.2)

        for p in self.cnn_timeline_1.parameters():
            nn.init.constant_(p,1.2)
        for p in self.end_layer.parameters():
            nn.init.constant_(p,1.2)

        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_sparse(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_sparse,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_sparse'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()

        for p in self.cnn1.parameters():
            nn.init.sparse_(p,sparsity= 0.5)
        for p in self.cnn2.parameters():
            nn.init.sparse_(p,sparsity= 0.5)
        for p in self.cnn3.parameters():
            nn.init.sparse_(p,sparsity= 0.5)
        for p in self.cnn4.parameters():
            nn.init.sparse_(p,sparsity= 0.5)
        for p in self.cnn5.parameters():
            nn.init.sparse_(p,sparsity= 0.5)
        for p in self.cnn6.parameters():
            nn.init.sparse_(p,sparsity= 0.5)

        for p in self.cnn_timeline_1.parameters():
            nn.init.sparse_(p,sparsity= 0.5)
        for p in self.end_layer.parameters():
            nn.init.sparse_(p,sparsity= 0.5)

        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_zeros(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_zeros,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_zeros'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()

        for p in self.cnn1.parameters():
            nn.init.zeros_(p)
        for p in self.cnn2.parameters():
            nn.init.zeros_(p)
        for p in self.cnn3.parameters():
            nn.init.zeros_(p)
        for p in self.cnn4.parameters():
            nn.init.zeros_(p)
        for p in self.cnn5.parameters():
            nn.init.zeros_(p)
        for p in self.cnn6.parameters():
            nn.init.zeros_(p)

        for p in self.cnn_timeline_1.parameters():
            nn.init.zeros_(p)
        for p in self.end_layer.parameters():
            nn.init.zeros_(p)

        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_eye(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_eye,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_eye'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.ReLU()

        for p in self.cnn1.parameters():
            nn.init.eye_(p)
        for p in self.cnn2.parameters():
            nn.init.eye_(p)
        for p in self.cnn3.parameters():
            nn.init.eye_(p)
        for p in self.cnn4.parameters():
            nn.init.eye_(p)
        for p in self.cnn5.parameters():
            nn.init.eye_(p)
        for p in self.cnn6.parameters():
            nn.init.eye_(p)

        for p in self.cnn_timeline_1.parameters():
            nn.init.eye_(p)
        for p in self.end_layer.parameters():
            nn.init.eye_(p)

        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_Leakyrelu(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_Leakyrelu,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_Leakyrelu: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.LeakyReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_Sigmoid(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_Sigmoid,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_Sigmoid: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.Sigmoid()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_Tanh(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_Tanh,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_Tanh: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.Tanh()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_Softmax(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_Softmax,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_Softmax: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act = nn.Softmax()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.act(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_Softmax1(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_Softmax1,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_Softmax1: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Softmax(dim = 0)
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act2(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_17_Softmax2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_17_Softmax2,self).__init__()
        self.model_name = 'cnn1d_sxy_3_17_Softmax2: 5kernelbase+CNN1d*3+CNN1d*3+timeline*2'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.cnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.cnn6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(512)

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(512,1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Softmax(dim = 0)
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        # x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        # x_signal = self.act1(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act2(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_26(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_26,self).__init__()
        self.model_name = 'cnn1d_sxy_3_26: backbone+avg5*1+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_27(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_27,self).__init__()
        self.model_name = 'cnn1d_sxy_3_27: backbone+avg3*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)


        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_28(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_28,self).__init__()
        self.model_name = 'cnn1d_sxy_3_28: backbone+avg3*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.avgpool4(x_signal)


        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_29(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_29,self).__init__()
        self.model_name = 'cnn1d_sxy_3_29: backbone+avg5*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_30(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_30,self).__init__()
        self.model_name = 'cnn1d_sxy_3_30: backbone+avg5*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.avgpool4(x_signal)


        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)


        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_31(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_31,self).__init__()
        self.model_name = 'cnn1d_sxy_3_31: backbone+avg5*4+relu'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_32(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_32,self).__init__()
        self.model_name = 'cnn1d_sxy_3_32: backbone+avg5*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.avgpool4(x_signal)


        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)


        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_33(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_33,self).__init__()
        self.model_name = 'cnn1d_sxy_3_33: backbone+avg5*4+relu'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_34(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_34,self).__init__()
        self.model_name = 'cnn1d_sxy_3_34: backbone+avg5*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.avgpool4(x_signal)


        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_35(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_35,self).__init__()
        self.model_name = 'cnn1d_sxy_3_35: backbone+avg5*1+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=4,stride=4,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 8, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_36(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_36,self).__init__()
        self.model_name = 'cnn1d_sxy_3_36: backbone+avg3*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_37(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_37,self).__init__()
        self.model_name = 'cnn1d_sxy_3_37: backbone+avg3*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.avgpool4(x_signal)


        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_38(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_38,self).__init__()
        self.model_name = 'cnn1d_sxy_3_38: backbone+avg5*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_39(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_39,self).__init__()
        self.model_name = 'cnn1d_sxy_3_39: backbone+avg5*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.dropout(x_signal)

        x_signal = x_signal.squeeze(2)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_40(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_40,self).__init__()
        self.model_name = 'cnn1d_sxy_3_40: backbone+avg5*4+relu'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_41(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_41,self).__init__()
        self.model_name = 'cnn1d_sxy_3_41: backbone+avg5*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.avgpool4(x_signal)


        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = x_signal.squeeze(2)

        x_signal = self.dropout(x_signal)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_42(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_42,self).__init__()
        self.model_name = 'cnn1d_sxy_3_42: backbone+avg5*4+relu'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=3,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 3, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_43(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_43,self).__init__()
        self.model_name = 'cnn1d_sxy_3_43: backbone+avg5*4+dropout'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 2,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_44(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_44,self).__init__()
        self.model_name = 'cnn1d_sxy_3_44: backbone+avg5*4+relu'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.avgpool4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_45(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_45,self).__init__()
        self.model_name = 'cnn1d_sxy_3_45: backbone+avg5*4+relu'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=5,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        # x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        # x_signal = self.avgpool4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = x_signal.squeeze(2)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_46(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_46,self).__init__()
        self.model_name = 'cnn1d_sxy_3_46: backbone+avg5*4+relu'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=4,stride=4,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=4,stride=4,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 2, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.avgpool1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.avgpool2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)

        x_signal = self.act3(x_signal)

        # x_signal = self.avgpool3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)

        x_signal = self.act4(x_signal)

        # x_signal = self.avgpool4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_47(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_47,self).__init__()
        self.model_name = 'cnn1d_sxy_3_47: backbone+avg5*4+relu'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=5,stride=1,padding = 2,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.avgpool4(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn1d_sxy_3_48(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn1d_sxy_3_48,self).__init__()
        self.model_name = 'cnn1d_sxy_3_48: backbone+avg5*4+relu'
        
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding = 2, padding_mode='replicate')
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.avgpool1 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.cnn5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.cnn6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()

        self.avgpool2 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.cnn8 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm6 = nn.BatchNorm1d(512)
        self.cnn9 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm7 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()

        self.avgpool3 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn10 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm8 = nn.BatchNorm1d(256)
        self.cnn11 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm9 = nn.BatchNorm1d(256)
        self.cnn12 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm10 = nn.BatchNorm1d(256)
        self.act4 = nn.ReLU() 

        self.avgpool4 = nn.AvgPool1d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.cnn13 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride = 1, padding = 1, padding_mode='replicate')
        self.batchnorm11 = nn.BatchNorm1d(128)
        self.cnn14 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm12 = nn.BatchNorm1d(128)
        self.cnn15 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride = 1, padding = 0, padding_mode='replicate')
        self.batchnorm13 = nn.BatchNorm1d(128) 
        self.act5 = nn.ReLU()  

        self.dropout = nn.Dropout(drop_rate)        

        self.cnn_timeline_1 = nn.Conv1d(in_channels= 32, out_channels= 1, kernel_size= 1, stride=1, padding=0)

        self.end_layer = nn.Linear(128,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

    def forward(self,x):

        x_signal = self.cnn1(x.permute(0,2,1))
        x_signal = self.cnn2(x_signal)
        x_signal = self.cnn3(x_signal)
        x_signal = self.batchnorm1(x_signal)
        x_signal = self.avgpool1(x_signal)

        x_signal = self.act1(x_signal)

        x_signal = self.cnn4(x_signal)
        x_signal = self.batchnorm2(x_signal)
        x_signal = self.cnn5(x_signal)
        x_signal = self.batchnorm3(x_signal)
        x_signal = self.cnn6(x_signal)
        x_signal = self.batchnorm4(x_signal)
        x_signal = self.avgpool2(x_signal)

        x_signal = self.act2(x_signal)

        x_signal = self.cnn7(x_signal)
        x_signal = self.batchnorm5(x_signal)
        x_signal = self.cnn8(x_signal)
        x_signal = self.batchnorm6(x_signal)
        x_signal = self.cnn9(x_signal)
        x_signal = self.batchnorm7(x_signal)
        x_signal = self.avgpool3(x_signal)

        x_signal = self.act3(x_signal)

        x_signal = self.cnn10(x_signal)
        x_signal = self.batchnorm8(x_signal)
        x_signal = self.cnn11(x_signal)
        x_signal = self.batchnorm9(x_signal)
        x_signal = self.cnn12(x_signal)
        x_signal = self.batchnorm10(x_signal)
        x_signal = self.avgpool4(x_signal)

        x_signal = self.act4(x_signal)

        x_signal = self.cnn13(x_signal)
        x_signal = self.batchnorm11(x_signal)
        x_signal = self.cnn14(x_signal)
        x_signal = self.batchnorm12(x_signal)
        x_signal = self.cnn15(x_signal)
        x_signal = self.batchnorm13(x_signal)

        x_signal = self.act5(x_signal)

        x_signal = self.cnn_timeline_1(x_signal.permute(0,2,1)).squeeze(1)

        out = self.act(x_signal)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


