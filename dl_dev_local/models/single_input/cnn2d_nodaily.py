from models.BasicModule import BasicModule
import torch.nn as nn
import torch

class cnn2d_sxy_1_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn2d_sxy_1_2,self).__init__()
        self.model_name = 'cnn2d_sxy_1_2: cnn_ordinary_ordinary_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.ordinary2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.pointwise = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride = 1)
        self.linear = nn.Linear(in_features= input_dim*20, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1)).permute(0,1,3,2)
        # x_signal = self.ordinary2(x_signal.unsqueeze(1))
        # x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal.contiguous().view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_2_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn2d_sxy_2_2,self).__init__()
        self.model_name = 'cnn2d_sxy_2_2: cnn_ordinary_ordinary_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.ordinary2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        # self.pointwise = nn.Conv2d(in_channels=15,out_channels=1,kernel_size=1,stride = 1)
        self.linear = nn.Linear(in_features= input_dim*20, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1)).permute(0,1,3,2)
        # x_signal = self.ordinary2(x_signal.unsqueeze(1))
        # x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal.contiguous().view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal
        x_signal = torch.mean(x_signal, axis = 2)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_3_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn2d_sxy_3_2,self).__init__()
        self.model_name = 'cnn2d_sxy_3_2: cnn_ordinary_ordinary_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=200, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.ordinary2 = nn.Conv2d(in_channels=1, out_channels=200, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.pointwise = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride = 1)
        self.linear = nn.Linear(in_features= input_dim*200, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1)).permute(0,1,3,2)
        # x_signal = self.ordinary2(x_signal.unsqueeze(1))
        # x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal.contiguous().view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_4_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn2d_sxy_4_2,self).__init__()
        self.model_name = 'cnn2d_sxy_4_2: cnn_ordinary_ordinary_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.ordinary2 = nn.Conv2d(in_channels=input_dim, out_channels=200, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.pointwise = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride = 1)
        self.linear = nn.Linear(in_features= 4000, out_features= 100)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1))

        x_signal = self.ordinary2(x_signal.permute(0,3,2,1)).permute(0,1,3,2)
        # x_signal = self.ordinary2(x_signal.unsqueeze(1))
        # x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal.contiguous().view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_5_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn2d_sxy_5_2,self).__init__()
        self.model_name = 'cnn2d_sxy_5_2: cnn_ordinary_ordinary_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.ordinary2 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool1 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool2 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool3 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.pointwise = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride = 1)
        self.linear = nn.Linear(in_features= 256*64, out_features= 256)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

        for p in self.ordinary1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary4.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1))

        x_signal = self.ordinary2(x_signal.permute(0,3,2,1))
        x_signal = self.avgpool1(x_signal)
        x_signal = self.ordinary3(x_signal)
        x_signal = self.avgpool2(x_signal)
        x_signal = self.ordinary4(x_signal)
        x_signal = self.avgpool3(x_signal)
        
        x_signal = x_signal.permute(0,1,3,2)
        # x_signal = self.ordinary2(x_signal.unsqueeze(1))
        # x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal.contiguous().view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_6_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn2d_sxy_6_2,self).__init__()
        self.model_name = 'cnn2d_sxy_6_2: cnn_ordinary_ordinary_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.ordinary2 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool1 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool2 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool3 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.pointwise = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride = 1)
        self.linear = nn.Linear(in_features= 256*64, out_features= 256)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

        for p in self.ordinary1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary4.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1))

        x_signal = self.ordinary2(x_signal.permute(0,3,2,1))

        x_signal = self.ordinary3(x_signal)

        x_signal = self.ordinary4(x_signal)

        
        x_signal = x_signal.permute(0,1,3,2)
        # x_signal = self.ordinary2(x_signal.unsqueeze(1))
        # x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal.contiguous().view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_7_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn2d_sxy_7_2,self).__init__()
        self.model_name = 'cnn2d_sxy_7_2: cnn_ordinary_ordinary_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')

        self.ordinary11 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')

        self.ordinary2 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool1 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool2 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool3 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.pointwise = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride = 1)
        self.linear = nn.Linear(in_features= 256*64, out_features= 256)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

        for p in self.ordinary1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary4.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1))
        x_signal = self.ordinary11(x_signal)

        x_signal = self.ordinary2(x_signal.permute(0,3,2,1))
        x_signal = self.avgpool1(x_signal)
        x_signal = self.ordinary3(x_signal)
        x_signal = self.avgpool2(x_signal)
        x_signal = self.ordinary4(x_signal)
        x_signal = self.avgpool3(x_signal)
        
        x_signal = x_signal.permute(0,1,3,2)
        # x_signal = self.ordinary2(x_signal.unsqueeze(1))
        # x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal.contiguous().view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_8_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn2d_sxy_8_2,self).__init__()
        self.model_name = 'cnn2d_sxy_8_2: cnn_ordinary_ordinary_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')

        self.ordinary11 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')

        self.ordinary2 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool1 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool2 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool3 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.pointwise = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride = 1)
        self.linear = nn.Linear(in_features= 256*64, out_features= 256)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

        for p in self.ordinary1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary4.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1))
        x_signal = self.ordinary11(x_signal)

        x_signal = self.ordinary2(x_signal.permute(0,3,2,1))

        x_signal = self.ordinary3(x_signal)

        x_signal = self.ordinary4(x_signal)
        
        x_signal = x_signal.permute(0,1,3,2)
        # x_signal = self.ordinary2(x_signal.unsqueeze(1))
        # x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal.contiguous().view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_9_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn2d_sxy_9_2,self).__init__()
        self.model_name = 'cnn2d_sxy_9_2: cnn_ordinary_ordinary_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')

        self.ordinary11 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')

        self.ordinary2 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool1 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool2 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool3 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.pointwise1 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride = 1)
        self.pointwise2 = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=1,stride = 1)
        self.pointwise3 = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=1,stride = 1) 

        self.linear = nn.Linear(in_features= 256*64, out_features= 256)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

        for p in self.ordinary1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary4.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1))
        x_signal = self.ordinary11(x_signal)

        x_signal = self.ordinary2(x_signal.permute(0,3,2,1))
        x_signal = self.avgpool1(x_signal)
        x_signal = self.ordinary3(x_signal)
        x_signal = self.avgpool2(x_signal)
        x_signal = self.ordinary4(x_signal)
        x_signal = self.avgpool3(x_signal)
        
        x_signal = x_signal.permute(0,1,3,2)
        # x_signal = self.ordinary2(x_signal.unsqueeze(1))
        # x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal.contiguous().view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal
        x_signal = self.pointwise1(x_signal.permute(0,2,1).unsqueeze(-1))
        x_signal = self.pointwise2(x_signal)
        x_signal = self.pointwise3(x_signal)
        
        
        x_signal = x_signal.squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_10_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn2d_sxy_10_2,self).__init__()
        self.model_name = 'cnn2d_sxy_10_2: cnn_ordinary_ordinary_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')

        self.ordinary11 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')

        self.ordinary2 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool1 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool2 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.ordinary4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool3 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)

        self.pointwise1 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride = 1)
        self.pointwise2 = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=1,stride = 1)
        self.pointwise3 = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=1,stride = 1) 

        self.linear = nn.Linear(in_features= 256*64, out_features= 256)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

        for p in self.ordinary1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary4.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1))
        x_signal = self.ordinary11(x_signal)

        x_signal = self.ordinary2(x_signal.permute(0,3,2,1))

        x_signal = self.ordinary3(x_signal)
 
        x_signal = self.ordinary4(x_signal)

        
        x_signal = x_signal.permute(0,1,3,2)
        # x_signal = self.ordinary2(x_signal.unsqueeze(1))
        # x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal.contiguous().view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal
        x_signal = self.pointwise1(x_signal.permute(0,2,1).unsqueeze(-1))
        x_signal = self.pointwise2(x_signal)
        x_signal = self.pointwise3(x_signal)
        
        
        x_signal = x_signal.squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_11_2(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.1):
        super(cnn2d_sxy_11_2,self).__init__()
        self.model_name = 'cnn2d_sxy_11_2: cnn_ordinary_ordinary_pointwise_1:waiting'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')

        self.ordinary11 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding = 1,stride = 1,padding_mode= 'replicate')

        self.ordinary2 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool1 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.ordinary3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool2 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.ordinary4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3,padding = 1,stride = 1,padding_mode= 'replicate')
        self.avgpool3 = nn.AvgPool2d(kernel_size=3,stride=1,padding = 1,count_include_pad= False)
        self.batchnorm3 = nn.BatchNorm2d(256)

        self.pointwise1 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,stride = 1)
        self.pointwise2 = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=1,stride = 1)
        self.pointwise3 = nn.Conv2d(in_channels=8,out_channels=1,kernel_size=1,stride = 1) 

        self.linear = nn.Linear(in_features= 256*64, out_features= 256)
        self.dropout = nn.Dropout(drop_rate)
        self.end_layer = nn.Linear(256,1)
        self.act = nn.ReLU()
        torch.set_num_threads(1)

        for p in self.ordinary1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.ordinary4.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1))
        x_signal = self.ordinary11(x_signal)

        x_signal = self.ordinary2(x_signal.permute(0,3,2,1))
        x_signal = self.avgpool1(x_signal)
        x_signal = self.batchnorm1(x_signal)

        x_signal = self.ordinary3(x_signal)
        x_signal = self.avgpool2(x_signal)
        x_signal = self.batchnorm2(x_signal)

        x_signal = self.ordinary4(x_signal)
        x_signal = self.avgpool3(x_signal)
        x_signal = self.batchnorm3(x_signal)

        x_signal = x_signal.permute(0,1,3,2)
        # x_signal = self.ordinary2(x_signal.unsqueeze(1))
        # x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal.contiguous().view(x_signal.shape[0],-1,x_signal.shape[-1])

        x_signal = x_signal
        x_signal = self.pointwise1(x_signal.permute(0,2,1).unsqueeze(-1))
        x_signal = self.pointwise2(x_signal)
        x_signal = self.pointwise3(x_signal)
        
        
        x_signal = x_signal.squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)

        x_signal = self.linear(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out



