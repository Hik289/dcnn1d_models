from models.BasicModule import BasicModule
import torch.nn as nn
import torch

class cnn_test(BasicModule):
    def __init__(self,input_dim=12,drop_rate=0.5):
        super(cnn_test,self).__init__()
        self.model_name = 'cnn_test_simplest_20'
        
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=20, kernel_size=3,padding = 1)
        torch.set_num_threads(1)
        self.linear = nn.Linear(in_features= 20, out_features= 20)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.cnn(x.permute(0,2,1))
        x_signal = x_signal[:,:,-1]
        #out = self.dropout(x_signal)
        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn_ordinary_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_ordinary_1,self).__init__()
        self.model_name = 'cnn_ordinary_1'
        
        self.cnn = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)

        self.linear = nn.Linear(in_features= 20, out_features= 20)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.cnn(x.unsqueeze(1)).squeeze(-1)
        x_signal = x_signal[:,:,-1]
        #out = self.dropout(x_signal)
        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn_pointwise_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_pointwise_1,self).__init__()
        self.model_name = 'cnn_pointwise_1'
        
        self.cnn = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=1)

        self.linear = nn.Linear(in_features= 20, out_features= 20)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.cnn(x.permute(0,2,1).unsqueeze(-1)).squeeze(-1)
        x_signal = torch.mean(x_signal[:,:,-5:],axis = 2)
        #out = self.dropout(x_signal)
        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out

class cnn_ordinary_pointwise_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_ordinary_pointwise_1,self).__init__()
        self.model_name = 'cnn_deepwise_pointwise_1'
        
        self.ordinary = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.linear = nn.Linear(in_features= 20, out_features= 20)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary(x.unsqueeze(1)).squeeze(-1)
        x_signal = x_signal[:,:,-5:]
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)
        #out = self.dropout(x_signal)
        out = self.linear(x_signal)
        out = self.act(out)

        out = self.end_layer(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn_ordinary_max_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_ordinary_max_1,self).__init__()
        self.model_name = 'cnn_ordinary_max_1'
        
        self.cnn = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.max = nn.MaxPool1d(kernel_size=3)
        self.linear = nn.Linear(in_features= 20, out_features= 20)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.cnn(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.max(x_signal)
        x_signal = x_signal[:,:,-1]

        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear(x_signal)
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn_ordinary_max_pointwise_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_ordinary_max_pointwise_1,self).__init__()
        self.model_name = 'cnn_deepwise_max_pointwise_1'
        
        self.ordinary = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)

        self.max = nn.MaxPool1d(kernel_size=3)
        self.pointwise = nn.Conv2d(in_channels=9,out_channels=1,kernel_size=1)
        self.linear = nn.Linear(in_features= 20, out_features= 20)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary(x.unsqueeze(1)).squeeze(-1)

        x_signal = self.max(x_signal)
        x_signal = x_signal[:,:,-9:]
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear(x_signal)
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn_ordinary_2_pointwise_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_ordinary_2_pointwise_1,self).__init__()
        self.model_name = 'cnn_deepwise_2_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.ordinary2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 20)
        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.linear = nn.Linear(in_features= 20, out_features= 20)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.ordinary2(x_signal.unsqueeze(1)).squeeze(-2)
        x_signal = x_signal[:,:,-5:]
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear(x_signal)
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn_ordinary_2(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_ordinary_2,self).__init__()
        self.model_name = 'cnn_deepwise_2'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.ordinary2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 20)
        # self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.linear = nn.Linear(in_features= 20, out_features= 20)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.ordinary2(x_signal.unsqueeze(1)).squeeze(-2)
        x_signal = torch.mean(x_signal[:,:,-5:],axis = 2)
        # x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear(x_signal)
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn_ordinary_ordinary_pointwise_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_ordinary_ordinary_pointwise_1,self).__init__()
        self.model_name = 'cnn_ordinary_ordinary_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.ordinary2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3)
        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.linear = nn.Linear(in_features= 360, out_features= 180)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(180,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.ordinary2(x_signal.unsqueeze(1))
        x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])
        x_signal = x_signal[:,:,-5:]
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear(x_signal)
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn_ordinary_ordinary_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_ordinary_ordinary_1,self).__init__()
        self.model_name = 'cnn_deepwise_ordinary_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.ordinary2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3)
        # self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.linear = nn.Linear(in_features= 360, out_features= 180)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(180,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.ordinary2(x_signal.unsqueeze(1))
        x_signal = x_signal.view(x_signal.shape[0],-1,x_signal.shape[-1])
        x_signal = torch.mean(x_signal[:,:,-5:],axis = 2)
        # x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear(x_signal)
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn_ordinary_ordinary_fc_pointwise_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_ordinary_ordinary_fc_pointwise_1,self).__init__()
        self.model_name = 'cnn_ordinary_ordinary_fc_pointwise_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.ordinary2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3)
        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)

        self.linear1 = nn.Linear(in_features= 20, out_features= 1)  
        self.linear2 = nn.Linear(in_features= 18, out_features= 50)      
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(50,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.ordinary2(x_signal.unsqueeze(1))
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)        
        x_signal = self.linear1(x_signal.permute(0,3,2,1)).squeeze(-1).permute(0,2,1)

        x_signal = x_signal[:,:,-5:]
        x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn_ordinary_ordinary_fc_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_ordinary_ordinary_fc_1,self).__init__()
        self.model_name = 'cnn_deepwise_ordinary_fc_1'
        
        self.ordinary1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.ordinary2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3)
        # self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.linear1 = nn.Linear(in_features= 20, out_features= 1)  
        self.linear2 = nn.Linear(in_features= 18, out_features= 50)     
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(50,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.ordinary1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.ordinary2(x_signal.unsqueeze(1))
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)        
        x_signal = self.linear1(x_signal.permute(0,3,2,1)).squeeze(-1).permute(0,2,1)
        x_signal = torch.mean(x_signal[:,:,-5:],axis = 2)
        # x_signal = self.pointwise(x_signal.permute(0,2,1).unsqueeze(-1)).squeeze(-1).squeeze(1)

        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out