from models.BasicModule import BasicModule
import torch.nn as nn
import torch

class cnn2d_sxy_1(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn2d_sxy_1,self).__init__()
        self.model_name = 'cnn2d_sxy_1: cnn_ordinary_ordinary_pointwise_1'
        
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


class cnn2d_sxy_2(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn2d_sxy_2,self).__init__()
        self.model_name = 'cnn2d_sxy_2:cnn_deepwise_ordinary_1'
        
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


class cnn2d_sxy_3(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn2d_sxy_3,self).__init__()
        self.model_name = 'cnn2d_sxy_3:cnn_ordinary_ordinary_fc_pointwise_1'
        
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


class cnn2d_sxy_4(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn2d_sxy_4,self).__init__()
        self.model_name = 'cnn2d_sxy_4:cnn_deepwise_ordinary_fc_1'
        
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


class cnn2d_sxy_5(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn2d_sxy_5,self).__init__()
        self.model_name = 'cnn2d_sxy_5: cnn_deepwise_deepwise_ordinary_pointwise_1'
        
        self.deepwise1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.deepwise2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3)
        self.deepwise_act = nn.ReLU()

        self.ordinary1 = nn.Conv2d(in_channels= 20, out_channels= 40, kernel_size=3)
        self.act1 = nn.ReLU()
        
        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.pointwise_act = nn.ReLU()


        self.linear1 = nn.Linear(in_features= 16, out_features= 1)
        self.linear2 = nn.Linear(in_features= 40, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.deepwise1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.deepwise2(x_signal.unsqueeze(1))
        x_signal = self.deepwise_act(x_signal)

        x_signal = self.ordinary1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = x_signal[:,:,:,-5:]
        x_signal = self.pointwise(x_signal.permute(0,3,2,1)).squeeze(1)
        x_signal = self.pointwise_act(x_signal)

        x_signal = self.dropout(x_signal)
        
        x_signal = self.linear1(x_signal.permute(0,2,1)).squeeze(-1)
        x_signal = self.linear2(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_6(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn2d_sxy_6,self).__init__()
        self.model_name = 'cnn2d_sxy_6: cnn_deepwise_deepwise_ordinarymax_pointwise_1'
        
        self.deepwise1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.deepwise2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3)
        self.deepwise_act = nn.ReLU()

        self.ordinary1 = nn.Conv2d(in_channels= 20, out_channels= 40, kernel_size=3)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size= 2)    

        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.pointwise_act = nn.ReLU()

        self.linear1 = nn.Linear(in_features= 8, out_features= 1)
        self.linear2 = nn.Linear(in_features= 40, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.deepwise1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.deepwise2(x_signal.unsqueeze(1))
        x_signal = self.deepwise_act(x_signal)

        x_signal = self.ordinary1(x_signal)
        x_signal = self.act1(x_signal)
        x_signal = self.avgpool1(x_signal)
        

        x_signal = x_signal[:,:,:,-5:]
        x_signal = self.pointwise(x_signal.permute(0,3,2,1)).squeeze(1)
        x_signal = self.pointwise_act(x_signal)

        x_signal = self.dropout(x_signal)
        
        x_signal = self.linear1(x_signal.permute(0,2,1)).squeeze(-1)
        x_signal = self.linear2(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_7(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn2d_sxy_7,self).__init__()
        self.model_name = 'cnn2d_sxy_7: cnn_deepwise_deepwise_ordinary_2_pointwise_1'
        
        self.deepwise1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.deepwise2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3)
        self.deepwise_act = nn.ReLU()

        self.ordinary1 = nn.Conv2d(in_channels= 20, out_channels= 40, kernel_size=3)
        self.act1 = nn.ReLU()

        self.ordinary2 = nn.Conv2d(in_channels= 40, out_channels= 80, kernel_size=3)
        self.act2 = nn.ReLU()

        
        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.pointwise_act = nn.ReLU()


        self.linear1 = nn.Linear(in_features= 14, out_features= 1)
        self.linear2 = nn.Linear(in_features= 80, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.deepwise1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.deepwise2(x_signal.unsqueeze(1))
        x_signal = self.deepwise_act(x_signal)

        x_signal = self.ordinary1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.ordinary2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = x_signal[:,:,:,-5:]
        x_signal = self.pointwise(x_signal.permute(0,3,2,1)).squeeze(1)
        x_signal = self.pointwise_act(x_signal)

        x_signal = self.dropout(x_signal)
        
        x_signal = self.linear1(x_signal.permute(0,2,1)).squeeze(-1)
        x_signal = self.linear2(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class cnn2d_sxy_8(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn2d_sxy_8,self).__init__()
        self.model_name = 'cnn2d_sxy_8: cnn_deepwise_deepwise_ordinary_3_pointwise_1'
        
        self.deepwise1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.deepwise2 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size= 3)
        self.deepwise_act = nn.ReLU()

        self.ordinary1 = nn.Conv2d(in_channels= 20, out_channels= 40, kernel_size=3)
        self.act1 = nn.ReLU()

        self.ordinary2 = nn.Conv2d(in_channels= 40, out_channels= 80, kernel_size=3)
        self.act2 = nn.ReLU()

        self.ordinary3 = nn.Conv2d(in_channels= 80, out_channels= 160, kernel_size=3)
        self.act3 = nn.ReLU()
        
        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.pointwise_act = nn.ReLU()


        self.linear1 = nn.Linear(in_features= 12, out_features= 1)
        self.linear2 = nn.Linear(in_features= 160, out_features= 100)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(100,1)
        self.act = nn.ReLU()
        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.deepwise1(x.unsqueeze(1)).squeeze(-1)
        x_signal = self.deepwise2(x_signal.unsqueeze(1))
        x_signal = self.deepwise_act(x_signal)

        x_signal = self.ordinary1(x_signal)
        x_signal = self.act1(x_signal)

        x_signal = self.ordinary2(x_signal)
        x_signal = self.act2(x_signal)

        x_signal = self.ordinary3(x_signal)
        x_signal = self.act3(x_signal)

        x_signal = x_signal[:,:,:,-5:]
        x_signal = self.pointwise(x_signal.permute(0,3,2,1)).squeeze(1)
        x_signal = self.pointwise_act(x_signal)

        x_signal = self.dropout(x_signal)
        
        x_signal = self.linear1(x_signal.permute(0,2,1)).squeeze(-1)
        x_signal = self.linear2(x_signal)

        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out