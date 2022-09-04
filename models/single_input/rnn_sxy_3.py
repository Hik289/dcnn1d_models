from models.BasicModule import BasicModule
import torch.nn as nn
import torch

from .rnn_cell_sxy.LSTMcell_sxy import LSTMcell,LSTMcell_v2,Conv1d_LSTMcell
from .rnn_cell_sxy.GRUcell_sxy import GRUcell,GRUcell_v2,Conv1d_GRUcell_v1,Conv1d_GRUcell_v2,Conv1d_GRUcell_v3,GRUcell_simple
from .rnn_cell_sxy.RNNcell_sxy import RNNcell,RNNcell_v2,Conv1d_RNNcell
from .rnn_cell_sxy.LSTMCcell_sxy import LSTMCcell,LSTMCcell_v2,Conv1d_LSTMCcell
from .rnn_cell_sxy.PASScell_1_sxy import PASScell_1,PASScell_1_v2,Conv1d_PASScell_1
from .rnn_cell_sxy.PASScell_2_sxy import PASScell_2,PASScell_2_v2,Conv1d_PASScell_2_v1,Conv1d_PASScell_2_v2,Conv1d_PASScell_2_v3

class rnn_sxy_2_1(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_1,self).__init__()
        self.model_name = 'rnn_sxy_2_1:daily'

        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 30)

        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()


    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x, axis = 2)

        x_signal,_ = self.rnn1(x_day.permute(1,0,2))

        # x_signal = self.linear(x_signal)
        # #out = self.dropout(x_signal)
        # x_signal = self.linear(x_signal)
        # x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out[-1,:,:]
        self.model_out['signals'] = x_signal[-1,:,:]
        return self.model_out


class rnn_sxy_2_2(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_2,self).__init__()
        self.model_name = 'rnn_sxy_2_2:daily'

        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 30)

        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()


    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x, axis = 2)

        x_signal,_ = self.rnn1(x_day)

        # x_signal = self.linear(x_signal)
        # #out = self.dropout(x_signal)
        # x_signal = self.linear(x_signal)
        # x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out[:,-1,:]
        self.model_out['signals'] = x_signal[:,-1,:]
        return self.model_out


class rnn_sxy_2_3(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_3,self).__init__()
        self.model_name = 'rnn_sxy_2_3:daily'

        self.rnn1 = nn.LSTM(input_size = 1, hidden_size = 250)
        self.act = nn.ReLU()
        self.end_layer = nn.Linear(250,1)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x, axis = 2)

        x_signal,_ = self.rnn1(x_day.permute(1,0,2)[:,:,0].unsqueeze(-1))

        # x_signal = self.linear(x_signal)
        # #out = self.dropout(x_signal)
        # x_signal = self.linear(x_signal)
        # x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out[-1,:,:]
        self.model_out['signals'] = x_signal[-1,:,:]
        return self.model_out


class rnn_sxy_2_4(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_4,self).__init__()
        self.model_name = 'rnn_sxy_2_4:daily'

        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250)

        self.linear = nn.Linear(250,30)

        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()


    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x, axis = 2)

        x_signal,_ = self.rnn1(x_day.permute(1,0,2))

        x_signal = self.linear(x_signal)
        out = self.dropout(x_signal)
        # x_signal = self.linear(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out[-1,:,:]
        self.model_out['signals'] = x_signal[-1,:,:]
        return self.model_out


class rnn_sxy_2_5(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_5,self).__init__()
        self.model_name = 'rnn_sxy_2_5:daily'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 1)    

        self.rnn5 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 1)

        self.pointwise1 = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 250, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))


        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn5(x_day.permute(1,0,2))

        x_signal = self.pointwise1(x_signal[-5:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-5:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_6(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_6,self).__init__()
        self.model_name = 'rnn_sxy_2_6:daily'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)    

        self.rnn5 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)

        self.pointwise1 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 250, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        for p in self.rnn1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn5.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.linear1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))


        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn5(x_day.permute(1,0,2))

        x_signal = self.pointwise1(x_signal[-10:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-4:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_7(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_7,self).__init__()
        self.model_name = 'rnn_sxy_2_7:daily'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)    

        self.rnn5 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)

        self.pointwise1 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 250, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))


        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn5(x_day.permute(1,0,2))

        x_signal = self.pointwise1(x_signal[-10:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-4:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_8(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_8,self).__init__()
        self.model_name = 'rnn_sxy_2_8:daily'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 100, num_layers = 4)    

        self.rnn5 = nn.LSTM(input_size = 3, hidden_size = 100, num_layers = 2)

        self.pointwise1 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 100, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))


        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn5(x_day.permute(1,0,2))

        x_signal = self.pointwise1(x_signal[-20:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-4:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_9(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_9,self).__init__()
        self.model_name = 'rnn_sxy_2_9:daily'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 100, num_layers = 3)    

        self.rnn5 = nn.LSTM(input_size = 3, hidden_size = 100, num_layers = 1)

        self.pointwise1 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 100, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))


        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn5(x_day.permute(1,0,2))

        x_signal = self.pointwise1(x_signal[-10:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-4:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_10(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_10,self).__init__()
        self.model_name = 'rnn_sxy_2_10:daily'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250)
        self.rnn2 = nn.LSTM(input_size = 250, hidden_size = 200)
        self.rnn3 = nn.LSTM(input_size = 200, hidden_size = 150)
        self.rnn4 = nn.LSTM(input_size = 150, hidden_size = 100)       

        self.rnn5 = nn.LSTM(input_size = 3, hidden_size = 200)
        self.rnn6 = nn.LSTM(input_size = 200, hidden_size = 150)
        self.rnn7 = nn.LSTM(input_size = 150, hidden_size = 100)

        self.pointwise1 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 100, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))
        x_signal,_ = self.rnn2(x_signal[:,:,:])
        x_signal,_ = self.rnn3(x_signal[:,:,:])
        x_signal,_ = self.rnn4(x_signal[:,:,:])

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn5(x_day.permute(1,0,2))
        x_day,_ = self.rnn6(x_day)
        x_day,_ = self.rnn7(x_day)

        x_signal = self.pointwise1(x_signal[-20:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-4:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_11(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_11,self).__init__()
        self.model_name = 'rnn_sxy_2_11:daily'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250)
        self.rnn2 = nn.LSTM(input_size = 250, hidden_size = 150)
        self.rnn3 = nn.LSTM(input_size = 150, hidden_size = 10)

        self.rnn4 = nn.LSTM(input_size = 3, hidden_size = 20)
        self.rnn5 = nn.LSTM(input_size = 20, hidden_size = 10)

        self.pointwise1 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 10, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))
        x_signal,_ = self.rnn2(x_signal[:,:,:])
        x_signal,_ = self.rnn3(x_signal[:,:,:])

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn4(x_day.permute(1,0,2))
        x_day,_ = self.rnn5(x_day)

        x_signal = self.pointwise1(x_signal[-15:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-3:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_12(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_12,self).__init__()
        self.model_name = 'rnn_sxy_2_12:daily'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250)
        self.rnn2 = nn.LSTM(input_size = 250, hidden_size = 10)

        self.rnn3 = nn.LSTM(input_size = 3, hidden_size = 20)
        self.rnn4 = nn.LSTM(input_size = 20, hidden_size = 10)

        self.pointwise1 = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 10, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))
        x_signal,_ = self.rnn2(x_signal[:,:,:])

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn3(x_day.permute(1,0,2))
        x_day,_ = self.rnn4(x_day)


        x_signal = self.pointwise1(x_signal[-5:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-5:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_13(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_13,self).__init__()
        self.model_name = 'rnn_sxy_2_13:daily'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250)
        self.rnn2 = nn.LSTM(input_size = 250, hidden_size = 200)
        self.rnn3 = nn.LSTM(input_size = 200, hidden_size = 150)
        self.rnn4 = nn.LSTM(input_size = 150, hidden_size = 100)      
        self.rnn5 = nn.LSTM(input_size = 100, hidden_size = 10)      

        self.rnn6 = nn.LSTM(input_size = 3, hidden_size = 200)
        self.rnn7 = nn.LSTM(input_size = 200, hidden_size = 150)
        self.rnn8 = nn.LSTM(input_size = 150, hidden_size = 100)
        self.rnn9 = nn.LSTM(input_size = 100, hidden_size = 10)

        self.pointwise1 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 10, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))
        x_signal,_ = self.rnn2(x_signal[:,:,:])
        x_signal,_ = self.rnn3(x_signal[:,:,:])
        x_signal,_ = self.rnn4(x_signal[:,:,:])
        x_signal,_ = self.rnn5(x_signal[:,:,:])

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn6(x_day.permute(1,0,2))
        x_day,_ = self.rnn7(x_day)
        x_day,_ = self.rnn8(x_day)
        x_day,_ = self.rnn9(x_day)

        x_signal = self.pointwise1(x_signal[-20:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-4:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_14(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_14,self).__init__()
        self.model_name = 'rnn_sxy_2_14:daily'
        
        self.rnn1 = nn.GRU(input_size = 3, hidden_size = 250)
        self.rnn2 = nn.GRU(input_size = 250, hidden_size = 150)
        self.rnn3 = nn.GRU(input_size = 150, hidden_size = 10)

        self.rnn4 = nn.GRU(input_size = 3, hidden_size = 20)
        self.rnn5 = nn.GRU(input_size = 20, hidden_size = 10)

        self.pointwise1 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 10, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))
        x_signal,_ = self.rnn2(x_signal[:,:,:])
        x_signal,_ = self.rnn3(x_signal[:,:,:])

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn4(x_day.permute(1,0,2))
        x_day,_ = self.rnn5(x_day)

        x_signal = self.pointwise1(x_signal[-15:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-3:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_15(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_15,self).__init__()
        self.model_name = 'rnn_sxy_2_15'

        self.i_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.f_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.g_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.o_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()  



        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)
        
        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()
        self.h_t = self.h_t.detach()
        i_t = self.i_t(x[:,0,:], self.h_t)
        f_t = self.f_t(x[:,0,:], self.h_t)
        g_t = self.g_t(x[:,0,:], self.h_t)
        o_t = self.o_t(x[:,0,:], self.h_t)
        c_t = i_t*self.act(g_t)
        self.h_t = o_t*self.act(c_t)
        temp_t = c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            i_t = self.i_t(x[:,i,:], self.h_t)
            f_t = self.f_t(x[:,i,:], self.h_t)
            g_t = self.g_t(x[:,i,:], self.h_t)
            o_t = self.o_t(x[:,i,:], self.h_t)
            c_t = f_t*c_t + i_t*self.act(g_t)
            self.h_t = (o_t*self.act(c_t))
            temp_t = torch.cat([temp_t,c_t.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_16(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_16,self).__init__()
        self.model_name = 'rnn_sxy_1_16'

        self.i_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.f_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.g_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.o_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()       

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)
        # if x.shape[0] != self.h_t.shape[0]:
        #     # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
        #     self.h_t = torch.zeros((x.shape[0], 20)).cuda()
        # self.h_t = self.h_t.detach()
        i_t = self.i_t(x[:,0,:])
        f_t = self.f_t(x[:,0,:])
        g_t = self.g_t(x[:,0,:])
        o_t = self.o_t(x[:,0,:])
        c_t = i_t*self.act(g_t)
        self.h_t = o_t*self.act(c_t)
        temp_t = c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            i_t = self.i_t(x[:,i,:], self.h_t)
            f_t = self.f_t(x[:,i,:], self.h_t)
            g_t = self.g_t(x[:,i,:], self.h_t)
            o_t = self.o_t(x[:,i,:], self.h_t)
            c_t = f_t*c_t + i_t*self.act(g_t)
            self.h_t = (o_t*self.act(c_t))
            temp_t = torch.cat([temp_t,c_t.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out



class rnn_sxy_2_17(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_17,self).__init__()
        self.model_name = 'rnn_sxy_2_17'

        self.z_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.r_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.n_t = nn.RNNCell(input_size= 3, hidden_size= 20)

        self.pointwise1 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)

        self.end_layer = nn.Linear(20,1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()       

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()
        self.h_t = self.h_t.detach()
        r_t = self.act2(self.r_t(x[:,0,:], self.h_t))
        z_t = self.act2(self.z_t(x[:,0,:], self.h_t))
        n_t = self.act1(self.n_t(x[:,0,:], r_t*self.h_t))

        self.h_t = (1-z_t)*n_t + z_t*self.h_t
        temp_t = n_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            r_t = self.act2(self.r_t(x[:,i,:], self.h_t))
            z_t = self.act2(self.z_t(x[:,i,:], self.h_t))
            n_t = self.act1(self.n_t(x[:,i,:], r_t*self.h_t))

            self.h_t = (1-z_t)*n_t + z_t*self.h_t
            temp_t = torch.cat([temp_t,n_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-1,:]

        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  


        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_17(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_17,self).__init__()
        self.model_name = 'rnn_sxy_2_17:rnn_sxy_2_false'

        self.z_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.r_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.n_t = nn.RNNCell(input_size= 3, hidden_size= 20)

        self.end_layer = nn.Linear(20,1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()       

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        # if x.shape[0] != self.h_t.shape[0]:
        #     # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
        #     self.h_t = torch.zeros((x.shape[0], 20)).cuda()
        # self.h_t = self.h_t.detach()
        r_t = self.act2(self.r_t(x[:,0,:]))
        z_t = self.act2(self.z_t(x[:,0,:]))
        n_t = self.act1(self.n_t(x[:,0,:]))

        self.h_t = (1-z_t)*n_t + z_t*self.h_t
        temp_t = n_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            r_t = self.act2(self.r_t(x[:,i,:], self.h_t))
            z_t = self.act2(self.z_t(x[:,i,:], self.h_t))
            n_t = self.act1(self.n_t(x[:,i,:], r_t*self.h_t))

            self.h_t = (1-z_t)*n_t + z_t*self.h_t
            temp_t = torch.cat([temp_t,n_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-1,:]
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out



class rnn_sxy_2_18(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_18,self).__init__()
        self.model_name = 'rnn_sxy_2_18:rnn_sxy_4'

        self.i_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.f_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.g_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.o_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()       

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  

        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()
        self.h_t = self.h_t.detach()
        i_t = self.i_t(x[:,0,:], self.h_t)
        f_t = self.f_t(x[:,0,:], self.h_t)
        g_t = self.g_t(x[:,0,:], self.h_t)
        o_t = self.o_t(x[:,0,:], self.h_t)
        c_t = i_t*self.act(g_t)
        self.h_t = o_t*self.act(c_t)
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            i_t = self.i_t(x[:,i,:], self.h_t)
            f_t = self.f_t(x[:,i,:], self.h_t)
            g_t = self.g_t(x[:,i,:], self.h_t)
            o_t = self.o_t(x[:,i,:], self.h_t)
            c_t = f_t*c_t + i_t*self.act(g_t)
            self.h_t = (o_t*self.act(c_t))
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_19(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_19,self).__init__()
        self.model_name = 'rnn_sxy_2_19'

        self.i_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.f_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.g_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.o_t = nn.RNNCell(input_size= 3, hidden_size= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()       

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        # if x.shape[0] != self.h_t.shape[0]:
        #     # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
        #     self.h_t = torch.zeros((x.shape[0], 20)).cuda()
        # self.h_t = self.h_t.detach()
        i_t = self.i_t(x[:,0,:])
        f_t = self.f_t(x[:,0,:])
        g_t = self.g_t(x[:,0,:])
        o_t = self.o_t(x[:,0,:])
        c_t = i_t*self.act(g_t)
        self.h_t = o_t*self.act(c_t)
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            i_t = self.i_t(x[:,i,:], self.h_t)
            f_t = self.f_t(x[:,i,:], self.h_t)
            g_t = self.g_t(x[:,i,:], self.h_t)
            o_t = self.o_t(x[:,i,:], self.h_t)
            c_t = f_t*c_t + i_t*self.act(g_t)
            self.h_t = (o_t*self.act(c_t))
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_20(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_20,self).__init__()
        self.model_name = 'rnn_sxy_2_20'

        self.lstm = LSTMcell(3,20)

        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()     
        self.c_t = torch.zeros((1200, 20)).cuda()     
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()
            self.c_t = torch.zeros((x.shape[0], 20)).cuda()
        self.h_t = self.h_t.detach()
        self.c_t = self.c_t.detach()

        out, self.c_t, self.h_t = self.lstm(x[:,0,:],self.c_t,self.h_t)
        temp_t = out.unsqueeze(1)
        for i in range(1,x.shape[1]):
            out, self.c_t, self.h_t = self.lstm(x[:,i,:],self.c_t,self.h_t)
            temp_t = torch.cat([temp_t,out.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out



class rnn_sxy_2_21(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_21,self).__init__()
        self.model_name = 'rnn_sxy_2_21'

        self.gru = GRUcell(3,20)

        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()      
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()

        self.h_t = self.h_t.detach()

        out, self.h_t = self.gru(x[:,0,:],self.h_t)
        temp_t = out.unsqueeze(1)
        for i in range(1,x.shape[1]):
            out, self.h_t = self.gru(x[:,i,:],self.h_t)
            temp_t = torch.cat([temp_t,out.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_22(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_22,self).__init__()
        self.model_name = 'rnn_sxy_2_22'

        self.rnn = RNNcell(3,20)

        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()      
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()

        self.h_t = self.h_t.detach()

        out, self.h_t = self.rnn(x[:,0,:],self.h_t)
        temp_t = out.unsqueeze(1)
        for i in range(1,x.shape[1]):
            out, self.h_t = self.rnn(x[:,i,:],self.h_t)
            temp_t = torch.cat([temp_t,out.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_23(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_23,self).__init__()
        self.model_name = 'rnn_sxy_2_23'

        self.lstmc = LSTMCcell(3,20)

        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()      
        self.c_t = torch.zeros((1200, 20)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()
            self.c_t = torch.zeros((x.shape[0], 20)).cuda()
        self.h_t = self.h_t.detach()
        self.c_t = self.c_t.detach()
        out, self.c_t, self.h_t = self.lstmc(x[:,0,:],self.c_t,self.h_t)
        temp_t = out.unsqueeze(1)
        for i in range(1,x.shape[1]):
            out, self.c_t, self.h_t = self.lstmc(x[:,i,:],self.c_t, self.h_t)
            temp_t = torch.cat([temp_t,out.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_24(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_24,self).__init__()
        self.model_name = 'rnn_sxy_2_24: PASScell_1'

        self.pass_1 = PASScell_1(3,20)

        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()      
        self.o_t = torch.zeros((1200, 20)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()
            self.o_t = torch.zeros((x.shape[0], 20)).cuda()
        self.h_t = self.h_t.detach()
        self.o_t = self.o_t.detach()
        self.o_t, self.h_t = self.pass_1(x[:,0,:],self.o_t,self.h_t)
        temp_t = self.o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.o_t, self.h_t = self.pass_1(x[:,i,:],self.o_t, self.h_t)
            temp_t = torch.cat([temp_t,self.o_t.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_25(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_25,self).__init__()
        self.model_name = 'rnn_sxy_2_25: LSTM+RNN'

        self.lstm = LSTMcell(20,30)
        self.rnn = RNNcell(3,20)

        self.end_layer = nn.Linear(30,100)
        self.end_layer2 = nn.Linear(100,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()     
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda() 
        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]

        out = self.end_layer(x_signal)
        out = self.act(out)
        out = self.end_layer2(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_26(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_26,self).__init__()
        self.model_name = 'rnn_sxy_2_26: PASScell_2'

        self.pass_2 = PASScell_2(3,20)

        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()      
        self.o_t = torch.zeros((1200, 20)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        
        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()
            self.o_t = torch.zeros((x.shape[0], 20)).cuda()
        self.h_t = self.h_t.detach()
        self.o_t = self.o_t.detach()
        self.o_t, self.h_t = self.pass_2(x[:,0,:],self.o_t,self.h_t)
        temp_t = self.o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.o_t, self.h_t = self.pass_2(x[:,i,:],self.o_t, self.h_t)
            temp_t = torch.cat([temp_t,self.o_t.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_27(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_27,self).__init__()
        self.model_name = 'rnn_sxy_2_27: LSTMC_fc'

        self.lstmc = LSTMCcell(3,20)

        self.end_layer = nn.Linear(20,100)
        self.end_layer2 = nn.Linear(100,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()      
        self.c_t = torch.zeros((1200, 20)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()
            self.c_t = torch.zeros((x.shape[0], 20)).cuda()
        self.h_t = self.h_t.detach()
        self.c_t = self.c_t.detach()
        out, self.c_t, self.h_t = self.lstmc(x[:,0,:],self.c_t,self.h_t)
        temp_t = out.unsqueeze(1)
        for i in range(1,x.shape[1]):
            out, self.c_t, self.h_t = self.lstmc(x[:,i,:],self.c_t, self.h_t)
            temp_t = torch.cat([temp_t,out.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-1,:]

        out = self.end_layer(x_signal)
        out = self.act(out)
        out = self.end_layer2(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_28(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_28,self).__init__()
        self.model_name = 'rnn_sxy_2_28: LSTMC_pointwise'

        self.lstmc = LSTMCcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.end_layer = nn.Linear(20,100)
        self.end_layer2 = nn.Linear(100,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()      
        self.c_t = torch.zeros((1200, 20)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()
            self.c_t = torch.zeros((x.shape[0], 20)).cuda()
        self.h_t = self.h_t.detach()
        self.c_t = self.c_t.detach()
        out, self.c_t, self.h_t = self.lstmc(x[:,0,:],self.c_t,self.h_t)
        temp_t = out.unsqueeze(1)
        for i in range(1,x.shape[1]):
            out, self.c_t, self.h_t = self.lstmc(x[:,i,:],self.c_t, self.h_t)
            temp_t = torch.cat([temp_t,out.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-5: ,:]

        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)
        out = self.act(out)
        out = self.end_layer2(out)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_29(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_29,self).__init__()
        self.model_name = 'rnn_sxy_2_29: LSTM+RNN+pointwise'

        self.lstm = LSTMcell(20,30)
        self.rnn = RNNcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda() 
        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_30(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_30,self).__init__()
        self.model_name = 'rnn_sxy_2_30: LSTMC_pointwise'

        self.lstmc = LSTMCcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)

        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.h_t = torch.zeros((1200, 20)).cuda()      
        self.c_t = torch.zeros((1200, 20)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.h_t = torch.zeros((x.shape[0], 20)).cuda()
            self.c_t = torch.zeros((x.shape[0], 20)).cuda()
        self.h_t = self.h_t.detach()
        self.c_t = self.c_t.detach()
        out, self.c_t, self.h_t = self.lstmc(x[:,0,:],self.c_t,self.h_t)
        temp_t = out.unsqueeze(1)
        for i in range(1,x.shape[1]):
            out, self.c_t, self.h_t = self.lstmc(x[:,i,:],self.c_t, self.h_t)
            temp_t = torch.cat([temp_t,out.unsqueeze(1)], axis = 1)
        x_signal = temp_t[:,-5: ,:]

        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)
 
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_31(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_31,self).__init__()
        self.model_name = 'rnn_sxy_2_31: LSTM+GRU+pointwise'

        self.lstm = LSTMcell(20,30)
        self.rnn = GRUcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda() 
        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_32(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_32,self).__init__()
        self.model_name = 'rnn_sxy_2_32: LSTM+LSTM+pointwise'

        self.lstm = LSTMcell(20,30)
        self.rnn = LSTMcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    
        self.lstm_c_t = torch.zeros((1200, 30)).cuda() 
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_33(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_33,self).__init__()
        self.model_name = 'rnn_sxy_2_33: LSTM+LSTMC+pointwise'

        self.lstm = LSTMcell(20,30)
        self.rnn = LSTMCcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_34(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_34,self).__init__()
        self.model_name = 'rnn_sxy_2_34: LSTM+PASScell_1+pointwise'

        self.lstm = LSTMcell(20,30)
        self.rnn = PASScell_1(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_35(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_35,self).__init__()
        self.model_name = 'rnn_sxy_2_7:daily'

        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)

        self.rnn2 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)

        self.pointwise1 = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.pointwise2 = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=1)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.linear1 = nn.Linear(250,30)
        self.linear2 = nn.Linear(250,30)

        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()


    def forward(self,x):
        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_signal_day,_ = self.rnn1(x_day.permute(1,0,2))
        x_signal_bar,_ = self.rnn2(x[:,-98:,:].permute(1,0,2))

        x_signal_day = x_signal_day[-5:,:,:]
        x_signal_bar = x_signal_bar[-5:,:,:]
        x_signal_day = self.pointwise1(x_signal_day.unsqueeze(0)).squeeze(1).squeeze(0)
        x_signal_bar = self.pointwise2(x_signal_bar.unsqueeze(0)).squeeze(1).squeeze(0)

        # x_signal_day = self.act1(x_signal_day)
        # x_signal_bar = self.act2(x_signal_bar)

        x_signal_day = self.linear1(x_signal_day)
        x_signal_bar = self.linear2(x_signal_bar)

        x_signal_2 = x_signal_bar+ x_signal_day
        
        x_signal_2 = self.dropout(x_signal_2)
        # x_signal = self.linear(x_signal)
        x_signal_2 = self.act(x_signal_2)

        out = self.end_layer(x_signal_2)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal_2
        return self.model_out


class rnn_sxy_2_36(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_36,self).__init__()
        self.model_name = 'rnn_sxy_2_36: LSTM+PASScell_2+pointwise'

        self.lstm = LSTMcell(20,30)
        self.rnn = PASScell_2(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_37(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_37,self).__init__()
        self.model_name = 'rnn_sxy_2_37: LSTMC+RNN+pointwise'

        self.lstm = LSTMCcell(20,30)
        self.rnn = RNNcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda() 
        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_38(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_38,self).__init__()
        self.model_name = 'rnn_sxy_2_38: LSTMC+GRU+pointwise'

        self.lstm = LSTMcell(20,30)
        self.rnn = GRUcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda() 
        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_39(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_39,self).__init__()
        self.model_name = 'rnn_sxy_2_39: LSTMC+LSTM+pointwise'

        self.lstm = LSTMCcell(20,30)
        self.rnn = LSTMcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    
        self.lstm_c_t = torch.zeros((1200, 30)).cuda() 
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_40(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_40,self).__init__()
        self.model_name = 'rnn_sxy_2_40: LSTMC+LSTMC+pointwise'

        self.lstm = LSTMCcell(20,30)
        self.rnn = LSTMCcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_41(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_41,self).__init__()
        self.model_name = 'rnn_sxy_2_41: LSTMC+PASScell_1+pointwise'

        self.lstm = LSTMCcell(20,30)
        self.rnn = PASScell_1(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_42(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_42,self).__init__()
        self.model_name = 'rnn_sxy_2_42: LSTMC+PASScell_2+pointwise'

        self.lstm = LSTMCcell(20,30)
        self.rnn = PASScell_2(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            o_t, self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_43(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_43,self).__init__()
        self.model_name = 'rnn_sxy_2_43: RNN+RNN+pointwise'

        self.lstm = RNNcell(20,30)
        self.rnn = RNNcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_44(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_44,self).__init__()
        self.model_name = 'rnn_sxy_2_44: RNN+GRU+pointwise'

        self.lstm = RNNcell(20,30)
        self.rnn = GRUcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_45(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_45,self).__init__()
        self.model_name = 'rnn_sxy_2_45: RNN+LSTM+pointwise'

        self.lstm = RNNcell(20,30)
        self.rnn = LSTMcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_46(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_46,self).__init__()
        self.model_name = 'rnn_sxy_2_46: RNN+LSTMC+pointwise'

        self.lstm = RNNcell(20,30)
        self.rnn = LSTMCcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_47(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_47,self).__init__()
        self.model_name = 'rnn_sxy_2_47: RNN+PASScell_1+pointwise'

        self.lstm = RNNcell(20,30)
        self.rnn = PASScell_1(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_48(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_48,self).__init__()
        self.model_name = 'rnn_sxy_2_48: RNN+PASScell_2+pointwise'

        self.lstm = RNNcell(20,30)
        self.rnn = PASScell_2(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_49(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_49,self).__init__()
        self.model_name = 'rnn_sxy_2_49: GRU+RNN+pointwise'

        self.lstm = GRUcell(20,30)
        self.rnn = RNNcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_50(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_50,self).__init__()
        self.model_name = 'rnn_sxy_2_50: GRU+GRU+pointwise'

        self.lstm = GRUcell(20,30)
        self.rnn = GRUcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_51(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_51,self).__init__()
        self.model_name = 'rnn_sxy_2_51: GRU+LSTM+pointwise'

        self.lstm = GRUcell(20,30)
        self.rnn = LSTMcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_52(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_52,self).__init__()
        self.model_name = 'rnn_sxy_2_52: GRU+LSTMC+pointwise'

        self.lstm = GRUcell(20,30)
        self.rnn = LSTMCcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_53(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_53,self).__init__()
        self.model_name = 'rnn_sxy_2_53: GRU+PASScell_1+pointwise'

        self.lstm = GRUcell(20,30)
        self.rnn = PASScell_1(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_54(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_54,self).__init__()
        self.model_name = 'rnn_sxy_2_54: GRU+PASScell_2+pointwise'

        self.lstm = GRUcell(20,30)
        self.rnn = PASScell_2(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   

        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 

        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_55(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_55,self).__init__()
        self.model_name = 'rnn_sxy_2_55: PASScell_1+GRU+pointwise'

        self.lstm = PASScell_1(20,30)
        self.rnn = GRUcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda() 
        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_56(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_56,self).__init__()
        self.model_name = 'rnn_sxy_2_56: PASScell_1+LSTM+pointwise'

        self.lstm = PASScell_1(20,30)
        self.rnn = LSTMcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    
        self.lstm_c_t = torch.zeros((1200, 30)).cuda() 
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_57(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_57,self).__init__()
        self.model_name = 'rnn_sxy_2_57: PASScell_1+LSTMC+pointwise'

        self.lstm = PASScell_1(20,30)
        self.rnn = LSTMCcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_58(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_58,self).__init__()
        self.model_name = 'rnn_sxy_2_58: PASScell_1+PASScell_1+pointwise'

        self.lstm = PASScell_1(20,30)
        self.rnn = PASScell_1(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_59(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_59,self).__init__()
        self.model_name = 'rnn_sxy_2_59: PASScell_1+PASScell_2+pointwise'

        self.lstm = PASScell_1(20,30)
        self.rnn = PASScell_2(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_60(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_60,self).__init__()
        self.model_name = 'rnn_sxy_2_60: PASScell_2+RNN+pointwise'

        self.lstm = PASScell_2(20,30)
        self.rnn = RNNcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda() 
        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_61(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_61,self).__init__()
        self.model_name = 'rnn_sxy_2_61: PASScell_2+GRU+pointwise'

        self.lstm = PASScell_2(20,30)
        self.rnn = GRUcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda() 
        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_62(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_62,self).__init__()
        self.model_name = 'rnn_sxy_2_62: PASScell_2+LSTM+pointwise'

        self.lstm = PASScell_2(20,30)
        self.rnn = LSTMcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    
        self.lstm_c_t = torch.zeros((1200, 30)).cuda() 
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_63(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_63,self).__init__()
        self.model_name = 'rnn_sxy_2_63: PASScell_2+LSTMC+pointwise'

        self.lstm = PASScell_2(20,30)
        self.rnn = LSTMCcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()    
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_64(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_64,self).__init__()
        self.model_name = 'rnn_sxy_2_64: PASScell_2+PASScell_1+pointwise'

        self.lstm = PASScell_2(20,30)
        self.rnn = PASScell_1(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_65(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_65,self).__init__()
        self.model_name = 'rnn_sxy_2_65: PASScell_2+PASScell_2+pointwise'

        self.lstm = PASScell_2(20,30)
        self.rnn = PASScell_2(3,20)

        self.pointwise = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_o_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()   
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.linear.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()  
            self.rnn_o_t = torch.zeros((x.shape[0], 20)).cuda()            
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda()
        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_o_t = self.rnn_o_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        self.rnn_o_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_o_t, self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_o_t,self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(self.rnn_o_t, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-5:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_66(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_66,self).__init__()
        self.model_name = 'rnn_sxy_2_66: LSTM+GRU+GRU+LSTM+pointwise'

        self.lstm_q = LSTMcell(150,150)
        self.rnn_q = GRUcell(3,150)

        self.pointwise_q = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)

        self.linear = nn.Linear(150,30)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        self.rnn_h_t_q = torch.zeros((1200, 150)).cuda()
        self.lstm_h_t_q = torch.zeros((1200, 150)).cuda()     
        self.lstm_c_t_q = torch.zeros((1200, 150)).cuda()  


        self.lstm = GRUcell(20,150)
        self.rnn = LSTMcell(3,20)
        self.pointwise = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 150)).cuda()  

        for p in self.lstm_q.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn_q.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise_q.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.lstm.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.linear.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)


    def forward(self,x):

        if x.shape[0] != self.rnn_h_t_q.shape[0]:
            self.rnn_h_t_q = torch.zeros((x.shape[0], 150)).cuda()            
            self.lstm_h_t_q = torch.zeros((x.shape[0], 150)).cuda()
            self.lstm_c_t_q = torch.zeros((x.shape[0], 150)).cuda() 
        self.rnn_h_t_q = self.rnn_h_t_q.detach()
        self.lstm_h_t_q = self.lstm_h_t_q.detach()
        self.lstm_c_t_q = self.lstm_c_t_q.detach()
        rnn_out_q, self.rnn_h_t_q = self.rnn_q(x[:,0,:],self.rnn_h_t_q)
        o_t_q, self.lstm_c_t_q, self.lstm_h_t_q = self.lstm_q(rnn_out_q, self.lstm_c_t_q, self.lstm_h_t_q)    
        temp_t_q = o_t_q.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out_q, self.rnn_h_t_q = self.rnn_q(x[:,i,:],self.rnn_h_t_q)
            o_t_q, self.lstm_c_t_q, self.lstm_h_t_q = self.lstm_q(rnn_out_q, self.lstm_c_t_q, self.lstm_h_t_q)    
            temp_t_q = torch.cat([temp_t_q,o_t_q.unsqueeze(1)], axis = 1)

        x_signal_q = temp_t_q[:,-15:,:]
        x_signal_q = self.pointwise_q(x_signal_q).squeeze(-1).squeeze(1)


        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)

        if x.shape[0] != self.rnn_h_t.shape[0]:
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda() 
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()           
            self.lstm_h_t = torch.zeros((x.shape[0], 150)).cuda()

        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-4:,:]
        x_signal = self.pointwise(x_signal).squeeze(-1).squeeze(1)

        out = self.linear(x_signal_q + x_signal)
        out = self.act(out)
        out = self.end_layer(out)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal_q
        return self.model_out


class rnn_sxy_2_67(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_67,self).__init__()
        self.model_name = 'rnn_sxy_2_67: LSTMC+PASScell_2+GRU+LSTM+pointwise'

        self.lstm_q = LSTMCcell(150,150)
        self.rnn_q = PASScell_2(3,150)

        self.pointwise_q = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)

        self.linear = nn.Linear(150,30)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        self.rnn_h_t_q = torch.zeros((1200, 150)).cuda()
        self.rnn_o_t_q = torch.zeros((1200, 150)).cuda()
        self.lstm_h_t_q = torch.zeros((1200, 150)).cuda()    
        self.lstm_c_t_q = torch.zeros((1200, 150)).cuda() 


        self.lstm = GRUcell(20,150)
        self.rnn = LSTMcell(3,20)
        self.pointwise = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.rnn_h_t = torch.zeros((1200, 20)).cuda()
        self.rnn_c_t = torch.zeros((1200, 20)).cuda()
        self.lstm_h_t = torch.zeros((1200, 150)).cuda() 


    def forward(self,x):

        if x.shape[0] != self.rnn_h_t_q.shape[0]:
            self.rnn_h_t_q = torch.zeros((x.shape[0], 150)).cuda()
            self.rnn_o_t_q = torch.zeros((x.shape[0], 150)).cuda()              
            self.lstm_h_t_q = torch.zeros((x.shape[0], 150)).cuda()
            self.lstm_c_t_q = torch.zeros((x.shape[0], 150)).cuda()
        self.rnn_h_t_q = self.rnn_h_t_q.detach()
        self.lstm_h_t_q = self.lstm_h_t_q.detach()
        self.lstm_c_t_q = self.lstm_c_t_q.detach()
        self.rnn_o_t_q, self.rnn_h_t_q = self.rnn_q(x[:,0,:],self.rnn_o_t_q,self.rnn_h_t_q)
        o_t_q, self.lstm_c_t_q, self.lstm_h_t_q = self.lstm_q(self.rnn_o_t_q, self.lstm_c_t_q, self.lstm_h_t_q)    
        temp_t_q = o_t_q.unsqueeze(1)
        for i in range(1,x.shape[1]):
            self.rnn_o_t_q, self.rnn_h_t_q = self.rnn_q(x[:,i,:],self.rnn_o_t_q,self.rnn_h_t_q)
            o_t_q, self.lstm_c_t_q, self.lstm_h_t_q = self.lstm_q(self.rnn_o_t_q, self.lstm_c_t_q, self.lstm_h_t_q)    
            temp_t_q = torch.cat([temp_t_q,o_t_q.unsqueeze(1)], axis = 1)

        x_signal_q = temp_t_q[:,-15:,:]
        x_signal_q = self.pointwise_q(x_signal_q).squeeze(-1).squeeze(1)


        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)

        if x.shape[0] != self.rnn_h_t.shape[0]:
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()
            self.rnn_c_t = torch.zeros((x.shape[0], 20)).cuda()          
            self.lstm_h_t = torch.zeros((x.shape[0], 150)).cuda()

        self.rnn_h_t = self.rnn_h_t.detach()
        self.rnn_c_t = self.rnn_c_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()

        rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_c_t, self.rnn_h_t)
        o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
        temp_t = o_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_c_t,self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_c_t,self.rnn_h_t)
            o_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,o_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-4:,:]
        x_signal = self.pointwise(x_signal).squeeze(-1).squeeze(1)

        out = self.linear(x_signal_q + x_signal)
        out = self.act(out)
        out = self.end_layer(out)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal_q
        return self.model_out


class rnn_sxy_2_68(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_68,self).__init__()
        self.model_name = 'rnn_sxy_2_68:minutes490+days10+minutes49'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 1)    

        self.rnn2 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 1)

        self.rnn3 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 1)

        self.pointwise1 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        self.pointwise3 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 250, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        for p in self.rnn1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))


        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn2(x_day.permute(1,0,2))

        x_last_day,_ = self.rnn3(x[:,:,-49:].permute(1,0,2))

        x_signal = self.pointwise1(x_signal[-15:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-3:,:,:].permute(1,0,2)).squeeze(1)
        x_last_day = self.pointwise3(x_last_day[-15:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day + x_last_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_69(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_69,self).__init__()
        self.model_name = 'rnn_sxy_2_69:minutes490_fc+days10_fc+minutes49_fc'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 1)    

        self.rnn2 = nn.LSTM(input_size = 3, hidden_size = 20, num_layers = 1)

        self.rnn3 = nn.LSTM(input_size = 3, hidden_size = 100, num_layers = 1)

        self.pointwise1 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        self.pointwise3 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 250, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.linear3 = nn.Linear(in_features= 100, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        for p in self.rnn1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))


        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn2(x_day.permute(1,0,2))

        x_last_day,_ = self.rnn3(x[:,:,-49:].permute(1,0,2))

        x_signal = self.pointwise1(x_signal[-15:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-3:,:,:].permute(1,0,2)).squeeze(1)
        x_last_day = self.pointwise3(x_last_day[-15:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = self.linear1(x_signal)
        x_day = self.linear2(x_day)
        x_last_day = self.linear3(x_last_day)

        x_signal = x_signal + x_day + x_last_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.dropout(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_70(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_70,self).__init__()
        self.model_name = 'rnn_sxy_2_70:(minutes490+days10+minutes49)cat_fc'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 1)    

        self.rnn2 = nn.LSTM(input_size = 3, hidden_size = 20, num_layers = 1)

        self.rnn3 = nn.LSTM(input_size = 3, hidden_size = 100, num_layers = 1)

        self.pointwise1 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        self.pointwise3 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 370, out_features= 200)

        self.end_layer = nn.Linear(200,1)
        self.act = nn.ReLU()

        for p in self.rnn1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))


        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn2(x_day.permute(1,0,2))

        x_last_day,_ = self.rnn3(x[:,:,-49:].permute(1,0,2))

        x_signal = self.pointwise1(x_signal[-15:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-3:,:,:].permute(1,0,2)).squeeze(1)
        x_last_day = self.pointwise3(x_last_day[-15:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = self.dropout(x_signal)
        x_signal = torch.cat([x_signal, x_day, x_last_day],axis = 1)

        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_71(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_71,self).__init__()
        self.model_name = 'rnn_sxy_2_71:minutes490_3+days10_1+minutes49_2'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)    

        self.rnn2 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 1)

        self.rnn3 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 2)

        self.pointwise1 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        self.pointwise3 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 250, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        for p in self.rnn1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.linear1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))


        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn2(x_day.permute(1,0,2))

        x_last_day,_ = self.rnn3(x[:,:,-49:].permute(1,0,2))

        x_signal = self.pointwise1(x_signal[-15:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-3:,:,:].permute(1,0,2)).squeeze(1)
        x_last_day = self.pointwise3(x_last_day[-15:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day + x_last_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_72(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_72,self).__init__()
        self.model_name = 'rnn_sxy_2_72:minutes490_fc+days10_fc+minutes49_fc'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)    

        self.rnn2 = nn.LSTM(input_size = 3, hidden_size = 20, num_layers = 1)

        self.rnn3 = nn.LSTM(input_size = 3, hidden_size = 100, num_layers = 2)

        self.pointwise1 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        self.pointwise3 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 250, out_features= 20)
        self.linear2 = nn.Linear(in_features= 20, out_features= 20)
        self.linear3 = nn.Linear(in_features= 100, out_features= 20)
        self.end_layer = nn.Linear(20,1)
        self.act = nn.ReLU()

        for p in self.rnn1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))


        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn2(x_day.permute(1,0,2))

        x_last_day,_ = self.rnn3(x[:,:,-49:].permute(1,0,2))

        x_signal = self.pointwise1(x_signal[-15:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-3:,:,:].permute(1,0,2)).squeeze(1)
        x_last_day = self.pointwise3(x_last_day[-10:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = self.linear1(x_signal)
        x_day = self.linear2(x_day)
        x_last_day = self.linear3(x_last_day)

        x_signal = x_signal + x_day + x_last_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.dropout(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_73(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_73,self).__init__()
        self.model_name = 'rnn_sxy_2_73:(minutes490+days10+minutes49)cat_fc'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 250, num_layers = 3)    

        self.rnn2 = nn.LSTM(input_size = 3, hidden_size = 20, num_layers = 1)

        self.rnn3 = nn.LSTM(input_size = 3, hidden_size = 100, num_layers = 2)

        self.pointwise1 = nn.Conv1d(in_channels=15, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        self.pointwise3 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 370, out_features= 200)

        self.end_layer = nn.Linear(200,1)
        self.act = nn.ReLU()

        for p in self.rnn1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))


        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn2(x_day.permute(1,0,2))

        x_last_day,_ = self.rnn3(x[:,:,-49:].permute(1,0,2))

        x_signal = self.pointwise1(x_signal[-15:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-3:,:,:].permute(1,0,2)).squeeze(1)
        x_last_day = self.pointwise3(x_last_day[-10:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = self.dropout(x_signal)
        x_signal = torch.cat([x_signal, x_day, x_last_day],axis = 1)

        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        # out = torch.mean(out, axis = 1)
        # x_signal = torch.mean(x_signal, axis = 1)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_74(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_74,self).__init__()
        self.model_name = 'rnn_sxy_2_74:daily'
        
        self.rnn1 = nn.LSTM(input_size = 3, hidden_size = 100)
        self.rnn2 = nn.LSTM(input_size = 100, hidden_size = 50)
        self.rnn3 = nn.LSTM(input_size = 50, hidden_size = 30) 

        self.rnn4 = nn.LSTM(input_size = 3, hidden_size = 50)
        self.rnn5 = nn.LSTM(input_size = 50, hidden_size = 30)

        self.rnn6 = nn.LSTM(input_size = 3, hidden_size = 100)
        self.rnn7 = nn.LSTM(input_size = 100, hidden_size = 30)

        self.pointwise1 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.pointwise3 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 30, out_features= 250)
        self.linear2 = nn.Linear(in_features= 250, out_features= 30)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        for p in self.rnn1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn4.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn5.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn6.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn7.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.linear1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)            
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))
        x_signal,_ = self.rnn2(x_signal[:,:,:])
        x_signal,_ = self.rnn3(x_signal[:,:,:])

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn4(x_day.permute(1,0,2))
        x_day,_ = self.rnn5(x_day)

        x_last_day,_ = self.rnn6(x[:,:,-49:].permute(1,0,2))
        x_last_day,_ = self.rnn7(x_last_day[:,:,:])


        x_signal = self.pointwise1(x_signal[-20:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-4:,:,:].permute(1,0,2)).squeeze(1)
        x_last_day = self.pointwise3(x_last_day[-10:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day + x_last_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_75(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_sxy_2_75,self).__init__()
        self.model_name = 'rnn_sxy_2_75:daily'
        
        self.rnn1 = nn.GRU(input_size = 3, hidden_size = 100)
        self.rnn2 = nn.GRU(input_size = 100, hidden_size = 50)
        self.rnn3 = nn.GRU(input_size = 50, hidden_size = 30) 

        self.rnn4 = nn.GRU(input_size = 3, hidden_size = 50)
        self.rnn5 = nn.GRU(input_size = 50, hidden_size = 30)

        self.rnn6 = nn.GRU(input_size = 3, hidden_size = 100)
        self.rnn7 = nn.GRU(input_size = 100, hidden_size = 30)

        self.pointwise1 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)
        self.pointwise2 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.pointwise3 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)

        self.dropout = nn.Dropout(0.1)     
        self.linear1 = nn.Linear(in_features= 30, out_features= 250)
        self.linear2 = nn.Linear(in_features= 250, out_features= 30)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        for p in self.rnn1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn4.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn5.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn6.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.rnn7.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.pointwise1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.pointwise3.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.linear1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.linear2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)            
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal,_ = self.rnn1(x[:,:,:].permute(1,0,2))
        x_signal,_ = self.rnn2(x_signal[:,:,:])
        x_signal,_ = self.rnn3(x_signal[:,:,:])

        x_day = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x_day = torch.mean(x_day, axis = 2)

        x_day,_ = self.rnn4(x_day.permute(1,0,2))
        x_day,_ = self.rnn5(x_day)

        x_last_day,_ = self.rnn6(x[:,:,-49:].permute(1,0,2))
        x_last_day,_ = self.rnn7(x_last_day[:,:,:])


        x_signal = self.pointwise1(x_signal[-20:,:,:].permute(1,0,2)).squeeze(1)
        x_day = self.pointwise2(x_day[-4:,:,:].permute(1,0,2)).squeeze(1)
        x_last_day = self.pointwise3(x_last_day[-10:,:,:].permute(1,0,2)).squeeze(1)

        x_signal = x_signal + x_day + x_last_day
        x_signal = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear1(x_signal)

        out = self.dropout(x_signal)
        x_signal = self.act(x_signal)
        x_signal = self.linear2(x_signal)
        x_signal = self.act(x_signal)

        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_76(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_76,self).__init__()
        self.model_name = 'rnn_sxy_76: PASScell_1+RNN+pointwise'

        self.lstm = PASScell_1(20,30)
        self.rnn = RNNcell(3,20)

        self.pointwise = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()

        # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
        self.rnn_h_t = torch.zeros((1200, 20)).cuda() 
        self.lstm_h_t = torch.zeros((1200, 30)).cuda()      
        self.lstm_c_t = torch.zeros((1200, 30)).cuda()  

        for p in self.rnn.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.lstm.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

        for p in self.pointwise.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)
        for p in self.end_layer.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):
        x = x.reshape((x.shape[0], 10, x.shape[1]//10, x.shape[-1]))
        x = torch.mean(x, axis = 2)  
        if x.shape[0] != self.rnn_h_t.shape[0]:
            # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
            self.rnn_h_t = torch.zeros((x.shape[0], 20)).cuda()             
            self.lstm_h_t = torch.zeros((x.shape[0], 30)).cuda() 
            self.lstm_c_t = torch.zeros((x.shape[0], 30)).cuda() 
        self.rnn_h_t = self.rnn_h_t.detach()
        self.lstm_h_t = self.lstm_h_t.detach()
        self.lstm_c_t = self.lstm_c_t.detach()
        rnn_out, self.rnn_h_t = self.rnn(x[:,0,:],self.rnn_h_t)
        self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
        temp_t = self.lstm_c_t.unsqueeze(1)
        for i in range(1,x.shape[1]):
            rnn_out, self.rnn_h_t = self.rnn(x[:,i,:],self.rnn_h_t)
            self.lstm_c_t, self.lstm_h_t = self.lstm(rnn_out, self.lstm_c_t, self.lstm_h_t)    
            temp_t = torch.cat([temp_t,self.lstm_c_t.unsqueeze(1)], axis = 1)

        x_signal = temp_t[:,-10:,:]
        x_signal = self.pointwise(x_signal.unsqueeze(-1)).squeeze(-1).squeeze(1)
        out = self.end_layer(x_signal)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnnmultilayers(BasicModule):
    """
    rnnmultilayers seires connection 2nd version:
    
    Parameters:
    
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of RNN layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in cells

    Input:
        A tensor of size (B, T, C, 1) or (1, T, C, B) for convRNNcell
        A tensor of size (B, T, C) for linearRNNcell 
        
    Input for cells:
        use init_hidden to initialize cells. Need to reshape the x into the following form:
            (0)Convcells: init_hidden(1,B) input(1,T[i],C,B) or init_hidden(B,1) input(B,T[i],C,1)
            (1)Linearcells: init_hidden(B) input(B,T[i],C)   
            ...  
    """

    def __init__(self, cells, input_dim, hidden_dim, kernel_size, batch_size = 1200, 
                       num_layers = 1, pointwise_num = 5,bias=True):
        super(rnnmultilayers, self).__init__()
        self.model_name = cells + str(kernel_size) + str(hidden_dim)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.cells = eval(cells)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.pointwise_num = pointwise_num
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(self.cells(input_dim=cur_input_dim,
                                        hidden_dim=self.hidden_dim[i],
                                        kernel_size=self.kernel_size[i],
                                        bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list).cuda()

        self.hidden_states, self.hidden_types = self._init_hidden(batch_size= self.batch_size)

        self.pointwise = nn.Conv1d(in_channels= self.pointwise_num, out_channels=1, kernel_size=1)
        self.end_layer = nn.Linear(self.hidden_dim[-1],1)
        self.act = nn.ReLU()


    def forward(self, x):
        """
        Parameters
        ----------
        x: input tensor todo
            4-D tensor (B, T, C, 1) or (1, T, C, B) for convRNNcell
            3-D tesor of size (B, T, C) for linearRNNcell 
        Returns
        -------
        (B,1) batchsize section
        """
        b,t,c = x.size()
        if b != self.batch_size:
            self.batch_size = b

            # Implement stateful rnn
            self.hidden_states, self.hidden_types = self._init_hidden(batch_size= self.batch_size)

        self._detach_hidden()

        layer_output_list = []

        seq_len = x.size(1)
        cur_layer_input = x

        for layer_idx in range(self.num_layers):

            (c, h) = self.hidden_states[layer_idx]

            if self.hidden_types[layer_idx] == 'no_convolution_time_series':
                pass
            elif self.hidden_types[layer_idx] == 'all_batch_one_kernel_cross_section':
                if cur_layer_input.shape[-1] == self.batch_size:   
                    cur_layer_input = cur_layer_input.permute(3,1,2,0)  
                else:           
                    cur_layer_input = cur_layer_input.squeeze(-1).unsqueeze(-1)
            elif self.hidden_types[layer_idx] == 'one_batch_cross_section':
                if cur_layer_input.shape[-1] != self.batch_size:
                    cur_layer_input = cur_layer_input.squeeze(-1).unsqueeze(-1).permute(3,1,2,0)

            hidden_state_output = []

            for t in range(seq_len):

                y, (c, h) = self.cell_list[layer_idx](x = cur_layer_input[:, t], cur_state = (c, h))
                # print('&&&',y.shape,h.shape)
                hidden_state_output.append(y)

            layer_output = torch.stack(hidden_state_output, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)


        layer_output_list = layer_output_list[-1]

        if layer_output_list.shape[-1] == self.batch_size:
            layer_output_list = layer_output_list.squeeze(0).permute(2,0,1)
        x_signal = self.pointwise(layer_output_list[:,-self.pointwise_num:,:].squeeze(-1)).squeeze(1)
        out = self.act(x_signal)
        out = self.end_layer(out)
        
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


    def _init_hidden(self, batch_size):
        hidden_states = []
        hidden_types = []
        for i in range(self.num_layers):

            if self.kernel_size[i] == 1:
                hidden_type = 'all_batch_one_kernel'
                hidden_state = self.cell_list[i].init_hidden(batch_size = batch_size, section_size = 1)
            elif self.kernel_size[i] == None:
                hidden_type = 'no_convolution'
                hidden_state = self.cell_list[i].init_hidden(batch_size = batch_size, section_size = None)
            else:
                hidden_type = 'one_batch'
                hidden_state = self.cell_list[i].init_hidden(batch_size = 1, section_size = batch_size)

            if len(hidden_state[1].shape) == 2:
                hidden_type += '_time_series'
            else:
                hidden_type += '_cross_section'

            hidden_states.append(hidden_state)
            hidden_types.append(hidden_type)
        return hidden_states, hidden_types


    def _detach_hidden(self):
        for i in range(self.num_layers):
            if self.hidden_states[i][0] != None:
                self.hidden_states[i] = (self.hidden_states[i][0].detach(),self.hidden_states[i][1].detach())
            else:
                self.hidden_states[i] = (None,self.hidden_states[i][1].detach())


    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
