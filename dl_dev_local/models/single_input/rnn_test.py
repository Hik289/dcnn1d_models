
from models.BasicModule import BasicModule
import torch.nn as nn
import torch

class rnn_test(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_test,self).__init__()
        self.model_name = 'rnn_test_simple_5'
        
        self.rnn = nn.LSTM(input_size = 3, hidden_size = 30)

        self.linear = nn.Linear(in_features= 30, out_features= 30)
        self.dropout = nn.Dropout(0.1)
        self.end_layer = nn.Linear(30,1)
        self.act = nn.ReLU()


    def forward(self,x):

        x_signal,_ = self.rnn(x.permute(1,0,2))

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


class rnn_4layer(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_4layer,self).__init__()
        self.model_name = 'rnn_4layer'
        
        self.rnn1 = nn.LSTM(input_size = input_dim, hidden_size = 250)
        self.rnn2 = nn.LSTM(input_size = 250, hidden_size = 200)
        self.rnn3 = nn.LSTM(input_size = 200, hidden_size = 150)
        self.rnn4 = nn.LSTM(input_size = 150, hidden_size = 100)
     
        self.linear = nn.Linear(in_features= 100, out_features= 100)
        self.dropout = nn.Dropout(0.2)
        self.end_layer = nn.Linear(100,1)
        # self.act = nn.ReLU()
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


class rnn_3layer(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_3layer,self).__init__()
        self.model_name = 'rnn_3layer'
        
        self.rnn1 = nn.LSTM(input_size = input_dim, hidden_size = 250)
        self.rnn2 = nn.LSTM(input_size = 250, hidden_size = 200)
        self.rnn3 = nn.LSTM(input_size = 200, hidden_size = 150)
     
        self.linear = nn.Linear(in_features= 150, out_features= 150)
        self.dropout = nn.Dropout(0.2)
        self.end_layer = nn.Linear(150,1)
        # self.act = nn.ReLU()
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




class rnn_2layer_fc(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_2layer_fc,self).__init__()
        self.model_name = 'rnn_2layer_fc'
        
        self.rnn1 = nn.LSTM(input_size = input_dim, hidden_size = 250)
        self.rnn2 = nn.LSTM(input_size = 250, hidden_size = 10)
        # self.rnn3 = nn.LSTM(input_size = 200, hidden_size = 150)
        self.dropout = nn.Dropout(0.2)     
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
        self.model_out['y_pred'] = out[-1,:,:]
        self.model_out['signals'] = x_signal[-1,:,:]
        return self.model_out


class rnn_4layer_fc(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_4layer_fc,self).__init__()
        self.model_name = 'rnn_4layer_fc'
        
        self.rnn1 = nn.LSTM(input_size = input_dim, hidden_size = 250)
        self.rnn2 = nn.LSTM(input_size = 250, hidden_size = 150)
        self.rnn3 = nn.LSTM(input_size = 150, hidden_size = 50)
        self.rnn4 = nn.LSTM(input_size = 50, hidden_size = 10)
     
        self.linear1 = nn.Linear(in_features= 10, out_features= 20)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(in_features= 20, out_features= 100)
        self.end_layer = nn.Linear(100,1)
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
        self.model_out['y_pred'] = out[-1,:,:]
        self.model_out['signals'] = x_signal[-1,:,:]
        return self.model_out


class rnn_3layerGRU_fc(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(rnn_3layerGRU_fc,self).__init__()
        self.model_name = 'rnn_3layerGRU_fc'
        
        self.rnn1 = nn.GRU(input_size = input_dim, hidden_size = 250)
        self.rnn2 = nn.GRU(input_size = 250, hidden_size = 150)
        self.rnn3 = nn.GRU(input_size = 150, hidden_size = 50)
     
        self.linear1 = nn.Linear(in_features= 50, out_features= 100)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(in_features= 100, out_features= 100)
        self.end_layer = nn.Linear(100,1)
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
        self.model_out['y_pred'] = out[-1,:,:]
        self.model_out['signals'] = x_signal[-1,:,:]
        return self.model_out