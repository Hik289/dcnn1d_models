
from models.BasicModule import BasicModule
from .rnn_cell_sxy.LSTMcell_sxy import LSTMcell
from .rnn_cell_sxy.GRUcell_sxy import GRUcell
from .rnn_cell_sxy.RNNcell_sxy import RNNcell
from .rnn_cell_sxy.LSTMCcell_sxy import LSTMCcell

from .rnn_cell_sxy.PASScell_1_sxy import PASScell_1
from .rnn_cell_sxy.PASScell_2_sxy import PASScell_2
import torch.nn as nn
import torch
class rnn_sxy_1(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_1,self).__init__()
        self.model_name = 'rnn_sxy_1'

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


class rnn_sxy_1_false(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_1_false,self).__init__()
        self.model_name = 'rnn_sxy_1_false'

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



class rnn_sxy_2(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2,self).__init__()
        self.model_name = 'rnn_sxy_2'

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
        out = self.end_layer(x_signal)
        self.model_out['y_pred'] = out
        self.model_out['signals'] = x_signal
        return self.model_out


class rnn_sxy_2_false(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_2_false,self).__init__()
        self.model_name = 'rnn_sxy_2_false'

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

# class rnn_sxy_3(BasicModule):
#     def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
#         super(rnn_sxy_3,self).__init__()
#         self.model_name = 'rnn_sxy_3'

#         self.lstm = LSTMcell(input_length = 3, hidden_length = 20)

#         self.end_layer = nn.Linear(20,1)
#         self.act = nn.ReLU()

#         # self.h_t = torch.zeros((1200, 20), device = torch.device('cuda:4'))
#         self.h_t = torch.zeros((1200, 20)).cuda()       

#         # for p in self.rnn.parameters():
#         #     nn.init.normal_(p,mean=0.0,std = 0.001)
#         # for p in self.linear.parameters():
#         #     nn.init.normal_(p,mean=0.0,std = 0.001)
#         # for p in self.end_layer.parameters():
#         #     nn.init.normal_(p,mean=0.0,std = 0.001)

#     def forward(self,x):
#         if x.shape[0] != self.h_t.shape[0]:
#             # self.h_t = torch.zeros((x.shape[0], 20),device = 'cuda:4')
#             self.h_t = torch.zeros((x.shape[0], 20)).cuda()
#         self.h_t = self.h_t.detach()
#         i_t = self.i_t(x[:,0,:], self.h_t)
#         f_t = self.f_t(x[:,0,:], self.h_t)
#         g_t = self.g_t(x[:,0,:], self.h_t)
#         o_t = self.o_t(x[:,0,:], self.h_t)
#         c_t = i_t*self.act(g_t)
#         self.h_t = o_t*self.act(c_t)
#         temp_t = c_t.unsqueeze(1)
#         for i in range(1,x.shape[1]):
#             i_t = self.i_t(x[:,i,:], self.h_t)
#             f_t = self.f_t(x[:,i,:], self.h_t)
#             g_t = self.g_t(x[:,i,:], self.h_t)
#             o_t = self.o_t(x[:,i,:], self.h_t)
#             c_t = f_t*c_t + i_t*self.act(g_t)
#             self.h_t = (o_t*self.act(c_t))
#             temp_t = torch.cat([temp_t,c_t.unsqueeze(1)], axis = 1)
#         x_signal = temp_t[:,-1,:]
#         out = self.end_layer(x_signal)
#         self.model_out['y_pred'] = out
#         self.model_out['signals'] = x_signal
#         return self.model_out


class rnn_sxy_4(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_4,self).__init__()
        self.model_name = 'rnn_sxy_4'

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


class rnn_sxy_4_false(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_4_false,self).__init__()
        self.model_name = 'rnn_sxy_4_false'

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


class rnn_sxy_5(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_5,self).__init__()
        self.model_name = 'rnn_sxy_5'

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



class rnn_sxy_6(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_6,self).__init__()
        self.model_name = 'rnn_sxy_6'

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


class rnn_sxy_7(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_7,self).__init__()
        self.model_name = 'rnn_sxy_7'

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


class rnn_sxy_8(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_8,self).__init__()
        self.model_name = 'rnn_sxy_8'

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


class rnn_sxy_9(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_9,self).__init__()
        self.model_name = 'rnn_sxy_9: PASScell_1'

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


class rnn_sxy_10(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_10,self).__init__()
        self.model_name = 'rnn_sxy_10: LSTM+RNN'

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


class rnn_sxy_11(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_11,self).__init__()
        self.model_name = 'rnn_sxy_11: PASScell_2'

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


class rnn_sxy_12(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_12,self).__init__()
        self.model_name = 'rnn_sxy_12: LSTMC_fc'

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


class rnn_sxy_13(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_13,self).__init__()
        self.model_name = 'rnn_sxy_13: LSTMC_pointwise'

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


class rnn_sxy_14(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_14,self).__init__()
        self.model_name = 'rnn_sxy_14: LSTM+RNN+pointwise'

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


class rnn_sxy_15(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_15,self).__init__()
        self.model_name = 'rnn_sxy_15: LSTMC_pointwise'

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


class rnn_sxy_16(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_16,self).__init__()
        self.model_name = 'rnn_sxy_16: LSTM+GRU+pointwise'

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


class rnn_sxy_17(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_17,self).__init__()
        self.model_name = 'rnn_sxy_17: LSTM+LSTM+pointwise'

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


class rnn_sxy_18(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_18,self).__init__()
        self.model_name = 'rnn_sxy_18: LSTM+LSTMC+pointwise'

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


class rnn_sxy_19(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_19,self).__init__()
        self.model_name = 'rnn_sxy_19: LSTM+PASScell_1+pointwise'

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


class rnn_sxy_20(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_20,self).__init__()
        self.model_name = 'rnn_sxy_20: LSTM+PASScell_2+pointwise'

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


class rnn_sxy_21(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_21,self).__init__()
        self.model_name = 'rnn_sxy_21: LSTMC+RNN+pointwise'

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


class rnn_sxy_22(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_22,self).__init__()
        self.model_name = 'rnn_sxy_22: LSTMC+GRU+pointwise'

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


class rnn_sxy_23(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_23,self).__init__()
        self.model_name = 'rnn_sxy_23: LSTMC+LSTM+pointwise'

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


class rnn_sxy_24(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_24,self).__init__()
        self.model_name = 'rnn_sxy_24: LSTMC+LSTMC+pointwise'

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


class rnn_sxy_25(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_25,self).__init__()
        self.model_name = 'rnn_sxy_25: LSTMC+PASScell_1+pointwise'

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


class rnn_sxy_26(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_26,self).__init__()
        self.model_name = 'rnn_sxy_26: LSTMC+PASScell_2+pointwise'

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


class rnn_sxy_27(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_27,self).__init__()
        self.model_name = 'rnn_sxy_27: RNN+RNN+pointwise'

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


class rnn_sxy_28(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_28,self).__init__()
        self.model_name = 'rnn_sxy_28: RNN+GRU+pointwise'

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


class rnn_sxy_29(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_29,self).__init__()
        self.model_name = 'rnn_sxy_29: RNN+LSTM+pointwise'

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


class rnn_sxy_30(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_30,self).__init__()
        self.model_name = 'rnn_sxy_30: RNN+LSTMC+pointwise'

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


class rnn_sxy_31(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_31,self).__init__()
        self.model_name = 'rnn_sxy_31: RNN+PASScell_1+pointwise'

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


class rnn_sxy_32(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_32,self).__init__()
        self.model_name = 'rnn_sxy_32: RNN+PASScell_2+pointwise'

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


class rnn_sxy_33(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_33,self).__init__()
        self.model_name = 'rnn_sxy_33: GRU+RNN+pointwise'

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


class rnn_sxy_34(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_34,self).__init__()
        self.model_name = 'rnn_sxy_34: GRU+GRU+pointwise'

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


class rnn_sxy_35(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_35,self).__init__()
        self.model_name = 'rnn_sxy_35: GRU+LSTM+pointwise'

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


class rnn_sxy_36(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_36,self).__init__()
        self.model_name = 'rnn_sxy_36: GRU+LSTMC+pointwise'

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


class rnn_sxy_37(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_37,self).__init__()
        self.model_name = 'rnn_sxy_37: GRU+PASScell_1+pointwise'

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


class rnn_sxy_38(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_38,self).__init__()
        self.model_name = 'rnn_sxy_38: GRU+PASScell_2+pointwise'

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


class rnn_sxy_39(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_39,self).__init__()
        self.model_name = 'rnn_sxy_39: PASScell_1+RNN+pointwise'

        self.lstm = PASScell_1(20,30)
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


class rnn_sxy_40(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_40,self).__init__()
        self.model_name = 'rnn_sxy_40: PASScell_1+GRU+pointwise'

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


class rnn_sxy_41(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_41,self).__init__()
        self.model_name = 'rnn_sxy_41: PASScell_1+LSTM+pointwise'

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


class rnn_sxy_42(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_42,self).__init__()
        self.model_name = 'rnn_sxy_42: PASScell_1+LSTMC+pointwise'

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


class rnn_sxy_43(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_43,self).__init__()
        self.model_name = 'rnn_sxy_43: PASScell_1+PASScell_1+pointwise'

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


class rnn_sxy_44(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_44,self).__init__()
        self.model_name = 'rnn_sxy_44: PASScell_1+PASScell_2+pointwise'

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


class rnn_sxy_45(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_45,self).__init__()
        self.model_name = 'rnn_sxy_45: PASScell_2+RNN+pointwise'

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


class rnn_sxy_46(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_46,self).__init__()
        self.model_name = 'rnn_sxy_46: PASScell_2+GRU+pointwise'

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


class rnn_sxy_47(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_47,self).__init__()
        self.model_name = 'rnn_sxy_47: PASScell_2+LSTM+pointwise'

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


class rnn_sxy_48(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_48,self).__init__()
        self.model_name = 'rnn_sxy_48: PASScell_2+LSTMC+pointwise'

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


class rnn_sxy_49(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_49,self).__init__()
        self.model_name = 'rnn_sxy_49: PASScell_2+PASScell_1+pointwise'

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


class rnn_sxy_50(BasicModule):
    def __init__(self,input_dim=482,hidden_size = 30, num_layers = 1, dropout = 0):
        super(rnn_sxy_50,self).__init__()
        self.model_name = 'rnn_sxy_50: PASScell_2+PASScell_2+pointwise'

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