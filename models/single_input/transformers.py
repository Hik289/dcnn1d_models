from models.BasicModule import BasicModule
import torch.nn as nn
import torch

class transformer_sxy_1(BasicModule):
    def __init__(self, input_dim = 12,nhead = 1, num_layers = 1, dim_feedforward = 2048,num_encoders = 1, 
                            batch_size = 1024, resnet = False,dropout = 0.1):
        super(transformer_sxy_1,self).__init__()
        self.model_name = 'transformer_sxy_1: BERT:encoder_1layer_no_resnet'

        self.resnet = resnet
        self.d_model = input_dim
        module_list = []
        for i in range(num_encoders):
            encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead= nhead, dim_feedforward= dim_feedforward)
            module_list.append(nn.TransformerEncoder(encoder_layer, num_layers= num_layers))
        self.module_list = nn.ModuleList(module_list).cuda()

        self.cls_token = torch.rand(batch_size, self.d_model).cuda()
        self.sep_token = torch.rand(batch_size, self.d_model).cuda()

        self.linear1 = nn.Linear(in_features= self.d_model, out_features= 100)
        self.linear2 = nn.Linear(in_features= 100, out_features= 1)
        self.dropout = nn.Dropout(dropout)

        torch.set_num_threads(1)

        # for p in self.module_list.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.1)
        # for p in self.linear1.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.1)
        # for p in self.linear2.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.1)


    def forward(self,x):
        if x.shape[0] != self.sep_token.shape[0]:
            self.cls_token = torch.rand(x.shape[0], self.d_model).cuda()
            self.sep_token = torch.rand(x.shape[0], self.d_model).cuda()            
        self.cls_token = self.cls_token.detach()
        self.sep_token = self.sep_token.detach()

        x_signal = torch.cat([self.cls_token.unsqueeze(1), x, self.sep_token.unsqueeze(1)], dim = 1)
        x_signal = self._positional_encoding(x_signal, d_model = x.shape[-1], max_len = 500)

        for module in self.module_list:
            if self.resnet ==  True:
                x_signal = module(x_signal) + x_signal
            else:
                x_signal = module(x_signal)

        self.cls_token = x_signal[:,0,:]
        self.sep_token = x_signal[:,-1,:]

        out = self.linear1(self.sep_token)
        # out = self.dropout(out)
        out = self.linear2(out)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = self.sep_token
        return self.model_out

    def _positional_encoding(self,x, d_model = 3, max_len= 500):

        pe = torch.zeros(max_len, d_model).cuda()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.Tensor([10000.0])) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            div_term = div_term[:-1]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        x = x + torch.autograd.Variable(pe[:, :x.size(1)], requires_grad=False).cuda()
        return x


class transformer_sxy_2_init(BasicModule):
    def __init__(self, input_dim = 12,nhead = 1, num_layers = 1, dim_feedforward = 2048,num_encoders = 1, 
                            batch_size = 1024, resnet = False,dropout = 0.1):
        super(transformer_sxy_2_init,self).__init__()
        self.model_name = 'transformer_sxy_2: BERT:no positionalencoding init'

        self.resnet = resnet
        self.d_model = input_dim
        module_list = []
        for i in range(num_encoders):
            encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead= nhead, dim_feedforward= dim_feedforward)
            module_list.append(nn.TransformerEncoder(encoder_layer, num_layers= num_layers))
        self.module_list = nn.ModuleList(module_list).cuda()

        self.cls_token = torch.rand(batch_size, self.d_model).cuda()
        self.sep_token = torch.rand(batch_size, self.d_model).cuda()

        self.linear1 = nn.Linear(in_features= self.d_model, out_features= 100)
        self.linear2 = nn.Linear(in_features= 100, out_features= 1)
        self.dropout = nn.Dropout(dropout)

        torch.set_num_threads(1)

        for p in self.module_list.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.1)
        for p in self.linear1.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.1)
        for p in self.linear2.parameters():
            nn.init.normal_(p,mean=0.0,std = 0.1)


    def forward(self,x):
        if x.shape[0] != self.sep_token.shape[0]:
            self.cls_token = torch.rand(x.shape[0], self.d_model).cuda()
            self.sep_token = torch.rand(x.shape[0], self.d_model).cuda()            
        self.cls_token = self.cls_token.detach()
        self.sep_token = self.sep_token.detach()

        x_signal = torch.cat([self.cls_token.unsqueeze(1), x, self.sep_token.unsqueeze(1)], dim = 1)
        #x_signal = self._positional_encoding(x_signal, d_model = x.shape[-1], max_len = 500)

        for module in self.module_list:
            if self.resnet ==  True:
                x_signal = module(x_signal) + x_signal
            else:
                x_signal = module(x_signal)

        self.cls_token = x_signal[:,0,:]
        self.sep_token = x_signal[:,-1,:]

        out = self.linear1(self.sep_token)
        # out = self.dropout(out)
        out = self.linear2(out)

        self.model_out['y_pred'] = out
        self.model_out['signals'] = self.sep_token
        return self.model_out

    def _positional_encoding(self,x, d_model = 3, max_len= 500):

        pe = torch.zeros(max_len, d_model).cuda()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.Tensor([10000.0])) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            div_term = div_term[:-1]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        x = x + torch.autograd.Variable(pe[:, :x.size(1)], requires_grad=False).cuda()
        return x